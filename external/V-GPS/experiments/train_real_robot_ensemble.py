# Training loop adapted from V-GPS (https://github.com/nakamotoo/V-GPS) experiments/train.py.
"""Ensemble finetuning on rollout caches (thesis: successful SAFE WidowX / LIBERO rollouts, pool ``finetune``).

(The loader also supports a real-robot demonstration dataset from preliminary experiments that are not part of the thesis.)
Original description for that dataset:

Finetunes a Bridge/Fractal SARSA ensemble on the 307 successful teleoperated
demonstrations -- the same demonstrations the GR00T policy was finetuned on, so the critic
monitors that policy rather than a different one. Both action spaces are 7-D
(x, y, z, rx, ry, rz, gripper), so weights transfer without the action-dim surgery the IVA
path needs; the auto-detection below is kept and should report 7 -> 7.

No negative demonstrations: every normal scene contains both cups and all three circles, so a
swapped instruction is still satisfiable and would teach the critic to reject feasible tasks.
The per-step cost therefore has to stay -- with neither negatives nor a step cost every member
fits Q = 0 and both readings of the ensemble collapse.

Structure follows train_iva_ensemble.py; only the dataset construction differs.

Run via slurm/train-real-robot-ensemble.sbatch.
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
import flax
from flax.training import checkpoints
from ml_collections import config_flags

from jaxrl_m.agents import agents
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors
import wandb
from jax.experimental.compilation_cache import compilation_cache

from octo.data.real_robot import make_real_robot_dataset
from octo.data.dataset import apply_trajectory_transforms, apply_frame_transforms
from flax.core import freeze, unfreeze

# V2 negative demo support (bridge/fractal-style swapping)
from experiments.utils.finetuning import collect_prompt_pool, swap_batch_prompts_tf

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")


def detect_checkpoint_action_dim(checkpoint_path, encoder_dim=512):
    """Detect action dimension from checkpoint by inspecting Dense_0 kernel shape.

    Dense_0 input = encoder_output + action, so action_dim = kernel_shape[1] - encoder_dim
    """
    ckpt = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    if ckpt is None:
        return None

    try:
        kernel = ckpt['state']['params']['modules_critic']['network']['Dense_0']['kernel']
        input_dim = kernel.shape[1]  # (critic_ensemble, input_dim, hidden)
        action_dim = input_dim - encoder_dim
        return action_dim
    except (KeyError, IndexError) as e:
        logging.warning(f"Could not detect action dim from checkpoint: {e}")
        return None


def merge_params(target_params, pretrained_params):
    """Copy pretrained params into target for matching keys + shapes.

    Inspired by Octo's merge_params. Uses flax.traverse_util to flatten,
    copy matching, skip mismatches, then unflatten.
    """
    flat_target = flax.traverse_util.flatten_dict(unfreeze(target_params))
    flat_pretrained = flax.traverse_util.flatten_dict(
        pretrained_params if isinstance(pretrained_params, dict) else unfreeze(pretrained_params)
    )

    # Find keys to copy (matching shape)
    keys_to_copy = [
        k for k in flat_target
        if k in flat_pretrained and flat_target[k].shape == flat_pretrained[k].shape
    ]
    shape_mismatch = [
        k for k in flat_target
        if k in flat_pretrained and flat_target[k].shape != flat_pretrained[k].shape
    ]
    missing = [k for k in flat_target if k not in flat_pretrained]

    # Log mismatches
    if shape_mismatch:
        for k in shape_mismatch:
            logging.info(f"Shape mismatch, skipping: {'.'.join(k)} "
                        f"({flat_pretrained[k].shape} -> {flat_target[k].shape})")
    if missing:
        for k in missing:
            logging.debug(f"Missing in pretrained: {'.'.join(k)}")

    logging.info(f"Copying {len(keys_to_copy)}/{len(flat_target)} params, "
                f"{len(shape_mismatch)} shape mismatches, {len(missing)} missing")

    # Copy matching params
    flat_target = flax.core.copy(
        flat_target, {k: flat_pretrained[k] for k in keys_to_copy}
    )
    return freeze(flax.traverse_util.unflatten_dict(flat_target))


def transfer_weights_with_action_dim_change(new_agent, old_checkpoint_path, old_action_dim, new_action_dim):
    """Transfer weights from checkpoint, copying params with matching shapes.

    For action dim mismatch, layers like Dense_0 (input=[encoder, action])
    will have shape mismatch and stay randomly initialized.
    """
    old_ckpt = checkpoints.restore_checkpoint(old_checkpoint_path, target=None)
    if old_ckpt is None:
        logging.warning(f"No checkpoint at {old_checkpoint_path}")
        return new_agent

    old_state = old_ckpt['state']

    # Merge params and target_params
    merged_params = merge_params(new_agent.state.params, old_state['params'])
    merged_target_params = merge_params(new_agent.state.target_params, old_state['target_params'])

    # Re-init optimizer state on the new param tree to avoid shape/type mismatches
    logging.info("Using fresh optimizer state for finetuning")
    fresh_opt_states = new_agent.state._tx_tree_map(
        lambda tx: tx.init(merged_params), new_agent.state.txs
    )

    new_state = new_agent.state.replace(
        params=merged_params,
        target_params=merged_target_params,
        opt_states=fresh_opt_states,
        step=0,  # reset step for finetuning with new action dimension
    )
    return new_agent.replace(state=new_state)

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "jaxrl_m_real_robot_ensemble", "WandB project name.")

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)


def get_latest_checkpoint_step(ckpt_dir, prefix='checkpoint_'):
    """Extract step number from latest checkpoint folder name."""
    if not tf.io.gfile.exists(ckpt_dir):
        return None

    try:
        checkpoint_files = tf.io.gfile.listdir(ckpt_dir)
        checkpoint_steps = []

        for filename in checkpoint_files:
            if filename.startswith(prefix):
                try:
                    step_str = filename[len(prefix):]
                    step = int(step_str)
                    checkpoint_steps.append(step)
                except ValueError:
                    continue

        if checkpoint_steps:
            return max(checkpoint_steps)
        else:
            return None
    except Exception as e:
        logging.warning(f"Error reading checkpoint directory {ckpt_dir}: {e}")
        return None


def create_ensemble_agents(ensemble_size, rng, template_batch, encoder_def, agent_class, agent_kwargs):
    """Create ensemble agents with vectorized creation."""
    logging.info("Creating ensemble agents...")
    logging.info(f"agent_class {agent_class}")
    logging.info("Using vectorized ensemble agent creation")
    ensemble_agents = agent_class.create_ensemble_vectorized(
        base_rng=rng,
        ensemble_size=ensemble_size,
        observations=template_batch["observations"],
        actions=template_batch["actions"],
        encoder_def=encoder_def,
        goals=template_batch["goals"],
        **agent_kwargs
    )
    return ensemble_agents


def aggregate_ensemble_metrics_cpu(ensemble_metrics):
    """Aggregate metrics that are already on CPU after device_get."""
    def compute_stats(x):
        return {
            'mean': jnp.mean(x, axis=0),
            'std': jnp.std(x, axis=0),
        }

    aggregated = jax.tree_map(compute_stats, ensemble_metrics)

    # Flatten for logging
    flattened = {}
    for metric_name, stats in aggregated.items():
        for stat_name, values in stats.items():
            if stat_name == 'std' and 'ood_q' in metric_name:
                flattened[f"ensemble_disagreement_ood_q"] = values
            elif stat_name == 'std' and 'online_q' in metric_name:
                flattened[f"ensemble_disagreement_online_q"] = values
            elif stat_name == 'std' and 'target_q' in metric_name:
                flattened[f"ensemble_disagreement_target_q"] = values
            else:
                flattened[f"{stat_name}_{metric_name}"] = values

    return flattened


def check_packing(pmapped_update, pack, unpack, agents, batch, ensemble_size, tol):
    """One update both ways -- packed (pmap over devices x vmap over members) and member by
    member on one device -- and compare the parameter updates, so every run's log carries the
    proof that packing several members per GPU changes nothing."""
    packed_agents, packed_info = pmapped_update(pack(agents), pack(batch))
    packed_agents, packed_info = unpack(packed_agents), unpack(packed_info)
    losses = []
    for k in range(ensemble_size):
        agent_k = jax.tree_map(lambda x: x[k], agents)
        new_k, info_k = agent_k.update(jax.tree_map(lambda x: x[k], batch))
        old = jax.tree_util.tree_leaves(agent_k.state.params)
        single = jax.tree_util.tree_leaves(jax.device_get(new_k.state.params))
        packed = jax.tree_util.tree_leaves(jax.tree_map(lambda x: x[k], packed_agents.state.params))
        d_single = np.concatenate([(np.asarray(n) - np.asarray(o)).ravel() for n, o in zip(single, old)])
        d_packed = np.concatenate([(np.asarray(n) - np.asarray(o)).ravel() for n, o in zip(packed, old)])
        rel = float(np.linalg.norm(d_packed - d_single) / np.linalg.norm(d_single))
        loss_single, loss_packed = float(info_k["critic_loss"]), float(packed_info["critic_loss"][k])
        logging.info(f"[packing check] member {k}: critic_loss single {loss_single:.6f} packed "
                     f"{loss_packed:.6f}; |update diff| / |update| = {rel:.2e}")
        if rel > tol or abs(loss_single - loss_packed) > tol * max(1.0, abs(loss_single)):
            raise RuntimeError(f"packed update of member {k} differs from the per-member update "
                               f"(relative {rel:.2e}, losses {loss_single} vs {loss_packed})")
        losses.append(loss_single)
    if len(set(np.round(losses, 6))) != ensemble_size:
        raise RuntimeError(f"members gave identical losses {losses}: not distinct")
    logging.info(f"[packing check] passed for {ensemble_size} members (tol {tol})")


def batch_log_metrics(wandb_logger, aggregated_metrics, step, prefix=""):
    """Batch logging to reduce wandb API calls."""
    log_dict = {}
    for key, value in aggregated_metrics.items():
        log_dict[f"{prefix}{key}"] = value
    wandb_logger.log(log_dict, step=step)


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)

    ensemble_size = FLAGS.config.ensemble_size or num_devices
    batch_size_per_member = FLAGS.config.batch_size // ensemble_size
    # Several members per GPU when fewer GPUs than members are free: pmap over devices, vmap
    # over the members a device holds. Each member still runs its own update on its own slice
    # of the batch, so the result does not depend on the packing
    # (experiments/check_packed_ensemble.py verifies this against a per-member update).
    if ensemble_size % num_devices:
        raise ValueError(f"ensemble_size {ensemble_size} is not a multiple of {num_devices} devices")
    members_per_device = ensemble_size // num_devices
    logging.info(f"{num_devices} devices x {members_per_device} members per device")

    def pack(tree):
        return jax.tree_map(lambda x: x.reshape(num_devices, members_per_device, *x.shape[1:]), tree)

    def unpack(tree):
        return jax.tree_map(lambda x: np.asarray(x).reshape(ensemble_size, *x.shape[2:]),
                            jax.device_get(tree))

    logging.info(f"Ensemble training: {ensemble_size} models, batch size {FLAGS.config.batch_size}, {batch_size_per_member} per member")
    logging.info(f"Real-robot cache: {FLAGS.config.real_robot_data_dir}, split: train")

    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(FLAGS.config.seed)

    # Determine resume step based on checkpoint action dimension
    # Same action dim: resume training, continue from checkpoint step
    # Different action dim: finetune, start from step 0
    resume_step = 0
    resume_path = FLAGS.config.get("resume_path")
    checkpoint_action_dim = None

    if resume_path:
        member_checkpoint_path = os.path.join(resume_path, "ensemble_member_0")
        checkpoint_action_dim = detect_checkpoint_action_dim(member_checkpoint_path)
        logging.info(f"Detected checkpoint action dim: {checkpoint_action_dim}")

    # Setup wandb
    if FLAGS.config.get('resume_wandb_id'):
        wandb_id = FLAGS.config.resume_wandb_id
        if FLAGS.name in wandb_id:
            unique_id = wandb_id.replace(f"{FLAGS.name}_", "", 1)
        else:
            unique_id = wandb_id.split('_')[-1]

        wandb_config = WandBLogger.get_default_config()
        wandb_config.update({
            "project": FLAGS.project,
            "exp_descriptor": f"{FLAGS.name}",
            "unique_identifier": unique_id
        })
        logging.info(f"Resuming wandb run: {wandb_id} with unique_id: {unique_id}")
    else:
        tags = ["ensemble", f"ensemble_size_{ensemble_size}", "real_robot"]
        if FLAGS.config.create_negative_demos:
            tags.extend(["negative_demos", f"negdemo_{FLAGS.config.negative_demo_ratio}"])

        wandb_config = WandBLogger.get_default_config()
        wandb_config.update({
            "project": FLAGS.project,
            "exp_descriptor": f"{FLAGS.name}",
            "tags": tags,
            "group": f"{FLAGS.name}"
        })
        logging.info("Starting new wandb run")

    variant = FLAGS.config.to_dict()
    variant.update({
        "ensemble_size": ensemble_size,
        "batch_size_per_member": batch_size_per_member,
    })

    wandb_logger = WandBLogger(wandb_config=wandb_config, variant=variant)

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # Text processor setup
    text_processor = None
    if FLAGS.config.get("text_processor"):
        text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)

    def process_text(batch, keep_language_str=False):
        decoded_strings = [s.decode("utf-8") for s in batch["goals"]["language"]]
        if text_processor is not None:
            batch["goals"]["language"] = text_processor.encode(decoded_strings)
        if keep_language_str:
            batch["goals"]["language_str"] = decoded_strings
        return batch

    def process_batch(batch, keep_language_str=False, training=False):
        """Preprocess training or validation batch."""
        def reshape_to_ensemble(x):
            return x.reshape(ensemble_size, batch_size_per_member, *x.shape[1:])

        pre_batch = {
            "actions": batch["action"].squeeze(),
            "next_actions": batch["next_action"].squeeze(),
            "goals": {
                "language": batch["task"]["language_instruction"],
            },
            "mc_returns": batch["mc_return"],
            "observations": {"image": batch["observation"]["image_primary"].squeeze()},
            "next_observations": {"image": batch["next_observation"]["image_primary"].squeeze()},
            "rewards": batch["reward"],
            "masks": batch["td_mask"],
        }

        processed_batch = process_text(pre_batch, keep_language_str=keep_language_str)

        if training:
            ensemble_batch = jax.tree_map(reshape_to_ensemble, processed_batch)
        else:
            ensemble_batch = processed_batch

        return ensemble_batch

    logging.info(f"Loading real-robot demonstrations from {FLAGS.config.real_robot_data_dir}")
    logging.info(f"tasks: {FLAGS.config.real_robot_tasks}, "
                 f"frame_stride: {FLAGS.config.real_robot_frame_stride}, "
                 f"step_reward: {FLAGS.config.real_robot_step_reward}, "
                 f"normalization: {FLAGS.config.real_robot_normalization}")

    train_data_raw = make_real_robot_dataset(
        data_dir=FLAGS.config.real_robot_data_dir,
        split="train",
        tasks=FLAGS.config.real_robot_tasks,
        seed=FLAGS.config.seed,
        discount=FLAGS.config.agent_kwargs.discount,
        step_reward=FLAGS.config.real_robot_step_reward,
        frame_stride=FLAGS.config.real_robot_frame_stride,
        num_terminal=FLAGS.config.real_robot_num_terminal,
        val_fraction=FLAGS.config.real_robot_val_fraction,
        normalization=FLAGS.config.real_robot_normalization,
        force_recompute_statistics=FLAGS.config.real_robot_force_recompute_statistics,
        train_pool=FLAGS.config.real_robot_train_pool,
        statistics_path=FLAGS.config.real_robot_statistics_path,
        causal_gripper=FLAGS.config.real_robot_causal_gripper,
        val_pool=FLAGS.config.real_robot_val_pool,
    )

    # Use transform kwargs from config (train_real_robot_config.py)
    traj_transform_kwargs = FLAGS.config.traj_transform_kwargs.to_dict()
    frame_transform_kwargs = FLAGS.config.frame_transform_kwargs.to_dict()

    # Apply trajectory transforms
    train_data = apply_trajectory_transforms(
        train_data_raw,
        train=True,
        **traj_transform_kwargs,
    )

    # Images arrive as encoded PNG bytes from the cache, so apply_frame_transforms
    # decodes and resizes them exactly as it does for the tfrecord datasets.

    # Apply frame transforms (decode and resize images)
    train_data = apply_frame_transforms(
        train_data,
        train=True,
        **frame_transform_kwargs,
    )

    # V2 negative demo setup (bridge/fractal-style per-batch swapping)
    prompt_pool_tensor = None
    if FLAGS.config.create_negative_demos:
        logging.info("V2 negative demos enabled: collecting prompt pool")
        unique_prompts = collect_prompt_pool(train_data, prompt_pool_size=10_000)
        prompt_pool_tensor = tf.constant(unique_prompts)
        logging.info(f"V2: Collected {len(unique_prompts)} unique prompts for batch-level swapping")

    # Batch and shuffle with memory optimization
    train_data = train_data.unbatch().shuffle(1000).repeat().batch(FLAGS.config.batch_size).with_ram_budget(1)

    # Raw training iterator (V2 swapping applied per-batch if enabled)
    train_iter_raw = train_data.iterator(prefetch=0)

    def get_training_batch():
        """Get batch, optionally applying V2 negative demo swapping."""
        batch = next(train_iter_raw)

        if FLAGS.config.create_negative_demos and prompt_pool_tensor is not None:
            # V2: Apply per-batch random swapping (returns all numpy)
            batch = swap_batch_prompts_tf(
                batch, prompt_pool_tensor,
                negative_ratio=FLAGS.config.negative_demo_ratio
            )
            # Mirror create_negative_demonstrations: swapped steps get reward=-1 and td_mask=1.
            # task_feasible=0 means swapped (infeasible), =1 means original
            swapped_mask = (batch["task_feasible"] == 0)
            batch["reward"] = np.where(swapped_mask, -1.0, batch["reward"])
            batch["td_mask"] = np.where(swapped_mask, 1.0, batch["td_mask"])

        return process_batch(batch, training=True)

    # Training iterator using get_training_batch
    # Note: We use a generator to make it work with the training loop
    def train_iterator_gen():
        while True:
            yield get_training_batch()

    train_iterator = train_iterator_gen()

    # Validation setup: always use _fp for eval to test FP detection capability
    # (even when training on _tp only, we want to eval on _fp for OOD detection)
    # Held-out demonstrations. There is no false-premise material in the training pool, so
    # this measures fit, not detection; detection is scored on the rollouts by
    # experiments/eval_real_robot.py.
    val_data_raw = make_real_robot_dataset(
        data_dir=FLAGS.config.real_robot_data_dir,
        split="val",
        tasks=FLAGS.config.real_robot_tasks,
        seed=FLAGS.config.seed,
        discount=FLAGS.config.agent_kwargs.discount,
        step_reward=FLAGS.config.real_robot_step_reward,
        frame_stride=FLAGS.config.real_robot_frame_stride,
        num_terminal=FLAGS.config.real_robot_num_terminal,
        val_fraction=FLAGS.config.real_robot_val_fraction,
        normalization=FLAGS.config.real_robot_normalization,
        force_recompute_statistics=FLAGS.config.real_robot_force_recompute_statistics,
        train_pool=FLAGS.config.real_robot_train_pool,
        statistics_path=FLAGS.config.real_robot_statistics_path,
        causal_gripper=FLAGS.config.real_robot_causal_gripper,
        val_pool=FLAGS.config.real_robot_val_pool,
    )

    val_data = apply_trajectory_transforms(
        val_data_raw,
        train=False,
        **traj_transform_kwargs,
    )

    val_data = apply_frame_transforms(
        val_data,
        train=False,
        **frame_transform_kwargs,
    )

    val_iter = (
        val_data.unbatch()
        .shuffle(1000)
        .repeat()
        .batch(batch_size_per_member)
        .iterator(prefetch=0)
    )
    val_iter = map(process_batch, val_iter)

    # Trajectory data for plotting
    val_traj_data_iter = map(
        partial(process_batch, keep_language_str=True),
        val_data.shuffle(1000).repeat().iterator()
    )

    prev_val_traj = next(val_traj_data_iter)
    logging.info(f"Val traj keys: {list(prev_val_traj.keys())}")
    logging.info(f"Val traj actions shape: {prev_val_traj['actions'].shape}")

    # Get example batch and create template for agent initialization
    example_batch = next(train_iterator)
    template_batch = jax.tree_map(lambda x: x[0], example_batch)

    logging.info(f"Ensemble batch shape: {jax.tree_map(lambda x: x.shape, example_batch)}")
    logging.info(f"Template batch shape: {jax.tree_map(lambda x: x.shape, template_batch)}")

    # Encoder setup
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # Create ensemble agents
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    agent_class = agents[FLAGS.config.agent]
    logging.info(f"rng in create_ensemble_agents is {rng}")
    ensemble_agents = create_ensemble_agents(
        ensemble_size, rng, template_batch, encoder_def, agent_class, FLAGS.config.agent_kwargs
    )

    # Checkpoint restoration with auto-detection of action dimension
    new_action_dim = template_batch["actions"].shape[-1]
    logging.info(f"New model action dim: {new_action_dim}")

    # Where the step counter starts. The IVA script infers this from the action dimension: a
    # matching dim means the same embodiment, hence a continuation of the same run. That
    # inference is wrong here -- Bridge/Fractal and this robot both happen to use 7-D actions,
    # so a matched dim would set the counter to the checkpoint's 200000 and `range(200000,
    # num_steps)` would train for zero steps. Finetuning onto a different dataset restarts the
    # count, so make it an explicit config choice rather than a consequence of a coincidence.
    if resume_path and FLAGS.config.real_robot_reset_step:
        resume_step = 0
        logging.info("resume_step=0: finetuning onto the real-robot demonstrations "
                     f"(checkpoint action dim {checkpoint_action_dim} -> {new_action_dim})")
    elif resume_path and checkpoint_action_dim == new_action_dim:
        member_checkpoint_path = os.path.join(resume_path, "ensemble_member_0")
        resume_step = get_latest_checkpoint_step(member_checkpoint_path) or 0
        logging.info(f"Same action dim, resuming from step: {resume_step}")
    elif resume_path:
        logging.info(f"Action dim mismatch ({checkpoint_action_dim}->{new_action_dim}), starting from step 0")

    if resume_path:
        restored_agents = []
        for member_idx in range(ensemble_size):
            member_checkpoint_path = os.path.join(resume_path, f"ensemble_member_{member_idx}")
            agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)

            if tf.io.gfile.exists(member_checkpoint_path):
                if checkpoint_action_dim != new_action_dim:
                    # Transfer weights with action dim change
                    agent = transfer_weights_with_action_dim_change(
                        agent, member_checkpoint_path,
                        old_action_dim=checkpoint_action_dim,
                        new_action_dim=new_action_dim
                    )
                    logging.info(f"Transferred weights for member {member_idx} ({checkpoint_action_dim}D->{new_action_dim}D)")
                else:
                    # Direct restore (same action dim)
                    agent = checkpoints.restore_checkpoint(member_checkpoint_path, target=agent)
                    logging.info(f"Restored member {member_idx} from step {resume_step}")
            else:
                logging.warning(f"No checkpoint at {member_checkpoint_path}, using random init")
            restored_agents.append(agent)

        ensemble_agents = jax.tree_map(lambda *args: jnp.stack(args, axis=0), *restored_agents)
        logging.info(f"Ensemble shape after restore: {jax.tree_map(lambda x: x.shape, ensemble_agents)}")
    else:
        logging.info("Training from scratch on the real-robot demonstrations")

    # Create pmapped functions (devices) over vmapped members (members_per_device)
    pmapped_update = jax.pmap(jax.vmap(lambda agent, batch: agent.update(batch)))
    pmapped_debug_metrics = jax.pmap(
        jax.vmap(lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key),
                 in_axes=(0, None, 0)),
        in_axes=(0, None, 0),
    )
    if FLAGS.config.check_packing:
        check_packing(pmapped_update, pack, unpack, ensemble_agents, example_batch,
                      ensemble_size, FLAGS.config.check_packing_tol)
    ensemble_agents = pack(ensemble_agents)

    timer = Timer()
    total_steps = int(FLAGS.config.num_steps)
    logging.info(f"Training from step {resume_step} to {total_steps}")

    for i in tqdm.tqdm(range(resume_step, total_steps)):
        timer.tick("total")

        timer.tick("dataset")
        ensemble_batch = next(train_iterator)
        timer.tock("dataset")

        timer.tick("train")
        ensemble_agents, ensemble_update_infos = pmapped_update(ensemble_agents, pack(ensemble_batch))
        timer.tock("train")

        if i % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating ensemble...")
            timer.tick("val")

            val_metrics_list = []
            for _ in range(FLAGS.config.num_val_batches):
                single_val_batch = next(val_iter)
                rng, val_rng = jax.random.split(rng)
                val_rngs = pack(jax.random.split(val_rng, ensemble_size))
                val_metrics = unpack(pmapped_debug_metrics(ensemble_agents, single_val_batch, val_rngs))
                val_metrics_list.append(val_metrics)

            avg_val_metrics_cpu = jax.tree_map(lambda *xs: np.mean(np.stack(xs), axis=0), *val_metrics_list)

            real_disagreement = jnp.mean(jnp.std(avg_val_metrics_cpu["online_q"], axis=0))
            logging.info(f"Aggregated real data disagreement ({FLAGS.config.num_val_batches} batches): {real_disagreement:.3f}")

            aggregated_val_metrics = aggregate_ensemble_metrics_cpu(avg_val_metrics_cpu)

            batch_log_metrics(wandb_logger, aggregated_val_metrics, i, "validation/")

            # Plot value functions (copies the whole ensemble to the host, hence its own cadence)
            assert "sarsa" in FLAGS.config.agent
            plot_now = i % FLAGS.config.plot_interval == 0
            if plot_now:
                logging.info("Plotting ensemble value functions...")
                host_agents = unpack(ensemble_agents)
            for num in range(2 if plot_now else 0):
                traj = next(val_traj_data_iter)
                rng, val_rng = jax.random.split(rng)
                val_rngs = jax.random.split(val_rng, ensemble_size)
                # TODO: The reward_to_go calculation is only correct for -1,-1...,-1,0,0,0 standard reward
                plot = agent_class.plot_ensemble_trajectory_values(host_agents, ensemble_size, traj, val_rngs, show_injection_timeline=False)
                plot = wandb.Image(plot)
                wandb_logger.log({f"ensemble_value_plots/traj_{num}": plot}, step=i)

                prev_val_traj = traj

            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving ensemble checkpoints...")
            host_agents = unpack(ensemble_agents)
            for member_idx in range(ensemble_size):
                member_agent = jax.tree_map(lambda x: x[member_idx], host_agents)
                member_save_dir = os.path.join(save_dir, f"ensemble_member_{member_idx}")
                checkpoints.save_checkpoint(member_save_dir, member_agent, step=i + 1, keep=100)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            ensemble_update_infos_cpu = unpack(ensemble_update_infos)
            aggregated_training_metrics = aggregate_ensemble_metrics_cpu(ensemble_update_infos_cpu)
            batch_log_metrics(wandb_logger, aggregated_training_metrics, i, "training/")
            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


if __name__ == "__main__":
    app.run(main)
