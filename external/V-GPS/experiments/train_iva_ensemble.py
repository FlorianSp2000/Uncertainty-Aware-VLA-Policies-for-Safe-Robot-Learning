# Training loop adapted from V-GPS (https://github.com/nakamotoo/V-GPS) experiments/train.py.
"""Ensemble training on IVA dataset with configurable FP handling.

This script trains/finetunes SARSA ensemble on IVA dataset.
Auto-detects checkpoint vs dataset action dimensions:
- Same dim: direct restore, continue from checkpoint step
- Different dim: transfer weights, start from step 0

Data Loading Strategies (iva_use_tp_only):
- False (default): Load _fp files (~84% TP + sparse FP injections)
- True: Load _tp files (no FP), train clean, eval on _fp for OOD detection

False Premise Strategies (iva_fp_strategy):
- "copy_prev": FP steps reuse the previous step's action (default)
- "zero":      FP steps get a zero action
Reward is -1.0 under both. Inert when iva_use_tp_only=True, since _tp files carry
an action on every step.

Note: Eval always uses _fp files to test FP/OOD detection capability.

Run via slurm/train-iva-ensemble.sbatch or slurm/train-iva-step-reward-ablation.sbatch.
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

from octo.data.iva import make_iva_dataset
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
flags.DEFINE_string("project", "jaxrl_m_iva_ensemble", "WandB project name.")

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

    logging.info(f"Ensemble training: {ensemble_size} models, batch size {FLAGS.config.batch_size}, {batch_size_per_member} per member")
    logging.info(f"IVA data: {FLAGS.config.iva_data_dir}, split: train")

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
        tags = ["ensemble", f"ensemble_size_{ensemble_size}", "iva"]
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

    def process_iva_batch(batch, keep_language_str=False, training=False):
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

    # Load IVA dataset
    use_tp_only = FLAGS.config.iva_use_tp_only
    fp_strategy = FLAGS.config.iva_fp_strategy
    logging.info(f"Loading IVA dataset from {FLAGS.config.iva_data_dir}")
    logging.info(f"IVA tasks: {FLAGS.config.iva_tasks}")
    logging.info(f"IVA use_tp_only: {use_tp_only}, fp_strategy: {fp_strategy}")

    # Load IVA dataset with configurable strategy
    train_data_raw = make_iva_dataset(
        data_dir=FLAGS.config.iva_data_dir,
        split="train",
        tasks=FLAGS.config.iva_tasks,
        seed=FLAGS.config.seed,
        discount=FLAGS.config.agent_kwargs.discount,
        use_tp_only=use_tp_only,
        fp_strategy=fp_strategy,
        step_reward=FLAGS.config.iva_step_reward,
    )

    # Use transform kwargs from config (already defined in train_iva_config.py)
    traj_transform_kwargs = FLAGS.config.traj_transform_kwargs.to_dict()
    frame_transform_kwargs = FLAGS.config.frame_transform_kwargs.to_dict()

    # Apply trajectory transforms
    train_data = apply_trajectory_transforms(
        train_data_raw,
        train=True,
        **traj_transform_kwargs,
    )

    # NOTE: Images are now loaded as bytes directly in make_iva_dataset()
    # (load_images_as_bytes=True by default), so no additional image loading needed

    # Apply frame transforms (decode and resize images)
    train_data = apply_frame_transforms(
        train_data,
        train=True,
        **frame_transform_kwargs,
    )

    # V2 negative demo setup (bridge/fractal-style per-batch swapping)
    prompt_pool_tensor = None
    if FLAGS.config.create_negative_demos:
        logging.info("V2 negative demos enabled: collecting prompt pool from IVA dataset")
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

        return process_iva_batch(batch, training=True)

    # Training iterator using get_training_batch
    # Note: We use a generator to make it work with the training loop
    def train_iterator_gen():
        while True:
            yield get_training_batch()

    train_iterator = train_iterator_gen()

    # Validation setup: always use _fp for eval to test FP detection capability
    # (even when training on _tp only, we want to eval on _fp for OOD detection)
    val_data_raw = make_iva_dataset(
        data_dir=FLAGS.config.iva_data_dir,
        split="eval",
        tasks=FLAGS.config.iva_tasks,
        seed=FLAGS.config.seed,
        discount=FLAGS.config.agent_kwargs.discount,
        use_tp_only=False,  # Always eval on FP for OOD detection
        fp_strategy=fp_strategy,
        step_reward=FLAGS.config.iva_step_reward,
    )

    val_data = apply_trajectory_transforms(
        val_data_raw,
        train=False,
        **traj_transform_kwargs,
    )

    # NOTE: Images are now loaded as bytes directly in make_iva_dataset()

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
    val_iter = map(process_iva_batch, val_iter)

    # Trajectory data for plotting
    val_traj_data_iter = map(
        partial(process_iva_batch, keep_language_str=True),
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

    # Determine resume_step based on action dim comparison
    if resume_path and checkpoint_action_dim == new_action_dim:
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
        logging.info("Training from scratch on IVA dataset")

    # Create pmapped functions
    pmapped_update = jax.pmap(lambda agent, batch: agent.update(batch))
    pmapped_debug_metrics = jax.pmap(
        lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key),
        in_axes=(0, None, 0),
    )

    timer = Timer()
    total_steps = int(FLAGS.config.num_steps)
    logging.info(f"Training from step {resume_step} to {total_steps}")

    for i in tqdm.tqdm(range(resume_step, total_steps)):
        timer.tick("total")

        timer.tick("dataset")
        ensemble_batch = next(train_iterator)
        timer.tock("dataset")

        timer.tick("train")
        ensemble_agents, ensemble_update_infos = pmapped_update(ensemble_agents, ensemble_batch)
        timer.tock("train")

        if i % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating ensemble...")
            timer.tick("val")

            val_metrics_list = []
            for _ in range(6):
                single_val_batch = next(val_iter)
                rng, val_rng = jax.random.split(rng)
                val_rngs = jax.random.split(val_rng, ensemble_size)
                val_metrics = pmapped_debug_metrics(ensemble_agents, single_val_batch, val_rngs)
                val_metrics_list.append(val_metrics)

            avg_val_metrics = jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *val_metrics_list)
            avg_val_metrics_cpu = jax.device_get(avg_val_metrics)

            real_disagreement = jnp.mean(jnp.std(avg_val_metrics_cpu["online_q"], axis=0))
            logging.info(f"Aggregated real data disagreement (6 iterations): {real_disagreement:.3f}")

            aggregated_val_metrics = aggregate_ensemble_metrics_cpu(avg_val_metrics_cpu)

            batch_log_metrics(wandb_logger, aggregated_val_metrics, i, "validation/")

            # Plot value functions
            assert "sarsa" in FLAGS.config.agent
            logging.info("Plotting ensemble value functions...")
            for num in range(2):
                traj = next(val_traj_data_iter)
                rng, val_rng = jax.random.split(rng)
                val_rngs = jax.random.split(val_rng, ensemble_size)
                # TODO: The reward_to_go calculation is only correct for -1,-1...,-1,0,0,0 standard reward
                plot = agent_class.plot_ensemble_trajectory_values(ensemble_agents, ensemble_size, traj, val_rngs, show_injection_timeline=True)
                plot = wandb.Image(plot)
                wandb_logger.log({f"ensemble_value_plots/traj_{num}": plot}, step=i)

                prev_val_traj = traj

            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving ensemble checkpoints...")
            for member_idx in range(ensemble_size):
                member_agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
                member_save_dir = os.path.join(save_dir, f"ensemble_member_{member_idx}")
                checkpoints.save_checkpoint(member_save_dir, member_agent, step=i + 1, keep=100)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            ensemble_update_infos_cpu = jax.device_get(ensemble_update_infos)
            aggregated_training_metrics = aggregate_ensemble_metrics_cpu(ensemble_update_infos_cpu)
            batch_log_metrics(wandb_logger, aggregated_training_metrics, i, "training/")
            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


if __name__ == "__main__":
    app.run(main)
