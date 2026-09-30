# Training loop adapted from V-GPS (https://github.com/nakamotoo/V-GPS) experiments/train.py.
"""
Reward Ablation Experiment for Q-Ensemble Uncertainty Detection

Simplified training script for ablating negative demo reward schemes:
- Bridge-only dataset (not Bridge+Fractal)
- Single validation set: 100% neg-demo (same images, original vs swapped language)
- Metrics: Q-value difference and disagreement ratio across language variants
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags

from jaxrl_m.agents import agents
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors
import wandb
from jax.experimental.compilation_cache import compilation_cache

from octo.data.dataset import make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs

from experiments.utils.finetuning import create_negative_demonstrations
from jaxrl_m.agents.continuous.sarsa_ensemble import SARSAEnsembleAgent

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "reward_ablation", "Experiment name.")
flags.DEFINE_string("project", "q-ensemble-ablations", "WandB project name.")

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("oxedata_config", None, "Data configuration.", lock_config=False)


def create_ensemble_agents(ensemble_size, rng, template_batch, encoder_def, agent_class, agent_kwargs):
    """Create ensemble agents with vectorized creation."""
    logging.info(f"Creating {ensemble_size} ensemble agents...")
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


def main(_):
    # Single GPU setup
    devices = jax.local_devices()
    logging.info(f"Available devices: {devices}")

    ensemble_size = FLAGS.config.ensemble_size
    batch_size = FLAGS.config.batch_size
    batch_size_per_member = batch_size // ensemble_size
    num_devices = len(devices)

    assert ensemble_size == num_devices, \
        f"Ensemble size ({ensemble_size}) must equal num devices ({num_devices}) for PMAP"
    assert batch_size % ensemble_size == 0, \
        f"batch_size ({batch_size}) must be divisible by ensemble_size ({ensemble_size})"

    logging.info(f"Ablation training: {ensemble_size} ensemble members, batch size {batch_size}, {batch_size_per_member} per member")
    logging.info(f"Negative demos: {FLAGS.config.create_negative_demos}, ratio {FLAGS.config.negative_demo_ratio}")
    logging.info(f"Negative final reward: {FLAGS.config.negative_final_reward}")
    logging.info(f"Training samples: {FLAGS.config.num_train_samples}")

    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(FLAGS.config.seed)

    # Setup wandb
    tags = ["ablation", f"ensemble_{ensemble_size}"]
    if FLAGS.config.create_negative_demos:
        tags.extend(["neg_demos", f"final_reward_{FLAGS.config.negative_final_reward}"])

    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({
        "project": FLAGS.project,
        "exp_descriptor": FLAGS.name,
        "tags": tags,
        "group": "reward_ablation"
    })

    variant = FLAGS.config.to_dict()
    variant.update({
        "oxe_config": FLAGS.oxedata_config.to_dict(),
        "ensemble_size": ensemble_size,
    })

    wandb_logger = WandBLogger(wandb_config=wandb_config, variant=variant)

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # Text processor setup
    assert FLAGS.config.get("text_processor") is not None, "Must specify text_processor in config"
    text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)

    def process_text(batch, keep_language_str=False):
        decoded_strings = [s.decode("utf-8") for s in batch["goals"]["language"]]
        batch["goals"]["language"] = text_processor.encode(decoded_strings)
        if keep_language_str:
            batch["goals"]["language_str"] = decoded_strings
        # Encode original_language the same way (avoids passing raw bytes to pmap)
        if "original_language" in batch["goals"]:
            orig_decoded = [s.decode("utf-8") for s in batch["goals"]["original_language"]]
            batch["goals"]["original_language"] = text_processor.encode(orig_decoded)
            if keep_language_str:
                batch["goals"]["original_language_str"] = orig_decoded
        return batch

    def process_batch(batch, keep_language_str=False, training=False):
        """Process raw batch to training format."""
        
        def reshape_to_ensemble(x):
            return x.reshape(ensemble_size, batch_size_per_member, *x.shape[1:])

        pre_batch = {
            "actions": batch["action"].squeeze(),
            "next_actions": batch["next_action"].squeeze(),
            "goals": {
                "language": batch["task"]["language_instruction"],
                # Add original language if it exists (only for negative demo datasets)
                # Adding additional keys for goals is not an issue despite _include_goals_in_obs because LCEncodingWrapper only looks at the "language" key
                **({"original_language": batch["task"]["original_language_instruction"]} 
                if "original_language_instruction" in batch["task"] else {}) 
            },
            "mc_returns": batch["mc_return"],
            "observations": {"image": batch["observation"]["image_primary"].squeeze()},
            "next_observations": {"image": batch["next_observation"]["image_primary"].squeeze()},
            "rewards": batch["reward"],
            "masks": batch["td_mask"],
        }
        
        processed_batch = process_text(pre_batch, keep_language_str=keep_language_str)

        if training: # in training, we want to reshape (batch_size,) → (ensemble_size, batch_size // ensemble_size)
            ensemble_batch = jax.tree_map(reshape_to_ensemble, processed_batch)
        else: # in validation, we want to keep the batch size as is as we are broadcasting the batch
            ensemble_batch = processed_batch 

        return ensemble_batch

    # Dataset Creation - Bridge only via make_single_dataset
    logging.info("Creating Bridge-only dataset...")

    # Create dataset kwargs for Bridge
    bridge_kwargs = make_oxe_dataset_kwargs(
        name="bridge_dataset",
        data_dir=FLAGS.oxedata_config.oxe_kwargs.data_dir,
        load_camera_views=FLAGS.oxedata_config.oxe_kwargs.load_camera_views,
        load_depth=FLAGS.oxedata_config.oxe_kwargs.load_depth,
        load_proprio=FLAGS.oxedata_config.oxe_kwargs.load_proprio,
        load_language=FLAGS.oxedata_config.oxe_kwargs.load_language,
        force_recompute_dataset_statistics=FLAGS.oxedata_config.oxe_kwargs.force_recompute_dataset_statistics,
        action_proprio_normalization_type=FLAGS.oxedata_config.oxe_kwargs.action_proprio_normalization_type,
        discount=FLAGS.oxedata_config.oxe_kwargs.discount,
        num_final_repeat=FLAGS.oxedata_config.oxe_kwargs.num_final_repeat,
    )

    traj_transform_kwargs = dict(FLAGS.oxedata_config.traj_transform_kwargs)
    frame_transform_kwargs = dict(FLAGS.oxedata_config.frame_transform_kwargs)

    # Training dataset
    train_data = make_single_dataset(
        dataset_kwargs=bridge_kwargs,
        train=True,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        exclude_empty_lang_instr=FLAGS.config.exclude_empty_lang_instr,
    )

    # Apply negative demonstrations if enabled
    if FLAGS.config.create_negative_demos:
        logging.info(f"Creating negative demos with final_reward={FLAGS.config.negative_final_reward}")
        train_data, _ = create_negative_demonstrations(
            train_data,
            negative_ratio=FLAGS.config.negative_demo_ratio,
            seed=FLAGS.config.seed,
            negative_final_reward=FLAGS.config.negative_final_reward,
        )

    # Flatten, take subset, shuffle, batch
    train_data_flat = train_data.flatten()
    if FLAGS.config.num_train_samples > 0:
        logging.info(f"Taking subset of {FLAGS.config.num_train_samples} samples")
        train_data_flat = train_data_flat.take(FLAGS.config.num_train_samples)

    train_data_batched = (
        train_data_flat
        .shuffle(10000)
        .repeat()
        .batch(batch_size)
        .iterator(prefetch=0)
    )
    train_iterator = map(partial(process_batch, training=True), train_data_batched)

    # Validation Dataset - 100% neg-demo (same images, original + swapped lang)
    logging.info("Creating validation dataset (100% neg-demo)...")

    val_infeasible_base = make_single_dataset(
        dataset_kwargs=bridge_kwargs,
        train=False,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        exclude_empty_lang_instr=True,  # Must exclude empty for swapping
    )
    val_neg_demo_data, _ = create_negative_demonstrations(
        val_infeasible_base,
        negative_ratio=1.0,  # 100% swapped, stores original_language_instruction
        seed=FLAGS.config.seed + 1,
        negative_final_reward=FLAGS.config.negative_final_reward,
    )
    val_neg_demo_iter = (
        val_neg_demo_data.flatten()
        .shuffle(1000)
        .repeat()
        .batch(batch_size_per_member)
        .iterator(prefetch=0)
    )
    val_neg_demo_iter = map(process_batch, val_neg_demo_iter)

    # Trajectory-level iterator for plotting (neg-demo trajectories)
    val_traj_iter = map(
        partial(process_batch, keep_language_str=True),
        val_neg_demo_data.shuffle(1000).repeat().iterator()
    )

    # Agent Setup
    logging.info("Setting up agent...")

    # Get template batch for agent creation
    template_batch = next(train_iterator)

    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
    agent_class = agents[FLAGS.config.agent]

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, agent_rng = jax.random.split(rng)

    # Extract single member's batch for template (same as train_ensemble.py line 419)
    template_batch_per_member = jax.tree_map(lambda x: x[0], template_batch)
    ensemble_agents = create_ensemble_agents(
        ensemble_size=ensemble_size,
        rng=agent_rng,
        template_batch=template_batch_per_member,
        encoder_def=encoder_def,
        agent_class=agent_class,
        agent_kwargs=FLAGS.config.agent_kwargs,
    )

    logging.info(f"Created ensemble with shape: {jax.tree_map(lambda x: x.shape, ensemble_agents.state.params)}")

    # Training Functions (8 GPUs with pmap)

    # VMAP version (single GPU) - commented for PMAP testing
    # @jax.jit
    # def update_step(ensemble_agents, ensemble_batch):
    #     """Update ensemble members, each with different data slice."""
    #     # vmap update over ensemble
    #     new_agents, infos = jax.vmap(lambda agent, b: agent.update(b))(ensemble_agents, ensemble_batch)
    #     return new_agents, infos
    #
    # @jax.jit
    # def compute_debug_metrics(ensemble_agents, batch, rng_keys):
    #     """Compute debug metrics for all ensemble members."""
    #     metrics = jax.vmap(
    #         lambda agent, rng: agent.get_debug_metrics(batch, seed=rng)
    #     )(ensemble_agents, rng_keys)
    #     return metrics

    # PMAP version (8 GPUs) - for testing VMAP vs PMAP performance
    pmapped_update = jax.pmap(lambda agent, batch: agent.update(batch))
    pmapped_debug_metrics = jax.pmap(
        lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key),
        in_axes=(0, None, 0),  # agent per-device, batch broadcast, rng per-device
    )

    def compute_val_metrics(ensemble_agents, val_neg_demo_iter, rng, num_iters=6):
        """Compute val metrics: same images with original vs swapped language."""
        original_metrics_list, modified_metrics_list = [], []
        for _ in range(num_iters):
            batch = next(val_neg_demo_iter)
            rng, val_rng = jax.random.split(rng)
            val_rngs = jax.random.split(val_rng, ensemble_size)
            # Same images, original (correct) language
            original_batch = jax.tree_map(lambda x: x, batch)
            original_batch["goals"]["language"] = batch["goals"]["original_language"]
            original_metrics_list.append(pmapped_debug_metrics(ensemble_agents, original_batch, val_rngs))
            # Same images, swapped (wrong) language
            modified_metrics_list.append(pmapped_debug_metrics(ensemble_agents, batch, val_rngs))

        avg_orig = jax.device_get(jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *original_metrics_list))
        avg_mod  = jax.device_get(jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *modified_metrics_list))
        q_diff = float(jnp.mean(avg_orig["online_q"] - avg_mod["online_q"]))
        orig_disag = float(jnp.std(avg_orig["online_q"]))
        mod_disag  = float(jnp.std(avg_mod["online_q"]))
        return {
            "val/q_value_difference_lang":    q_diff,
            "val/mean_q_original_lang":       float(jnp.mean(avg_orig["online_q"])),
            "val/mean_q_swapped_lang":        float(jnp.mean(avg_mod["online_q"])),
            "val/disagreement_original_lang": orig_disag,
            "val/disagreement_swapped_lang":  mod_disag,
            "val/disagreement_ratio_lang":    mod_disag / (orig_disag + 1e-8),
        }, rng

    # Training Loop
    timer = Timer()
    total_steps = int(FLAGS.config.num_steps)
    logging.info(f"Training for {total_steps} steps")

    for step in tqdm.tqdm(range(total_steps)):
        timer.tick("total")

        # Get batch and update
        timer.tick("dataset")
        batch = next(train_iterator)
        timer.tock("dataset")

        timer.tick("train")
        # ensemble_agents, update_infos = update_step(ensemble_agents, batch)  # VMAP
        ensemble_agents, update_infos = pmapped_update(ensemble_agents, batch)  # PMAP
        timer.tock("train")

        # Validation
        if step % FLAGS.config.eval_interval == 0:
            timer.tick("val")

            val_metrics, rng = compute_val_metrics(ensemble_agents, val_neg_demo_iter, rng)

            timer.tock("val")

            wandb_logger.log(val_metrics, step=step)

            logging.info(
                f"Step {step}: Q_diff_lang={val_metrics['val/q_value_difference_lang']:.3f}, "
                f"Disag_ratio={val_metrics['val/disagreement_ratio_lang']:.3f}, "
                f"Q_orig={val_metrics['val/mean_q_original_lang']:.3f}, "
                f"Q_swap={val_metrics['val/mean_q_swapped_lang']:.3f}"
            )

            # Trajectory value plots: same images, swapped vs original language
            for num in range(2):
                traj = next(val_traj_iter)
                rng, val_rng = jax.random.split(rng)
                val_rngs = jax.random.split(val_rng, ensemble_size)

                # Plot with swapped (wrong) language — expected low Q-values
                plot = SARSAEnsembleAgent.plot_ensemble_trajectory_values(
                    ensemble_agents, ensemble_size, traj, val_rngs, plot_label="Swapped"
                )
                wandb_logger.log({f"ensemble_value_plots/traj_swapped_lang_{num}": wandb.Image(plot)}, step=step)

                # Plot with original (correct) language — expected higher Q-values
                original_goals = jax.tree_map(lambda x: x, traj["goals"])
                original_goals["language"] = traj["goals"]["original_language"]
                if "original_language_str" in traj["goals"]:
                    original_goals["language_str"] = traj["goals"]["original_language_str"]
                plot = SARSAEnsembleAgent.plot_ensemble_trajectory_values(
                    ensemble_agents, ensemble_size, traj, val_rngs, goals=original_goals,
                    plot_label="Original"
                )
                wandb_logger.log({f"ensemble_value_plots/traj_original_lang_{num}": wandb.Image(plot)}, step=step)

        # Training logging
        if step % FLAGS.config.log_interval == 0:
            critic_info = jax.device_get(update_infos)

            # Per-member values for std/disagreement
            member_losses = critic_info["critic_loss"]  # (ensemble_size,)
            member_q = critic_info["online_q"]  # (ensemble_size,)
            member_target_q = critic_info["target_q"]  # (ensemble_size,)

            train_log = {
                "train/critic_loss": float(jnp.mean(member_losses)),
                "train/critic_loss_std": float(jnp.std(member_losses)),
                "train/mean_q": float(jnp.mean(member_q)),
                "train/target_q": float(jnp.mean(member_target_q)),
                "train/td_error": float(jnp.mean(critic_info["td_err"])),
                "train/rewards": float(jnp.mean(critic_info["rewards"])),
                "train/ensemble_disagreement_q": float(jnp.std(member_q)),
            }
            wandb_logger.log(train_log, step=step)

            logging.info(
                f"Step {step}: loss={float(jnp.mean(member_losses)):.4f}, "
                f"Q={float(jnp.mean(member_q)):.3f}, "
                f"target_Q={float(jnp.mean(member_target_q)):.3f}, "
                f"td_err={float(jnp.mean(critic_info['td_err'])):.4f}"
            )

        timer.tock("total")

        # Checkpoint saving
        if (step + 1) % FLAGS.config.save_interval == 0:
            logging.info(f"Saving checkpoint at step {step + 1}")

            # VMAP checkpoint (single file) - commented for PMAP
            # checkpoint_dir = os.path.join(save_dir, "checkpoints")
            # tf.io.gfile.makedirs(checkpoint_dir)
            # checkpoints.save_checkpoint(
            #     checkpoint_dir,
            #     jax.device_get(ensemble_agents),
            #     step=step + 1,
            #     keep=2,
            # )

            # PMAP checkpoint (per-member files)
            for member_idx in range(ensemble_size):
                member_agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
                member_save_dir = os.path.join(save_dir, f"ensemble_member_{member_idx}")
                checkpoints.save_checkpoint(
                    member_save_dir,
                    member_agent,
                    step=step + 1,
                    keep=2,
                )

    logging.info("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
