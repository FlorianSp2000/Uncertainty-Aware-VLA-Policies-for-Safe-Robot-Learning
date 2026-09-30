# Training loop adapted from V-GPS (https://github.com/nakamotoo/V-GPS) experiments/train.py.
"""
Binary classifier training script.
Uses ResNet+MUSE architecture for feasibility classification.
Simplified from train_ensemble.py - single GPU, no pmap.
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

from octo.utils.train_callbacks import create_validation_dataset
from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_utils import filter_eval_datasets

from experiments.utils.finetuning import (
    create_negative_demonstrations_classifier,  # V1 for validation (deterministic)
    collect_prompt_pool,  # V2: collect prompts from any dataset (flattened or trajectory)
    swap_batch_prompts_tf,  # V2: per-batch random swapping (also creates task_feasible)
)

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "binary_classifier_baseline", "WandB project name.")
flags.DEFINE_integer(
    "train_take_n",
    -1,
    "Number of trajectories to take from training set; set <=0 to use full dataset.",
)

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("oxedata_config", None, "Data configuration.", lock_config=False)


def main(_):
    logging.info(f"Binary classifier training on single GPU")
    logging.info(f"Batch size: {FLAGS.config.batch_size}")
    logging.info(f"Negative demo ratio: {FLAGS.config.negative_demo_ratio}")

    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(FLAGS.config.seed)

    # Setup wandb
    take_tag = (
        f"take_{FLAGS.train_take_n}"
        if FLAGS.train_take_n > 0
        else "take_full"
    )
    tags = [
        "binary_classifier",
        f"negdemo_{FLAGS.config.negative_demo_ratio}",
        take_tag,
    ]
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update({
        "project": FLAGS.project,
        "exp_descriptor": f"{FLAGS.name}",
        "tags": tags,
    })

    variant = FLAGS.config.to_dict()
    variant.update({
        "oxe_config": FLAGS.oxedata_config.to_dict(),
        "train_take_n": FLAGS.train_take_n,
    })

    wandb_logger = WandBLogger(wandb_config=wandb_config, variant=variant)

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # Text processor setup
    text_processor = None
    if FLAGS.config.get("text_processor"):
        text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)

    def process_text(batch):
        """Process language instructions."""
        decoded_strings = [s.decode("utf-8") for s in batch["goals"]["language"]]
        if text_processor is not None:
            batch["goals"]["language"] = text_processor.encode(decoded_strings)
        return batch

    def process_batch(batch, keep_language_str=False):
        """Preprocess batch for binary classifier.

        Uses np.asarray() to handle both V2 training (numpy) and V1 validation (TF tensors).
        """
        lang = np.asarray(batch["task"]["language_instruction"])
        pre_batch = {
            "goals": {
                "language": lang,
            },
            "observations": {"image": np.asarray(batch["observation"]["image_primary"]).squeeze()},
            "task_feasible": np.asarray(batch["task_feasible"])[:, None],  # (B,) -> (B, 1)
        }

        processed_batch = process_text(pre_batch)

        if keep_language_str:
            decoded_strings = [s.decode("utf-8") for s in lang]
            processed_batch["goals"]["language_str"] = decoded_strings

        return processed_batch

    def process_trajectory(traj):
        """Preprocess single trajectory for per-timestep evaluation.

        Different from process_batch: uses ONE language prompt for entire trajectory,
        and properly shapes data for sequential evaluation (not batch).
        """
        # Get all language prompts from trajectory (should be same for all timesteps)
        lang_all = np.asarray(traj["task"]["language_instruction"])
        # Use first prompt (they should all be the same for a trajectory)
        lang_single = lang_all[0] if lang_all.ndim > 0 else lang_all
        if isinstance(lang_single, bytes):
            lang_str = lang_single.decode("utf-8")
        else:
            lang_str = str(lang_single)

        # Encode single prompt
        if text_processor is not None:
            lang_encoded = text_processor.encode([lang_str])  # (1, embed_dim)
        else:
            lang_encoded = np.array([[lang_str]])

        # Get images: (T, 1, H, W, C) → (T, H, W, C)
        images = np.asarray(traj["observation"]["image_primary"]).squeeze()
        traj_len = images.shape[0]

        # Get feasibility labels
        task_feasible = np.asarray(traj["task_feasible"])
        if task_feasible.ndim == 0:
            task_feasible = np.full((traj_len,), task_feasible)
        task_feasible = task_feasible[:, None]  # (T,) -> (T, 1)

        return {
            "observations": {"image": images},  # (T, H, W, C)
            "goals": {
                "language": lang_encoded,  # (1, embed_dim)
                "language_str": [lang_str],  # List with single string
            },
            "task_feasible": task_feasible,  # (T, 1)
        }

    # Dataset creation with OXE preprocessing
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(**FLAGS.oxedata_config["oxe_kwargs"])
        del FLAGS.oxedata_config["oxe_kwargs"]

    # Create training dataset WITHOUT negative demos (V2: apply at batch level)
    train_data = make_interleaved_dataset(
        **FLAGS.oxedata_config, train=True,
        exclude_empty_lang_instr=FLAGS.config.exclude_empty_lang_instr,
        seed=FLAGS.config.seed,
        create_negative_demos=False,  # V2: don't apply V1 negative demo creation
    )

    # Take subset if needed
    train_take_n = FLAGS.train_take_n
    if train_take_n > 0:
        logging.info("Using %d training trajectories", train_take_n)
        train_data = train_data.take(train_take_n)
    else:
        logging.info("Using full training dataset")

    # V2: Collect prompt pool from flattened samples (no traj_map needed)
    # swap_batch_prompts_tf will create task_feasible field per batch
    unique_prompts = collect_prompt_pool(train_data, prompt_pool_size=10_000)
    prompt_pool_tensor = tf.constant(unique_prompts)
    logging.info(f"V2: Collected {len(unique_prompts)} unique prompts for batch-level swapping")

    # Raw training iterator (swap applied per-batch before process_batch)
    train_iter_raw = (
        train_data
        .repeat()
        .unbatch()
        .shuffle(1000)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )

    def get_training_batch():
        """Get batch with V2 per-batch random swapping (before tokenization)."""
        batch = next(train_iter_raw)
        batch = swap_batch_prompts_tf(batch, prompt_pool_tensor, negative_ratio=FLAGS.config.negative_demo_ratio)
        return process_batch(batch)

    # Bridge validation dataset
    val_bridge_kwargs_list, _ = filter_eval_datasets(
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
        ["bridge_dataset"],
    )
    val_bridge_ds = create_validation_dataset(
        val_bridge_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False,
        exclude_empty_lang_instr=FLAGS.config.exclude_empty_lang_instr
    )

    # Validation with 50% negative demos for balanced evaluation
    val_bridge_with_neg, _ = create_negative_demonstrations_classifier(
        val_bridge_ds.filter(lambda traj: traj['task']['language_instruction'][0] != b''),
        negative_ratio=0.5,
        seed=FLAGS.config.seed + 10
    )
    # TODO: in binary octo we only eval bridge_traj not batchwise
    val_bridge_iterator = (
        val_bridge_with_neg
        .unbatch()
        .shuffle(1000)
        .repeat()
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    val_bridge_iterator = map(process_batch, val_bridge_iterator)

    # Fractal validation dataset
    val_fractal_kwargs_list, _ = filter_eval_datasets(
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
        ["fractal20220817_data"],
    )
    val_fractal_ds = create_validation_dataset(
        val_fractal_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False,
        exclude_empty_lang_instr=FLAGS.config.exclude_empty_lang_instr
    )

    # Validation with 50% negative demos for balanced evaluation
    val_fractal_with_neg, _ = create_negative_demonstrations_classifier(
        val_fractal_ds.filter(lambda traj: traj['task']['language_instruction'][0] != b''),
        negative_ratio=0.5,
        seed=FLAGS.config.seed + 20
    )
    val_fractal_iterator = (
        val_fractal_with_neg
        .unbatch()
        .shuffle(1000)
        .repeat()
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    val_fractal_iterator = map(process_batch, val_fractal_iterator)

    # Trajectory iterator for plotting (occasional)
    # Use process_trajectory instead of process_batch for proper per-timestep handling
    val_traj_iterator = map(
        process_trajectory,
        val_bridge_with_neg.shuffle(1000).repeat().iterator()
    )

    # Get example batch using V2 swapping
    example_batch = get_training_batch()
    logging.info(f"Example batch shape: {jax.tree_map(lambda x: x.shape, example_batch)}")

    # Encoder setup
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # Create agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    agent_class = agents[FLAGS.config.agent]

    agent = agent_class.create(
        rng=rng,
        observations=example_batch["observations"],
        encoder_def=encoder_def,
        goals=example_batch["goals"],
        **FLAGS.config.agent_kwargs
    )

    logging.info(f"Agent created: {agent_class.__name__}")

    # Training loop
    timer = Timer()
    total_steps = int(FLAGS.config.num_steps)

    for i in tqdm.tqdm(range(total_steps), total=total_steps, dynamic_ncols=True):
        # timer.tick("total")

        timer.tick("dataset")
        batch = get_training_batch()  # V2: per-batch random swapping
        timer.tock("dataset")

        timer.tick("train")
        agent, update_info = agent.update(batch)
        timer.tock("train")

        # Logging
        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log(
                {f"training/{k}": v for k, v in update_info.items()},
                step=i,
            )
            wandb_logger.log({"timer": timer.get_average_times()}, step=i)

        # Validation
        if i % FLAGS.config.eval_interval == 0:
            logging.info(f"Running validation at step {i}...")
            timer.tick("val")

            # Compute validation metrics over multiple batches
            val_metrics_list = []
            for _ in range(8):
                val_batch = next(val_bridge_iterator)
                val_metrics = agent.get_debug_metrics(val_batch)
                val_metrics_list.append(val_metrics)

            # Average metrics
            avg_val_metrics = jax.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs), axis=0),
                *val_metrics_list
            )
            avg_val_metrics = jax.device_get(avg_val_metrics)

            logging.info(f"Bridge Validation - Accuracy: {avg_val_metrics['accuracy']:.3f}, "
                        f"Mean prob: {avg_val_metrics['mean_prob']:.3f}")

            wandb_logger.log(
                {f"validation/bridge/{k}": v for k, v in avg_val_metrics.items()},
                step=i,
            )

            # Fractal validation
            val_fractal_metrics_list = []
            for _ in range(8):
                val_fractal_batch = next(val_fractal_iterator)
                val_fractal_metrics = agent.get_debug_metrics(val_fractal_batch)
                val_fractal_metrics_list.append(val_fractal_metrics)

            # Average fractal metrics
            avg_val_fractal_metrics = jax.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs), axis=0),
                *val_fractal_metrics_list
            )
            avg_val_fractal_metrics = jax.device_get(avg_val_fractal_metrics)

            logging.info(f"Fractal Validation - Accuracy: {avg_val_fractal_metrics['accuracy']:.3f}, "
                        f"Mean prob: {avg_val_fractal_metrics['mean_prob']:.3f}")

            # Log fractal validation metrics
            wandb_logger.log(
                {f"validation/fractal/{k}": v for k, v in avg_val_fractal_metrics.items()},
                step=i,
            )

            # Occasional trajectory plot (every 10k steps)
            if i % (FLAGS.config.eval_interval * 2) == 0 and i > 0:
                logging.info("Plotting trajectory probabilities...")
                traj = next(val_traj_iterator)
                rng, plot_rng = jax.random.split(rng)

                plot = agent.plot_values(traj, seed=plot_rng)
                plot = wandb.Image(plot)
                wandb_logger.log({"plots/trajectory": plot}, step=i)

            timer.tock("val")

        # Save checkpoint
        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info(f"Saving checkpoint at step {i+1}...")
            checkpoint_path = os.path.join(save_dir, "checkpoints")
            checkpoints.save_checkpoint(
                checkpoint_path,
                agent,
                step=i+1,
                keep=3,
            )
        
        # timer.tock("total")

    # Final save (only if not already saved in last iteration)
    if total_steps % FLAGS.config.save_interval != 0:
        logging.info("Training completed. Saving final checkpoint...")
        checkpoint_path = os.path.join(save_dir, "checkpoints")
        checkpoints.save_checkpoint(
            checkpoint_path,
            agent,
            step=total_steps,
            keep=3,
        )
    else:
        logging.info("Training completed. Final checkpoint already saved at last interval.")

    # Final validation
    logging.info("Running final validation...")

    # Bridge final validation
    val_metrics_list = []
    for _ in range(8):
        val_batch = next(val_bridge_iterator)
        val_metrics = agent.get_debug_metrics(val_batch)
        val_metrics_list.append(val_metrics)

    avg_val_metrics = jax.tree_map(
        lambda *xs: jnp.mean(jnp.stack(xs), axis=0),
        *val_metrics_list
    )
    avg_val_metrics = jax.device_get(avg_val_metrics)

    logging.info(f"Final Bridge validation - Accuracy: {avg_val_metrics['accuracy']:.3f}")
    wandb_logger.log(
        {f"validation/bridge/{k}": v for k, v in avg_val_metrics.items()},
        step=total_steps,
    )

    # Fractal final validation
    val_fractal_metrics_list = []
    for _ in range(8):
        val_fractal_batch = next(val_fractal_iterator)
        val_fractal_metrics = agent.get_debug_metrics(val_fractal_batch)
        val_fractal_metrics_list.append(val_fractal_metrics)

    avg_val_fractal_metrics = jax.tree_map(
        lambda *xs: jnp.mean(jnp.stack(xs), axis=0),
        *val_fractal_metrics_list
    )
    avg_val_fractal_metrics = jax.device_get(avg_val_fractal_metrics)

    logging.info(f"Final Fractal validation - Accuracy: {avg_val_fractal_metrics['accuracy']:.3f}")
    wandb_logger.log(
        {f"validation/fractal/{k}": v for k, v in avg_val_fractal_metrics.items()},
        step=total_steps,
    )


if __name__ == "__main__":
    app.run(main)
