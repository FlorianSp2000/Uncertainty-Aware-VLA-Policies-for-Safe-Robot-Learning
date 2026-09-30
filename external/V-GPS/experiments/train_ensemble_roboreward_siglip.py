# Training loop adapted from V-GPS (https://github.com/nakamotoo/V-GPS) experiments/train.py.
"""
SARSA ensemble training on RoboReward RLDS dataset with SigLIP So400m/14 backbone.

Cloned from train_ensemble_roboreward.py with:
  - Pretrained SigLIP weight injection after agent creation
  - Periodic backbone freeze verification logging
  - 224x224 image resize (set via oxedata_config override)
"""
import os
import pickle
from functools import partial
import matplotlib.pyplot as plt

import dlimp as dl
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax import traverse_util
from flax.training import checkpoints
from ml_collections import config_flags

from jaxrl_m.agents import agents
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.data.text_processing import text_processors
import wandb
from jax.experimental.compilation_cache import compilation_cache
from utils.evaluate_inpainting_uncertainty import (
    run_ensemble_inference_on_inpaintings,
    plot_inpainting_separation,
    plot_inpainting_outliers,
)

from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_validation_dataset

compilation_cache.initialize_cache("/tmp/jax_compilation_cache")

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("project", "jaxrl_m_siglip_ensemble", "WandB project name.")

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
config_flags.DEFINE_config_file("oxedata_config", None, "Data configuration.", lock_config=False)


def get_latest_checkpoint_step(ckpt_dir, prefix="checkpoint_"):
    """Extract step number from latest checkpoint folder name."""
    if not tf.io.gfile.exists(ckpt_dir):
        return None
    try:
        steps = []
        for fname in tf.io.gfile.listdir(ckpt_dir):
            if fname.startswith(prefix):
                try:
                    steps.append(int(fname[len(prefix):]))
                except ValueError:
                    continue
        return max(steps) if steps else None
    except Exception as e:
        logging.warning(f"Error reading checkpoint dir {ckpt_dir}: {e}")
        return None


def create_ensemble_agents(ensemble_size, rng, template_batch, encoder_def, agent_class, agent_kwargs):
    """Create ensemble using pmap init (1 member per GPU) to avoid OOM from vmapped SigLIP."""
    logging.info(f"Creating {ensemble_size} ensemble agents via pmap init...")

    from jaxrl_m.agents.continuous.sarsa_ensemble import (
        SARSAEnsembleAgent, MLP, Critic, ModuleDict, ensemblize, make_optimizer, JaxRLTrainState,
    )
    from functools import partial as ft_partial

    # --- replicate encoder/model setup from create_ensemble_vectorized ---
    goal_conditioned = agent_kwargs.get("goal_conditioned", False)
    language_conditioned = agent_kwargs.get("language_conditioned", False)
    if language_conditioned:
        goal_conditioned = True

    enc_def = agent_class._create_encoder_def(
        encoder_def, use_proprio=False, enable_stacking=False,
        goal_conditioned=goal_conditioned,
        early_goal_concat=agent_kwargs.get("early_goal_concat", False),
        shared_goal_encoder=agent_kwargs.get("shared_goal_encoder", True),
        language_conditioned=language_conditioned,
        use_siglip_wrapper=agent_kwargs.get("use_siglip_wrapper", False),
        freeze_siglip_backbone=agent_kwargs.get("freeze_siglip_backbone", True),
    )

    network_kwargs = agent_kwargs.get("network_kwargs", {"hidden_dims": [256, 256], "activate_final": True, "use_layer_norm": False})
    critic_ensemble_size = agent_kwargs.get("critic_ensemble_size", 2)
    contrastive_weight = agent_kwargs["contrastive_weight"]
    sow_embeddings = contrastive_weight > 0

    critic_backbone = ft_partial(MLP, **network_kwargs)
    critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(name="critic_ensemble")
    critic_def = ft_partial(Critic, encoder=enc_def, network=critic_backbone, sow_embeddings=sow_embeddings)(name="critic")
    model_def = ModuleDict({"critic": critic_def})

    txs = {"critic": make_optimizer(
        learning_rate=agent_kwargs.get("learning_rate", 3e-4),
        warmup_steps=agent_kwargs.get("warmup_steps", 1000),
    )}

    network_input = (template_batch["observations"], template_batch["goals"]) if goal_conditioned else template_batch["observations"]
    actions = template_batch["actions"]

    # --- pmap init: each GPU materializes 1 member ---
    agent_rngs = jax.random.split(rng, ensemble_size + 1)
    init_rngs = agent_rngs[:-1]
    state_rngs = jax.random.split(agent_rngs[-1], ensemble_size)

    def _init_params(rng_key):
        return model_def.init(rng_key, critic=[network_input, actions])["params"]

    def _init_state(rng_key, params):
        return JaxRLTrainState.create(
            apply_fn=model_def.apply, params=params, txs=txs,
            target_params=params, rng=rng_key,
        )

    logging.info("pmap param init across %d devices...", ensemble_size)
    pmapped_init_params = jax.pmap(_init_params)
    ensemble_params = pmapped_init_params(init_rngs)

    logging.info("pmap state init across %d devices...", ensemble_size)
    pmapped_init_state = jax.pmap(_init_state)
    ensemble_states = pmapped_init_state(state_rngs, ensemble_params)

    # --- wrap in agent objects ---
    config = {
        "ensemble_size": ensemble_size,
        "critic_ensemble_size": critic_ensemble_size,
        "critic_subsample_size": agent_kwargs.get("critic_subsample_size", None),
        "discount": agent_kwargs.get("discount", 0.98),
        "soft_target_update_rate": agent_kwargs.get("soft_target_update_rate", 5e-3),
        "goal_conditioned": goal_conditioned,
        "language_conditioned": language_conditioned,
        "use_min_q": agent_kwargs.get("use_min_q", True),
        "num_ood_actions": agent_kwargs.get("num_ood_actions", 10),
        "ood_action_sample_method": agent_kwargs.get("ood_action_sample_method", "uniform"),
        "contrastive_weight": contrastive_weight,
        "contrastive_margin": agent_kwargs["contrastive_margin"],
    }

    def create_single_agent(state):
        return agent_class(state=state, config=config)

    ensemble_agents = jax.vmap(create_single_agent)(ensemble_states)
    logging.info("Ensemble agents created successfully via pmap init")
    return ensemble_agents


def aggregate_ensemble_metrics_cpu(ensemble_metrics):
    def compute_stats(x):
        return {"mean": jnp.mean(x, axis=0), "std": jnp.std(x, axis=0)}

    aggregated = jax.tree_map(compute_stats, ensemble_metrics)
    flattened = {}
    for metric_name, stats in aggregated.items():
        for stat_name, values in stats.items():
            if stat_name == "std" and "ood_q" in metric_name:
                flattened["ensemble_disagreement_ood_q"] = values
            elif stat_name == "std" and "online_q" in metric_name:
                flattened["ensemble_disagreement_online_q"] = values
            elif stat_name == "std" and "target_q" in metric_name:
                flattened["ensemble_disagreement_target_q"] = values
            else:
                flattened[f"{stat_name}_{metric_name}"] = values
    return flattened


def batch_log_metrics(wandb_logger, aggregated_metrics, member_metrics_list, step, prefix=""):
    log_dict = {}
    for key, value in aggregated_metrics.items():
        log_dict[f"{prefix}{key}"] = value
    for member_idx, member_metrics in enumerate(member_metrics_list):
        if member_idx >= 2:
            break
        for key, value in member_metrics.items():
            log_dict[f"{prefix}member_{member_idx}_{key}"] = value
    wandb_logger.log(log_dict, step=step)


def log_param_component_norms(ensemble_agents, step, wandb_logger):
    """Log param norms split by backbone / projection / critic MLP."""
    member0_params = jax.tree_map(lambda x: x[0], ensemble_agents.state.params)
    flat_p = traverse_util.flatten_dict(member0_params, sep="/")

    backbone_norm = 0.0
    proj_norm = 0.0
    mlp_norm = 0.0
    for k, v in flat_p.items():
        norm_val = float(jnp.linalg.norm(v))
        if "siglip/" in k:
            backbone_norm += norm_val
        elif "vision_language_proj" in k:
            proj_norm += norm_val
        elif "critic_ensemble" in k:
            mlp_norm += norm_val

    wandb_logger.log({
        "debug/backbone_param_norm": backbone_norm,
        "debug/proj_param_norm": proj_norm,
        "debug/mlp_param_norm": mlp_norm,
    }, step=step)
    logging.info(
        f"[param norms] backbone={backbone_norm:.2f}  proj={proj_norm:.4f}  mlp={mlp_norm:.4f}"
    )


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    ensemble_size = FLAGS.config.ensemble_size or num_devices
    batch_size_per_member = FLAGS.config.batch_size // ensemble_size

    logging.info(
        f"SigLIP roboreward ensemble: {ensemble_size} models, "
        f"batch_size={FLAGS.config.batch_size}, per_member={batch_size_per_member}"
    )

    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(FLAGS.config.seed)

    # ── Resume ────────────────────────────────────────────────────────────
    resume_step = 0
    if FLAGS.config.resume_path:
        ckpt_path = os.path.join(FLAGS.config.resume_path, "ensemble_member_0")
        resume_step = get_latest_checkpoint_step(ckpt_path) or 0
        logging.info(f"Resuming from step {resume_step}")

    # ── WandB ─────────────────────────────────────────────────────────────
    if FLAGS.config.get("resume_wandb_id"):
        wandb_id = FLAGS.config.resume_wandb_id
        unique_id = (
            wandb_id.replace(f"{FLAGS.name}_", "", 1)
            if FLAGS.name in wandb_id
            else wandb_id.split("_")[-1]
        )
        wandb_config = WandBLogger.get_default_config()
        wandb_config.update({
            "project": FLAGS.project,
            "exp_descriptor": FLAGS.name,
            "unique_identifier": unique_id,
        })
    else:
        wandb_config = WandBLogger.get_default_config()
        wandb_config.update({
            "project": FLAGS.project,
            "exp_descriptor": FLAGS.name,
            "tags": ["ensemble", f"ensemble_size_{ensemble_size}", "roboreward", "siglip"],
            "group": FLAGS.name,
        })

    variant = FLAGS.config.to_dict()
    variant.update({
        "oxe_config": FLAGS.oxedata_config.to_dict(),
        "ensemble_size": ensemble_size,
        "batch_size_per_member": batch_size_per_member,
    })
    wandb_logger = WandBLogger(wandb_config=wandb_config, variant=variant)

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # ── Text processor ────────────────────────────────────────────────────
    text_processor = None
    if FLAGS.config.get("text_processor"):
        text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)

    def process_text(batch, keep_language_str=False):
        decoded = [s.decode("utf-8") for s in batch["goals"]["language"]]
        if text_processor is not None:
            batch["goals"]["language"] = text_processor.encode(decoded)
        if keep_language_str:
            batch["goals"]["language_str"] = decoded
        return batch

    def process_oxe_batch(batch, keep_language_str=False, training=False):
        def reshape_to_ensemble(x):
            return x.reshape(ensemble_size, batch_size_per_member, *x.shape[1:])

        pre_batch = {
            "actions": batch["action"].squeeze(),
            "next_actions": batch["next_action"].squeeze(),
            "goals": {"language": batch["task"]["language_instruction"]},
            "mc_returns": batch["mc_return"],
            "observations": {"image": batch["observation"]["image_primary"].squeeze()},
            "next_observations": {"image": batch["next_observation"]["image_primary"].squeeze()},
            "rewards": batch["reward"],
            "masks": batch["td_mask"],
        }
        if keep_language_str:
            pre_batch["dataset_name"] = batch["dataset_name"]
            if "roboreward_score" in batch:
                pre_batch["roboreward_score"] = batch["roboreward_score"]
        processed = process_text(pre_batch, keep_language_str=keep_language_str)
        if training:
            return jax.tree_map(reshape_to_ensemble, processed)
        return processed

    # ── OXE dataset kwargs ─────────────────────────────────────────────────
    if "oxe_kwargs" in FLAGS.oxedata_config:
        (
            FLAGS.oxedata_config["dataset_kwargs_list"],
            FLAGS.oxedata_config["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(**FLAGS.oxedata_config["oxe_kwargs"])
        del FLAGS.oxedata_config["oxe_kwargs"]

    dataset_kwargs_list = FLAGS.oxedata_config["dataset_kwargs_list"]
    sample_weights = FLAGS.oxedata_config["sample_weights"]

    if not FLAGS.config.get("reward_scheme"):
        raise ValueError("reward_scheme missing from config — use :ensemble_sarsa_roboreward_siglip variant")
    reward_scheme = FLAGS.config.reward_scheme.to_dict()
    for kw in dataset_kwargs_list:
        kw["reward_scheme"] = reward_scheme
    logging.info(f"Using reward_scheme: {reward_scheme}")
    traj_transform_kwargs = FLAGS.oxedata_config["traj_transform_kwargs"]
    frame_transform_kwargs = FLAGS.oxedata_config["frame_transform_kwargs"]

    logging.info(
        f"Loaded {len(dataset_kwargs_list)} roboreward datasets "
        f"(first 3: {[kw['name'] for kw in dataset_kwargs_list[:3]]} ...)"
    )
    logging.info(f"Image resize_size: {frame_transform_kwargs.get('resize_size', 'NOT SET')}")

    # ── Training data ──────────────────────────────────────────────────────
    train_data = make_interleaved_dataset(
        **FLAGS.oxedata_config,
        train=True,
        exclude_empty_lang_instr=FLAGS.config.exclude_empty_lang_instr,
        seed=FLAGS.config.seed,
    )
    train_iterator = map(
        partial(process_oxe_batch, training=True),
        train_data.iterator(prefetch=0),
    )

    # ── Per-dataset val datasets ──────────────────────────────────────────
    logging.info("Building per-dataset val trajectory datasets...")
    val_ds_per_dataset = [
        create_validation_dataset(
            kw, traj_transform_kwargs, frame_transform_kwargs, train=False,
        )
        for kw in dataset_kwargs_list
    ]
    uniform_weights = [1.0] * len(val_ds_per_dataset)

    val_frame_mixed = dl.DLataset.sample_from_datasets(val_ds_per_dataset, weights=uniform_weights)
    val_frame_iter = map(
        process_oxe_batch,
        val_frame_mixed.unbatch().shuffle(1000).repeat().batch(batch_size_per_member).iterator(prefetch=0),
    )

    val_traj_mixed = dl.DLataset.sample_from_datasets(val_ds_per_dataset, weights=uniform_weights)
    val_traj_iter = map(
        partial(process_oxe_batch, keep_language_str=True),
        val_traj_mixed.shuffle(500).repeat().iterator(),
    )
    prev_val_traj = next(val_traj_iter)
    logging.info(f"Val traj keys: {list(prev_val_traj.keys())}")
    logging.info(f"Val traj actions shape: {prev_val_traj['actions'].shape}")

    # ── Agent init ─────────────────────────────────────────────────────────
    example_batch = next(train_iterator)
    template_batch = jax.tree_map(lambda x: x[0], example_batch)
    logging.info(f"Ensemble batch shapes: {jax.tree_map(lambda x: x.shape, example_batch)}")
    _ex_action = example_batch["actions"][0, 0]
    logging.info(f"[example action] shape={_ex_action.shape}  values={_ex_action}")
    _r = prev_val_traj["rewards"]
    _m = prev_val_traj["masks"]
    _mc = prev_val_traj["mc_returns"]
    logging.info(f"[reward check] val traj len={len(_r)}, dataset={prev_val_traj.get('dataset_name', ['?'])[0]}")
    logging.info(f"[reward check] rewards  first4={_r[:4]}  last4={_r[-4:]}")
    logging.info(f"[reward check] td_mask  first4={_m[:4]}  last4={_m[-4:]}")
    logging.info(f"[reward check] mc_ret   first4={_mc[:4]}  last4={_mc[-4:]}")

    # Log image shape to verify 224x224 resize
    _img_shape = example_batch["observations"]["image"].shape
    logging.info(f"[SigLIP check] image batch shape: {_img_shape}")
    assert _img_shape[-2] == _img_shape[-3] == 224, (
        f"Expected 224x224 images for SigLIP, got {_img_shape[-3]}x{_img_shape[-2]}. "
        f"Set --oxedata_config.frame_transform_kwargs.resize_size '(224,224)'"
    )

    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    agent_class = agents[FLAGS.config.agent]
    ensemble_agents = create_ensemble_agents(
        ensemble_size, rng, template_batch, encoder_def, agent_class, FLAGS.config.agent_kwargs
    )

    # ── Inject pretrained SigLIP weights ──────────────────────────────────
    siglip_ckpt = FLAGS.config.get("siglip_checkpoint_path")
    assert siglip_ckpt, "siglip_checkpoint_path must be set in config"
    logging.info(f"Loading pretrained SigLIP weights from: {siglip_ckpt}")

    # Log param tree structure before injection (for debugging key paths)
    member0_params = jax.tree_map(lambda x: x[0], ensemble_agents.state.params)
    flat_keys = list(traverse_util.flatten_dict(member0_params, sep="/").keys())
    siglip_keys = [k for k in flat_keys if "siglip" in k]
    logging.info(f"Agent param tree has {len(flat_keys)} leaves, {len(siglip_keys)} with 'siglip'")
    if siglip_keys:
        logging.info(f"SigLIP key examples: {siglip_keys[:5]}")
    else:
        logging.warning("No 'siglip' keys found in param tree! Check encoder naming.")
        logging.info(f"All param key prefixes: {sorted(set(k.split('/')[0] for k in flat_keys))}")

    from jaxrl_m.utils.siglip_weight_utils import load_and_inject_siglip_weights
    ensemble_agents = load_and_inject_siglip_weights(
        ensemble_agents,
        pi0_checkpoint_path=siglip_ckpt,
        ensemble_size=ensemble_size,
    )
    logging.info("Pretrained SigLIP weights injected successfully")

    # ── Checkpoint restore (after weight injection) ────────────────────────
    if FLAGS.config.resume_path:
        logging.info("Restoring ensemble agents from checkpoint...")
        restored = []
        for member_idx in range(ensemble_size):
            member_ckpt = os.path.join(FLAGS.config.resume_path, f"ensemble_member_{member_idx}")
            agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
            if tf.io.gfile.exists(member_ckpt):
                agent = checkpoints.restore_checkpoint(member_ckpt, target=agent)
                logging.info("Restored member %d from %s", member_idx, member_ckpt)
            restored.append(agent)
        ensemble_agents = jax.tree_map(lambda *args: jnp.stack(args, axis=0), *restored)
        logging.info(f"Restored ensemble shape: {jax.tree_map(lambda x: x.shape, ensemble_agents)}")

    # ── Pmapped functions ──────────────────────────────────────────────────
    pmapped_update = jax.pmap(lambda agent, batch: agent.update(batch))
    pmapped_debug_metrics = jax.pmap(
        lambda agent, batch, rng_key: agent.get_debug_metrics(batch, seed=rng_key),
        in_axes=(0, None, 0),
    )

    # ── Inpainting eval datasets ───────────────────────────────────────────
    inpainting_datasets = {}
    if FLAGS.config.eval_inpainting:
        pkl_paths = [p for p in FLAGS.config.eval_inpainting_data_paths.split(":") if p]
        assert pkl_paths, "eval_inpainting=True but eval_inpainting_data_paths is empty"
        for pkl_path in pkl_paths:
            ds_name = os.path.splitext(os.path.basename(pkl_path))[0]
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            failed = {(d["dataset"], d["trajectory_id"]) for d in data.get("failed_ids", [])}
            trajs = [
                t for t in data["trajectories"]
                if t.get("first_image_inpainted") is not None
                and t.get("last_image_inpainted") is not None
                and (t.get("dataset"), t.get("trajectory_id")) not in failed
            ]
            inpainting_datasets[ds_name] = trajs
            logging.info(f"[inpainting eval] loaded {len(trajs)} trajs from {ds_name}")

    # ── Training loop ──────────────────────────────────────────────────────
    timer = Timer()
    total_steps = int(FLAGS.config.num_steps)
    logging.info(f"Training from step {resume_step} to {total_steps}")

    # Log initial param norms (step 0)
    log_param_component_norms(ensemble_agents, 0, wandb_logger)

    for i in tqdm.tqdm(range(resume_step, total_steps)):
        timer.tick("total")

        timer.tick("dataset")
        ensemble_batch = next(train_iterator)
        timer.tock("dataset")

        timer.tick("train")
        ensemble_agents, ensemble_update_infos = pmapped_update(ensemble_agents, ensemble_batch)
        timer.tock("train")

        if i % FLAGS.config.eval_interval == 0:
            logging.info(f"Evaluating at step {i}...")
            timer.tick("val")

            # Backbone freeze verification
            log_param_component_norms(ensemble_agents, i, wandb_logger)

            # Frame-level val (6 batches averaged)
            val_metrics_list = []
            for _ in range(6):
                single_val_batch = next(val_frame_iter)
                rng, val_rng = jax.random.split(rng)
                val_rngs = jax.random.split(val_rng, ensemble_size)
                val_metrics = pmapped_debug_metrics(ensemble_agents, single_val_batch, val_rngs)
                val_metrics_list.append(val_metrics)

            avg_val_metrics = jax.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *val_metrics_list
            )
            avg_val_metrics_cpu = jax.device_get(avg_val_metrics)
            real_disagreement = jnp.mean(jnp.std(avg_val_metrics_cpu["online_q"], axis=0))
            logging.info(f"Frame-level disagreement (6 iters): {real_disagreement:.4f}")

            aggregated_val = aggregate_ensemble_metrics_cpu(avg_val_metrics_cpu)
            batch_log_metrics(wandb_logger, aggregated_val, [], i, "validation/")

            # Traj-level plots
            if "sarsa" in FLAGS.config.agent:
                logging.info("Plotting ensemble trajectory value functions...")
                for num in range(2):
                    traj = next(val_traj_iter)
                    rng, val_rng = jax.random.split(rng)
                    val_rngs = jax.random.split(val_rng, ensemble_size)

                    plot = agent_class.plot_ensemble_trajectory_values(
                        ensemble_agents, ensemble_size, traj, val_rngs
                    )
                    wandb_logger.log(
                        {f"ensemble_value_plots/traj_{num}": wandb.Image(plot)}, step=i
                    )

            # Inpainting OOD eval
            if FLAGS.config.eval_inpainting and inpainting_datasets:
                logging.info("Running inpainting OOD eval...")
                for ds_name, trajs in inpainting_datasets.items():
                    inp_results = run_ensemble_inference_on_inpaintings(
                        trajs, ensemble_agents, text_processor,
                        model_path=FLAGS.config.save_dir,
                        dataset_name=ds_name,
                        ensemble_size=ensemble_size,
                        model_type_tag="siglip_ensemble",
                        save_results=False,
                    )
                    fig, metrics = plot_inpainting_separation(inp_results, ds_name)
                    outlier_fig = plot_inpainting_outliers(trajs, inp_results, ds_name)
                    wandb_logger.log({
                        f"inpainting_eval/{ds_name}_plot": wandb.Image(fig),
                        f"inpainting_eval/{ds_name}_outliers": wandb.Image(outlier_fig),
                        f"inpainting_eval/{ds_name}_mean_delta_unc": metrics["mean_delta_unc"],
                        f"inpainting_eval/{ds_name}_pct_paired_increase": metrics["pct_paired_increase"],
                        f"inpainting_eval/{ds_name}_auc_unc": metrics["auc_unc"],
                    }, step=i)
                    plt.close(fig)
                    plt.close(outlier_fig)
                    logging.info(
                        f"[inpainting eval] {ds_name}:"
                        f" delta_unc={metrics['mean_delta_unc']:.4f}"
                        f" pct_paired↑={metrics['pct_paired_increase']:.1f}%"
                        f" auc={metrics['auc_unc']:.3f}"
                    )

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
            aggregated_train = aggregate_ensemble_metrics_cpu(ensemble_update_infos_cpu)
            batch_log_metrics(wandb_logger, aggregated_train, [], i, "training/")
            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


if __name__ == "__main__":
    app.run(main)
