# Adapted from Octo (https://github.com/octo-models/octo, MIT) examples/02_finetune_new_observation_action.py and scripts/finetune.py.
"""
This script is inspired from 02_finetune_new_observation_action.py and original finetune.py in Octo, finetuning octo to a new action space 
(7D end-effector + 1D binary classification) where the new additional action dimension is intended to indicate whether
the task is ill-posed or not. Ill-posed tasks are those where the image observations are not matching the language instruction,
for example when target object is not in visual observation present.

The main todo of this script is to adapt the dataset loading part to be compatible with the bridge_fractal dataset with negative examples that
we built in V-GPS/experiments/train_ensemble.py for example.
"""
from absl import app, flags, logging
from ml_collections import config_flags
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from functools import partial
import optax

from octo.data.dataset import make_interleaved_dataset, make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.utils.train_callbacks import create_deterministic_validation_dataset, create_validation_dataset
from octo.utils.train_utils import filter_eval_datasets
from octo.model.components.action_heads import BinaryClassificationHead
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    create_optimizer,
    merge_params,
    process_text,
    TrainState,
)

# V-GPS imports for validation and logging
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.utils.timer_utils import Timer

# Import negative demo creation functions
# V2 (no memorization): prepare_prompt_pool_and_dataset + swap_batch_prompts_tf
# V1 (legacy): create_negative_demonstrations_classifier (used for deterministic validation)
from experiments.utils.finetuning import (
    prepare_prompt_pool_and_dataset,
    swap_batch_prompts_tf,
    create_negative_demonstrations_classifier,
    create_negative_demonstrations_octo_debug,
)

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import wandb as wandb_lib

FLAGS = flags.FLAGS

flags.DEFINE_float("action_loss_weight", 1.0, "Weight for action loss.")
flags.DEFINE_float("classification_loss_weight", 1.0, "Weight for classification loss.")

# V-GPS dataset integration flags
flags.DEFINE_float("negative_demo_ratio", 0.50, "Ratio of negative demonstrations")
flags.DEFINE_bool("exclude_empty_lang_instr", True, "Exclude empty language instructions")
flags.DEFINE_string("experiment_name", None, "Experiment name for WandB logging")
flags.DEFINE_bool("use_debug_labels", False, "Use debug mode: replace language with 'feasible'/'infeasible' strings")
flags.DEFINE_integer(
    "train_take_n",
    5000,
    "Limit number of training trajectories; set <=0 to use full dataset.",
)

# Config files (matching train_ensemble.py pattern)
# TODO: Atm using data_config.py from V-GPS. This might actually not be ideal as that config contains unrelated RL parameters (num_steps etc.). We could consider 
# making the data loading more smooth by removing the RL-specific parts not only from config also from data loading functions.

config_flags.DEFINE_config_file("oxedata_config", None, "Data configuration.", lock_config=False)

# Octo finetune configuration for core hyperparameters 
config_flags.DEFINE_config_file("config", None, "Octo finetuning configuration.", lock_config=False)

def main(_):
    assert (
        FLAGS.config.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # Setup WandB using V-GPS pattern
    wandb_config = WandBLogger.get_default_config()
    exp_name = FLAGS.experiment_name
    tags = ["octo", "finetuning", "dual_head", f"negdemo_{FLAGS.negative_demo_ratio}"]
    if FLAGS.use_debug_labels:
        tags.append("debug_labels")
    wandb_config.update({
        "project": FLAGS.config.wandb.project,
        "exp_descriptor": exp_name,
        "tags": tags
    })
    
    variant = {
        "train_steps": FLAGS.config.num_steps,
        "batch_size": FLAGS.config.batch_size,
        "negative_demo_ratio": FLAGS.negative_demo_ratio,
        "action_loss_weight": FLAGS.action_loss_weight,
        "classification_loss_weight": FLAGS.classification_loss_weight,
        "finetuning_mode": FLAGS.config.finetuning_mode,
        "pretrained_path": FLAGS.config.pretrained_path,
        "seed": FLAGS.config.seed,
        "use_debug_labels": FLAGS.use_debug_labels,
    }
    
    wandb_logger = WandBLogger(wandb_config=wandb_config, variant=variant)

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.config.pretrained_path, step=FLAGS.config.pretrained_step)

    # Process OXE dataset kwargs
    (
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
    ) = make_oxe_dataset_kwargs_and_weights(**FLAGS.oxedata_config["oxe_kwargs"])
    oxe_kwargs = FLAGS.oxedata_config["oxe_kwargs"]
    del FLAGS.oxedata_config["oxe_kwargs"]

    logging.info(f"FLAGS.oxedata_config: {FLAGS.oxedata_config}")

    # Get bridge dataset kwargs
    bridge_kwargs_list, _ = filter_eval_datasets(
        FLAGS.oxedata_config["dataset_kwargs_list"],
        FLAGS.oxedata_config["sample_weights"],
        ["bridge_dataset"],
    )
    # Create dataset with negative demonstrations
    # dataset = make_interleaved_dataset(
    #     **FLAGS.oxedata_config, 
    #     train=True, 
    #     exclude_empty_lang_instr=FLAGS.exclude_empty_lang_instr, 
    #     seed=FLAGS.config.seed,
    #     create_negative_demos=True,  # Always enable for training
    #     negative_demo_ratio=FLAGS.negative_demo_ratio,
    #     octo_dataset=True  # Use Octo-specific negative demo creation
    # )

    # Create training dataset (bridge only, smaller subset)
    # bridge_train_ds = create_validation_dataset(
    #     bridge_kwargs_list[0],
    #     FLAGS.oxedata_config["traj_transform_kwargs"],
    #     FLAGS.oxedata_config["frame_transform_kwargs"],
    #     train=True, # ! IMPORTANT for training set
    #     exclude_empty_lang_instr=FLAGS.exclude_empty_lang_instr
    # )
    # Use data_config transforms (not finetune_config) for consistency with the SayNo ensemble
    bridge_train_ds = make_single_dataset(
        bridge_kwargs_list[0],
        traj_transform_kwargs=FLAGS.oxedata_config["traj_transform_kwargs"],
        frame_transform_kwargs=FLAGS.oxedata_config["frame_transform_kwargs"],
        train=True,
        exclude_empty_lang_instr=FLAGS.exclude_empty_lang_instr
    )

    dataset_stats = bridge_train_ds.dataset_statistics
    # Take subset and add negative demos
    if FLAGS.train_take_n and FLAGS.train_take_n > 0:
        logging.info("Limiting training data to first %d trajectories", FLAGS.train_take_n)
        bridge_train_ds = bridge_train_ds.take(FLAGS.train_take_n)
    else:
        logging.info("Using full bridge training dataset (no take limit)")

    # Select negative demo creation function based on debug flag
    if FLAGS.use_debug_labels:
        # Debug mode: use V1 with literal "feasible"/"infeasible" strings
        logging.info("Using DEBUG mode: replacing language with 'feasible'/'infeasible' label strings")
        bridge_train_with_neg, _ = create_negative_demonstrations_octo_debug(
            bridge_train_ds.filter(lambda traj: traj['task']['language_instruction'][0] != b''),
            negative_ratio=FLAGS.negative_demo_ratio,
            seed=FLAGS.config.seed
        )
        prompt_pool_tensor = None  # Not used in debug mode
    else:
        # V2: Collect prompt pool, swapping happens per-batch (prevents memorization)
        logging.info("Using V2 mode: per-batch random swapping (prevents memorization)")
        bridge_train_prepared, prompt_pool = prepare_prompt_pool_and_dataset(
            bridge_train_ds.filter(lambda traj: traj['task']['language_instruction'][0] != b''),
            prompt_pool_size=10_000
        )
        bridge_train_with_neg = bridge_train_prepared
        prompt_pool_tensor = tf.constant(prompt_pool)
        logging.info(f"Prompt pool size: {len(prompt_pool)} unique prompts")

    # Create raw iterator (no process_batch yet - swap happens before tokenization)
    train_data_iter_raw = (
        bridge_train_with_neg
        .repeat()
        .unbatch()
        .shuffle(1000)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        """Process batch for dual-head architecture: actions + feasibility labels.

        Uses np.asarray() to handle both V2 training (returns numpy) and V1 validation (returns TF tensors).
        """
        batch["task"]["language_instruction"] = np.asarray(batch["task"]["language_instruction"])
        batch["task_feasible"] = np.asarray(batch["task_feasible"])[:, None]  # (B,) -> (B, 1)
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    def get_training_batch():
        """Get batch with V2 per-batch random swapping (before tokenization)."""
        batch = next(train_data_iter_raw)
        # V2: Apply per-batch random swapping BEFORE tokenization (prevents memorization)
        if prompt_pool_tensor is not None:
            batch = swap_batch_prompts_tf(batch, prompt_pool_tensor, negative_ratio=FLAGS.negative_demo_ratio)
        return process_batch(batch)

    example_batch = get_training_batch()
    
    logging.info(f"example_batch underlying shapes {jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, example_batch)}")
    labels = example_batch["task_feasible"]
    print(f"Batch label distribution: {jnp.sum(labels)} feasible out of {len(labels)}")

    # Create bridge validation dataset
    val_bridge_traj_data = create_validation_dataset(
        bridge_kwargs_list[0],
        FLAGS.oxedata_config["traj_transform_kwargs"],
        FLAGS.oxedata_config["frame_transform_kwargs"],
        train=False,
        exclude_empty_lang_instr=FLAGS.exclude_empty_lang_instr
    )
    # Create version with 50% negative demos for classification evaluation
    if FLAGS.use_debug_labels:
        val_bridge_with_neg, _ = create_negative_demonstrations_octo_debug( # TODO:
            val_bridge_traj_data.filter(lambda traj: traj['task']['language_instruction'][0] != b''),
            negative_ratio=0.5,  # 50% for balanced evaluation
            seed=FLAGS.config.seed + 10
        )
    else:
        val_bridge_with_neg, _ = create_negative_demonstrations_classifier(
            val_bridge_traj_data.filter(lambda traj: traj['task']['language_instruction'][0] != b''),
            negative_ratio=0.5,  # 50% for balanced evaluation
            seed=FLAGS.config.seed + 10
        )
    val_bridge_traj_iter = map(
        partial(process_batch),
        val_bridge_with_neg.shuffle(1000).repeat().iterator()
    )
    # # Fractal validation dataset
    # val_fractal_kwargs_list, _ = filter_eval_datasets(
    #     FLAGS.oxedata_config["dataset_kwargs_list"],
    #     FLAGS.oxedata_config["sample_weights"],
    #     ["fractal20220817_data"],
    # )
    # # create_deterministic_validation_dataset
    # val_fractal_traj_data = create_validation_dataset(
    #     val_fractal_kwargs_list[0],
    #     FLAGS.oxedata_config["traj_transform_kwargs"], 
    #     FLAGS.oxedata_config["frame_transform_kwargs"],
    #     train=False,
    #     exclude_empty_lang_instr=FLAGS.exclude_empty_lang_instr
    # )
    # # Create version with 50% negative demos for classification evaluation  
    # val_fractal_with_neg, _ = create_negative_demonstrations_classifier(
    #     val_fractal_traj_data.filter(lambda traj: traj['task']['language_instruction'][0] != b''),
    #     negative_ratio=0.5,  # 50% for balanced evaluation
    #     seed=FLAGS.config.seed + 20
    # )
    # val_fractal_traj_iter = map(
    #     partial(process_batch),
    #     val_fractal_with_neg.shuffle(1000).repeat().iterator()
    # )

    config = pretrained_model.config
    del config["model"]["observation_tokenizers"]["wrist"]
    logging.info(f"pretrained model config before modification: {config}")

    # Note: With octo_finetune config_string, data_config.py now provides action_horizon=4
    # which preserves Octo's pretrained action chunking weights

    # Add binary classification head
    if FLAGS.config.finetuning_mode == "feasibility_head_only":
        # Add separate readout token for feasibility (trainable from scratch)
        config["model"]["readouts"]["feasibility"] = 1
        config["model"]["heads"]["feasibility"] = ModuleSpec.create(
            BinaryClassificationHead,
            readout_key="readout_feasibility",
            use_map=False,
        )
        logging.info("Using separate readout_feasibility token (trainable)")
    else:
        # Share readout_action token (original behavior)
        config["model"]["heads"]["feasibility"] = ModuleSpec.create(
            BinaryClassificationHead,
            readout_key="readout_action",
            use_map=False,
        )
        logging.info("Using shared readout_action token")

    # Initialize model and merge pretrained weights
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset_stats,
    )
    # copy pre-trained params into target_params for every param that has corresponding key + shape.
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    model = model.replace(params=merged_params)
    del pretrained_model

    # Create optimizer using config-driven approach
    optimizer_kwargs = FLAGS.config.optimizer.to_dict()
    
    # The config file already sets frozen_keys based on the mode ("full", "head_only")
    # No need to check FLAGS.freeze_transformer
    logging.info(f"Using finetuning mode: {FLAGS.config.finetuning_mode}")
    logging.info(f"Frozen keys: {optimizer_kwargs.get('frozen_keys')}")
    
    tx, lr_callable, param_norm_callable = create_optimizer(
        model.params,
        **optimizer_kwargs,
    )
    
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(FLAGS.config.seed),  # Use config seed
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        """Combined loss for action prediction + feasibility classification.
        
        :param params: Model parameters
        :param batch: Training batch with:
                     - batch["action"]: (batch, window, horizon, 7) for EE pose
                     - batch["task_feasible"]: (batch, window) or (batch, window, 1) for labels
        :param rng: Random key
        :param train: Training mode flag
        :returns: Total loss and metrics dict
        """

        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )

        # if FLAGS.action_loss_weight > 0:
        #     # Action loss (7D end-effector pose)
        #     # Mask out action loss for infeasible samples (task_feasible=0)
            
        #     # task_feasible shape: (batch,) -> expand to (batch, window, horizon, action_dim)
        #     feasibility_mask = batch["task_feasible"].astype(jnp.float32)  # (batch,)
            
        #     # Expand to match action shape: (batch,) -> (batch, window, horizon, action_dim)
        #     # batch["action"].shape is (256, 1, 4, 7) = (batch, window, horizon, action_dim)
        #     # data loader returns (batch_size,) of 0/1 labels
        #     feasibility_action_mask = feasibility_mask[:, None, None, None]  # (batch, 1, 1, 1)
        #     feasibility_action_mask = jnp.broadcast_to(
        #         feasibility_action_mask, 
        #         batch["action"].shape  # (batch, window, horizon, action_dim)
        #     )
            
        #     # Combine original action_pad_mask with feasibility mask
        #     # Only compute loss when both action is valid AND task is feasible
        #     # Convert float32 feasibility mask to boolean for logical AND operation
        #     feasibility_action_mask = feasibility_action_mask.astype(jnp.bool_)
        #     combined_action_mask = batch["action_pad_mask"] & feasibility_action_mask
            
        #     action_loss, action_metrics = bound_module.heads["action"].loss(
        #         transformer_embeddings,
        #         batch["action"],  # Shape: (batch, window, horizon, 7)
        #         batch["observation"]["timestep_pad_mask"],
        #         combined_action_mask,  # Now includes feasibility masking
        #         train=train,
        #     )
        # else: 
        action_loss = 0.0
        action_metrics = {"loss": 0.0, "mse": 0.0}
        
        # Classification loss (binary feasibility)
        # Pass labels directly, not embedded in action tensor
        classification_loss, classification_metrics = bound_module.heads["feasibility"].loss(
            transformer_embeddings,
            batch["task_feasible"],  # Shape: (batch, window) or (batch, window, 1)
            batch["observation"]["timestep_pad_mask"],
            action_pad_mask=None,  # Not used for classification but needed for signature
            train=train,
        )
        
        total_loss = (
            FLAGS.action_loss_weight * action_loss +
            FLAGS.classification_loss_weight * classification_loss
        )
        
        metrics = {
            "action": action_metrics,
            "feasibility": classification_metrics,
            "total_loss": total_loss,
        }

        return total_loss, metrics
    
    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        # Add elaborate metrics like finetune.py
        grad_norm = optax.global_norm(grads)
        updates, _ = tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info
    
    def evaluate_on_dataset(state, data_iter, dataset_name, num_batches=50):
        """Evaluate model on trajectory-level validation dataset."""
        action_mse_errors = []  # DiffusionActionHead returns "mse"
        classification_accuracies = []  # BinaryClassificationHead returns "accuracy"
        mean_probs = []
        std_probs = []
        probs_feasible = []
        probs_infeasible = []
        total_losses = []

        for i in range(num_batches):
            try:
                batch = next(data_iter)
                # Evaluate without dropout (train=False)
                loss, metrics = loss_fn(state.model.params, batch, state.rng, train=False)

                # Extract definitive action metrics from DiffusionActionHead/ContinuousActionHead
                action_metrics = metrics["action"]
                action_mse_errors.append(float(action_metrics["mse"]))  # Always present

                # Extract definitive classification metrics from BinaryClassificationHead
                classification_metrics = metrics["feasibility"]
                classification_accuracies.append(float(classification_metrics["accuracy"]))  # Always present

                # Extract probability metrics - fail fast if keys missing
                mean_probs.append(float(classification_metrics["mean_prob"]))
                std_probs.append(float(classification_metrics["std_prob"]))
                probs_feasible.append(float(classification_metrics["prob_given_feasible"]))
                probs_infeasible.append(float(classification_metrics["prob_given_infeasible"]))

                total_losses.append(float(loss))

            except (StopIteration, tf.errors.OutOfRangeError):
                logging.warning(f"Dataset {dataset_name} exhausted at batch {i}")
                break

        # Compute averages
        eval_results = {
            f"{dataset_name}/total_loss": float(np.mean(total_losses)),
            f"{dataset_name}/action_mse": float(np.mean(action_mse_errors)),
            f"{dataset_name}/classification_accuracy": float(np.mean(classification_accuracies)),
            f"{dataset_name}/mean_prob": float(np.mean(mean_probs)),
            f"{dataset_name}/std_prob": float(np.mean(std_probs)),
            f"{dataset_name}/prob_given_feasible": float(np.mean(probs_feasible)),
            f"{dataset_name}/prob_given_infeasible": float(np.mean(probs_infeasible)),
            f"{dataset_name}/num_batches_evaluated": len(total_losses),
        }

        logging.info(f"✅ {dataset_name} evaluation: "
                    f"Acc={eval_results[f'{dataset_name}/classification_accuracy']:.3f}, "
                    f"P(pred|Y=1)={eval_results[f'{dataset_name}/prob_given_feasible']:.3f}, "
                    f"P(pred|Y=0)={eval_results[f'{dataset_name}/prob_given_infeasible']:.3f}")

        return eval_results

    def process_trajectory_for_plot(traj):
        """Process single trajectory for per-timestep feasibility plotting.

        Treats each timestep as a separate batch element for Octo model.
        """
        # Get images: (T, 1, H, W, C) → (T, H, W, C) → (T, 1, H, W, C) for Octo
        images = np.asarray(traj["observation"]["image_primary"])  # (T, 1, H, W, C)
        traj_len = images.shape[0]

        # Get single language prompt (all timesteps have same prompt in trajectory)
        lang_all = np.asarray(traj["task"]["language_instruction"])
        lang_single = lang_all[0] if lang_all.ndim > 0 else lang_all
        if isinstance(lang_single, bytes):
            lang_str = lang_single.decode("utf-8")
        else:
            lang_str = str(lang_single)

        # Get feasibility labels
        task_feasible = np.asarray(traj["task_feasible"])
        if task_feasible.ndim == 0:
            task_feasible = np.full((traj_len,), task_feasible)

        return {
            "images": images,  # (T, 1, H, W, C)
            "traj_len": traj_len,
            "lang_str": lang_str,
            "task_feasible": task_feasible,  # (T,)
        }

    def plot_octo_trajectory(state, traj_data, text_processor_fn):
        """Plot feasibility predictions over trajectory for Octo model.

        Args:
            state: TrainState with model params
            traj_data: Output from process_trajectory_for_plot
            text_processor_fn: Function to process text for Octo model
        """
        images = traj_data["images"]  # (T, 1, H, W, C)
        traj_len = traj_data["traj_len"]
        lang_str = traj_data["lang_str"]
        task_feasible = traj_data["task_feasible"]

        # Process each timestep as batch element
        # Create batch with T elements (one per timestep)
        batch_for_eval = {
            "observation": {
                "image_primary": images,  # (T, 1, H, W, C)
                "timestep_pad_mask": np.ones((traj_len, 1), dtype=bool),
            },
            "task": {
                "language_instruction": np.array([lang_str.encode("utf-8")] * traj_len),
            },
            "task_feasible": task_feasible[:, None],  # (T, 1)
            "action": np.zeros((traj_len, 1, 4, 7)),  # Dummy action
            "action_pad_mask": np.ones((traj_len, 1, 4), dtype=bool),
        }

        # Process through text processor
        batch_for_eval = process_text(batch_for_eval, text_processor_fn)

        # Get model predictions
        rng = jax.random.PRNGKey(0)
        _, metrics = loss_fn(state.model.params, batch_for_eval, rng, train=False)
        feasibility_metrics = metrics["feasibility"]

        # Extract probabilities - must exist, fail fast if not
        probs = np.array(feasibility_metrics["probs"]).squeeze()  # (T, 1, 1) -> (T,)

        # Create plot
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), dpi=150)
        canvas = FigureCanvas(fig)

        # Row 0: Trajectory images
        images_squeezed = images.squeeze()  # (T, H, W, C)
        num_images = min(16, traj_len)
        interval = max(1, traj_len // num_images)
        sel_images = images_squeezed[::interval][:num_images]
        sel_images_concat = np.concatenate([img for img in sel_images], axis=1)
        axs[0].imshow(sel_images_concat)
        axs[0].set_title(f"Trajectory ({num_images}/{traj_len} frames)")
        axs[0].axis('off')

        # Row 1: Language instruction
        axs[1].text(0.5, 0.5, lang_str, fontsize=12, ha='center', va='center', wrap=True)
        axs[1].set_title("Language Instruction")
        axs[1].axis('off')

        # Row 2: Feasibility over time
        timesteps = np.arange(traj_len)
        axs[2].plot(timesteps, probs, 'b-o', label='P(feasible)', linewidth=2)
        axs[2].plot(timesteps, task_feasible, 'g--s', label='Ground Truth', alpha=0.7)
        axs[2].axhline(y=0.5, color='r', linestyle=':', alpha=0.5)
        axs[2].set_xlabel("Timestep")
        axs[2].set_ylabel("Probability / Label")
        axs[2].set_ylim(-0.1, 1.1)
        axs[2].legend()
        axs[2].set_title("Feasibility Prediction vs Ground Truth")

        plt.tight_layout()
        canvas.draw()
        plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return wandb_lib.Image(plot_image)

    # Create trajectory iterator for plotting (separate from batch evaluation)
    val_traj_plot_iterator = map(
        process_trajectory_for_plot,
        val_bridge_with_neg.shuffle(1000).repeat().iterator()
    )

    # Run finetuning loop with validation
    logging.info("Starting finetuning...")
    timer = Timer()
    
    for i in tqdm.tqdm(range(FLAGS.config.num_steps), total=FLAGS.config.num_steps, dynamic_ncols=True):
        timer.tick("dataset")
        batch = get_training_batch()
        timer.tock("dataset")

        # DEBUG: Log first batch BEFORE JIT boundary
        if i == 0:
            logging.info("=== FIRST BATCH DEBUG INFO (before JIT) ===")
            logging.info(f"batch['task_feasible'] shape: {batch['task_feasible'].shape}")
            logging.info(f"batch['task_feasible'] sample (first 10): {batch['task_feasible'][:10]}")
            logging.info(f"batch['task_feasible'] mean: {jnp.mean(batch['task_feasible']):.3f}")
            logging.info(f"batch['observation']['timestep_pad_mask'] shape: {batch['observation']['timestep_pad_mask'].shape}")
            logging.info(f"batch['observation']['timestep_pad_mask'] sample (first 5): {batch['observation']['timestep_pad_mask'][:5]}")
            logging.info(f"batch['observation']['timestep_pad_mask'] all True? {jnp.all(batch['observation']['timestep_pad_mask'])}")
            logging.info(f"batch['action'] shape: {batch['action'].shape}")
            logging.info(f"batch['action_pad_mask'] shape: {batch['action_pad_mask'].shape}")

        timer.tick("train")
        train_state, update_info = train_step(train_state, batch)
        timer.tock("train")
        
        # Log training metrics and timer info
        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)

            # Flatten and log with proper structure
            training_metrics = flax.traverse_util.flatten_dict({"training": update_info}, sep="/")

            # Also log timer
            training_metrics.update({"timer": timer.get_average_times()})

            wandb_logger.log(training_metrics, step=i)
            
        # Run validation
        if (i) % FLAGS.config.eval_interval == 0:
            logging.info(f"Running validation at step {i+1}...")
            timer.tick("val")

            validation_metrics = {}
            
            # Evaluate on bridge validation dataset
            bridge_metrics = evaluate_on_dataset(
                train_state, val_bridge_traj_iter, "bridge", num_batches=30
            )
            validation_metrics.update(bridge_metrics)
                        # Evaluate on fractal validation dataset  
            # fractal_metrics = evaluate_on_dataset(
            #     train_state, val_fractal_traj_iter, "fractal", num_batches=30
            # )
            # validation_metrics.update(fractal_metrics)

            # Log validation metrics under 'validation' group
            wandb_logger.log(
                flax.traverse_util.flatten_dict({"validation": validation_metrics}, sep="/"),
                step=i,
            )

            # Trajectory plotting (every 10k steps within eval interval)
            if i % 10000 == 0:
                try:
                    logging.info("Plotting trajectory feasibility...")
                    traj_data = next(val_traj_plot_iterator)
                    plot = plot_octo_trajectory(train_state, traj_data, text_processor)
                    wandb_logger.log({"plots/trajectory": plot}, step=i)
                except Exception as e:
                    logging.warning(f"Trajectory plotting failed: {e}")

            logging.info("Validation completed")
            timer.tock("val")
            
        # Save checkpoint
        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info(f"Saving checkpoint at step {i+1}...")
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.config.save_dir)
    
    # Final save
    train_state.model.save_pretrained(step=FLAGS.config.num_steps, checkpoint_path=FLAGS.config.save_dir)
    
    logging.info("Training completed")
    
    # Final validation run
    logging.info("Running final validation...")
    final_validation_metrics = {}
    
    final_bridge_metrics = evaluate_on_dataset(
        train_state, val_bridge_traj_iter, "bridge", num_batches=100
    )
    final_validation_metrics.update(final_bridge_metrics)
    # final_fractal_metrics = evaluate_on_dataset(
    #     train_state, val_fractal_traj_iter, "fractal", num_batches=100
    # )
    # final_validation_metrics.update(final_fractal_metrics)
    
    wandb_logger.log(
        flax.traverse_util.flatten_dict({"validation": final_validation_metrics}, sep="/"),
        step=FLAGS.config.num_steps
    )


if __name__ == "__main__":
    app.run(main)
