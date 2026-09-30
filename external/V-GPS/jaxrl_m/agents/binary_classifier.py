# Classifier agent following the agent pattern of V-GPS/jaxrl_m (https://github.com/nakamotoo/V-GPS).
"""
Binary Classifier Agent
Uses SARSA's ResNet+MUSE architecture for binary feasibility classification.
Replaces Q-value head with binary classification head.
"""
import copy
from functools import partial
from typing import Optional, Union, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper, GCEncodingWrapper, LCEncodingWrapper
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.networks.mlp import MLP, default_init

# For plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class BinaryClassifier(nn.Module):
    """Binary classification network (replaces Critic).

    Takes observations only (no actions) and outputs binary logit.
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        # Encode observations
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)

        # MLP backbone
        outputs = self.network(obs_enc, train=train)

        # Output logit (BCE loss applies sigmoid)
        if self.init_final is not None:
            logit = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            logit = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(logit, -1)  # (batch_size,)


class BinaryClassifierAgent(flax.struct.PyTreeNode):
    """
    Binary classifier agent for feasibility prediction.
    Uses same encoder architecture as SARSA but outputs binary classification.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def _include_goals_in_obs(self, batch, which_obs: str):
        """Include goals in observations if goal-conditioned."""
        assert which_obs in ("observations", "next_observations")
        obs = batch[which_obs]
        if self.config["goal_conditioned"]:
            obs = (obs, batch["goals"])
        return obs

    def forward_classifier(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """Forward pass for classifier network."""
        if train:
            assert rng is not None, "Must specify rng when training"

        logits = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="classifier",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

        # Shape check: logits should be (batch_size,)
        chex.assert_rank(logits, 1)

        return logits

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: str = None,
        networks_to_update: frozenset[str] = frozenset({"classifier"}),
    ):
        """Update classifier using BCE loss."""

        def loss_fn(params, rng):
            # Get labels (0 = infeasible, 1 = feasible) - shape (batch, 1)
            labels = batch["task_feasible"].astype(jnp.float32).squeeze(-1)  # (batch,)
            batch_size = labels.shape[0]

            # Forward pass - no actions
            logits = self.forward_classifier(
                self._include_goals_in_obs(batch, "observations"),
                rng=rng,
                grad_params=params,
                train=True,
            )  # (batch_size,)

            # Shape assertions
            chex.assert_shape(logits, (batch_size,))
            chex.assert_shape(labels, (batch_size,))

            # Binary cross-entropy loss
            bce_loss = optax.sigmoid_binary_cross_entropy(logits, labels)
            classifier_loss = jnp.mean(bce_loss)

            # Compute metrics
            probs = jax.nn.sigmoid(logits)
            preds = (probs > 0.5).astype(jnp.float32)
            accuracy = jnp.mean(preds == labels)

            info = {
                "classifier_loss": classifier_loss,
                "accuracy": accuracy,
                "mean_prob": jnp.mean(probs),
            }

            return classifier_loss, info

        # Setup loss functions
        loss_fns = {"classifier": partial(loss_fn)}

        # Only compute gradients for specified networks
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"Invalid networks: {networks_to_update}"
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        # Update RNG
        rng, new_rng = jax.random.split(self.state.rng)

        # Apply loss functions and compute gradient/update norms
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # Compute grad/update/param norms (only for classifier network)
        if "classifier" in networks_to_update:
            # Get gradients that were just computed
            (_, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                self.state.params, rng
            )
            grad_norm = optax.global_norm(grads)

            # Compute updates (what optimizer will apply)
            updates, _ = self.state.txs["classifier"].update(
                grads, self.state.opt_states["classifier"], self.state.params
            )
            update_norm = optax.global_norm(updates)
            param_norm = optax.global_norm(self.state.params)

            info.update({
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm,
            })

        # Update RNG
        new_state = new_state.replace(rng=new_rng)

        # Log learning rate
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info["learning_rate"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        """Debug metrics for validation."""
        rng = jax.random.PRNGKey(0)

        # Forward pass
        logits = self.forward_classifier(
            self._include_goals_in_obs(batch, "observations"),
            rng=rng,
            train=False,
        )

        # Compute metrics
        probs = jax.nn.sigmoid(logits)
        labels = batch["task_feasible"].astype(jnp.float32).squeeze(-1)  # (batch,)
        preds = (probs > 0.5).astype(jnp.float32)

        accuracy = jnp.mean(preds == labels)

        # Conditional probabilities - avoid JIT issues with jnp.maximum
        prob_feasible = jnp.sum(jnp.where(labels == 1.0, probs, 0.0)) / jnp.maximum(jnp.sum(labels == 1.0), 1.0)
        prob_infeasible = jnp.sum(jnp.where(labels == 0.0, probs, 0.0)) / jnp.maximum(jnp.sum(labels == 0.0), 1.0)

        return {
            "accuracy": accuracy,
            "mean_prob": jnp.mean(probs),
            "prob_given_feasible": prob_feasible,
            "prob_given_infeasible": prob_infeasible,
        }

    def get_eval_values(self, traj, seed, goals):
        """Evaluate probabilities over trajectory.

        Handles trajectory data (T, H, W, C) by processing each timestep
        as a batch element, with language repeated for all timesteps.
        """
        images = traj["observations"]["image"]  # (T, H, W, C)
        traj_len = images.shape[0]

        # Language: if shape is (1, embed_dim), repeat for all timesteps
        lang = goals["language"]
        if lang.shape[0] == 1 and traj_len > 1:
            lang = jnp.repeat(lang, traj_len, axis=0)  # (T, embed_dim)

        # Create batch-like structure for _include_goals_in_obs
        batch = {
            "observations": {"image": images},  # (T, H, W, C) treated as (B, H, W, C)
            "goals": {"language": lang},  # (T, embed_dim)
        }

        # Get probabilities - forward pass treats T as batch dimension
        logits = self.forward_classifier(
            self._include_goals_in_obs(batch, "observations"),
            seed,
            train=False,
        )
        probs = jax.nn.sigmoid(logits)

        return {
            "prob": probs,
            "task_feasible": traj["task_feasible"],
        }

    def plot_values(self, traj, seed=None, goals=None):
        """Plot probabilities over trajectory."""
        from absl import logging
        if goals is None:
            goals = traj["goals"]
        else:
            # Handle goal length mismatch
            traj_len = traj["observations"]["image"].shape[0]
            if goals["language"].shape[0] != traj_len:
                if goals["language"].shape[0] > traj_len:
                    goals = {k: v[:traj_len] for k, v in goals.items()}
                else:
                    num_repeat = traj_len - goals["language"].shape[0]
                    for k, v in goals.items():
                        rep = jnp.repeat(v[-1:], num_repeat, axis=0)
                        goals[k] = jnp.concatenate([v, rep], axis=0)

        metrics = self.get_eval_values(traj, seed, goals)
        images = traj["observations"]["image"].squeeze()

        # Determine number of rows (add prompt row if language_str available)
        num_rows = 4 if "language_str" in goals else 3
        fig, axs = plt.subplots(num_rows, 1, figsize=(10, 16), dpi=150)
        canvas = FigureCanvas(fig)
        current_row = 0

        # Row 0: Plot images (show more frames for better resolution)
        full_traj_length = images.shape[0]
        num_images_to_show = min(16, full_traj_length)  # Show up to 16 frames
        interval = max(1, full_traj_length // num_images_to_show)
        sel_images = images[::interval][:num_images_to_show]
        sel_images = np.split(sel_images, sel_images.shape[0], 0)
        sel_images = [a.squeeze() for a in sel_images]
        sel_images = np.concatenate(sel_images, axis=1)
        axs[current_row].imshow(sel_images)
        axs[current_row].set_title(f"Trajectory Images (showing {len(sel_images)}/{full_traj_length} frames, every {interval}th)")
        axs[current_row].axis('off')
        current_row += 1

        # Row 1 (optional): Plot language prompt
        if "language_str" in goals:
            unique_prompts = set(goals["language_str"])
            axs[current_row].text(0.5, 0.5, "\n".join(unique_prompts), fontsize=12, ha='center', va='center')
            axs[current_row].set_title("Language Instruction")
            axs[current_row].axis('off')
            current_row += 1

        # Row 2: Plot probabilities
        probs = np.array(metrics["prob"])
        axs[current_row].plot(probs, linestyle='--', marker='o', color='blue')
        axs[current_row].axhline(y=0.5, color='r', linestyle='-', alpha=0.5)
        axs[current_row].set_ylabel("P(feasible)")
        axs[current_row].set_title("Feasibility Probability")
        axs[current_row].set_ylim(0, 1)
        current_row += 1

        # Row 3: Plot ground truth labels
        labels = np.array(metrics["task_feasible"])
        axs[current_row].plot(labels, linestyle='-', marker='s', color='green')
        axs[current_row].set_ylabel("Label")
        axs[current_row].set_title("Ground Truth (1=feasible, 0=infeasible)")
        axs[current_row].set_ylim(-0.1, 1.1)
        axs[current_row].set_xlabel("Time Step")

        plt.tight_layout()
        canvas.draw()
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return out_image

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        # Example arrays for model init
        observations: Data,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = False,
        # Goal conditioning
        goals: Optional[Data] = None,
        early_goal_concat: bool = False,
        shared_goal_encoder: bool = True,
        language_conditioned: bool = False,
        goal_conditioned: bool = False,
        # Network config
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        # Optimizer config
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        **kwargs,
    ):
        """Create binary classifier agent."""

        # Language conditioning requires goal conditioning
        if language_conditioned:
            goal_conditioned = True

        # Create encoder
        encoder_def = cls._create_encoder_def(
            encoder_def,
            use_proprio=False,
            enable_stacking=False,
            goal_conditioned=goal_conditioned,
            early_goal_concat=early_goal_concat,
            shared_goal_encoder=shared_goal_encoder,
            language_conditioned=language_conditioned,
        )

        encoders = {
            "classifier": encoder_def,
        }

        # Create classifier network
        classifier_backbone = partial(MLP, **network_kwargs)
        classifier_def = partial(
            BinaryClassifier, encoder=encoders["classifier"], network=classifier_backbone()
        )(name="classifier")

        networks = {"classifier": classifier_def}
        model_def = ModuleDict(networks)

        # Optimizer
        txs = {
            "classifier": make_optimizer(
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
            ),
        }

        # Initialize parameters - NOTE: no actions input
        rng, init_rng = jax.random.split(rng)
        network_input = (observations, goals) if goal_conditioned else observations
        params = model_def.init(
            init_rng,
            classifier=[network_input],
        )["params"]

        # Create training state (no target params needed)
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,  # Keep for compatibility but unused
            rng=create_rng,
        )

        # Configuration
        config = {
            "goal_conditioned": goal_conditioned,
            "language_conditioned": language_conditioned,
        }

        return cls(state=state, config=config)

    @classmethod
    def _create_encoder_def(
        cls,
        encoder_def: nn.Module,
        use_proprio: bool,
        enable_stacking: bool,
        goal_conditioned: bool,
        early_goal_concat: bool,
        shared_goal_encoder: bool,
        language_conditioned: bool,
    ):
        """Create encoder definition (same as SARSA)."""
        if goal_conditioned and not language_conditioned:
            if early_goal_concat:
                goal_encoder_def = None
            else:
                goal_encoder_def = (
                    encoder_def if shared_goal_encoder else copy.deepcopy(encoder_def)
                )

            encoder_def = GCEncodingWrapper(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )
        elif language_conditioned:
            if shared_goal_encoder is not None or early_goal_concat is not None:
                pass  # Allow but ignore these params for language conditioning
            encoder_def = LCEncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )
        else:
            encoder_def = EncodingWrapper(
                encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
                enable_stacking=enable_stacking,
            )

        return encoder_def
