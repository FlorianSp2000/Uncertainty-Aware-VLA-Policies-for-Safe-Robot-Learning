# SARSA critic agent built on the CQL agent of V-GPS/jaxrl_m (https://github.com/nakamotoo/V-GPS).
"""
Enhanced SARSA Agent with vectorized ensemble creation support
"""
import copy
from functools import partial
import textwrap
from typing import Optional, Union, Tuple
import chex
from absl import logging

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper, GCEncodingWrapper, LCEncodingWrapper, SigLIPLCEncodingWrapper
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.networks.actor_critic_nets import Critic, ensemblize
from jaxrl_m.networks.mlp import MLP

# For plotting (optional)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class SARSAEnsembleAgent(flax.struct.PyTreeNode):
    """
    Enhanced SARSA agent with vectorized ensemble creation support.
    
    This class extends the standard SARSA agent with optimized ensemble creation
    that can initialize multiple agents in parallel using JAX vectorization.
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

    def forward_critic(self, observations: Union[Data, Tuple[Data, Data]], actions: jax.Array, rng: PRNGKey,
                      *, grad_params: Optional[Params] = None, train: bool = True) -> jax.Array:
        """Forward pass for critic network"""
        if train:
            assert rng is not None, "Must specify rng when training"
        # logging.info(f"rng passed to forward_critic: {rng}")
        # logging.info(f"shape of actions: {actions.shape}")
        if jnp.ndim(actions) == 3:
            # 3D case: (batch_size, num_ood_actions, action_dim) used for OOD actions
            q_values = jax.vmap(
                lambda a: self.state.apply_fn(
                    {"params": grad_params or self.state.params},
                    observations, a, name="critic",
                    rngs={"dropout": rng} if train else {},
                    train=train,
                ), in_axes=1, out_axes=-1, # in_axes=1 means the original function is run in parallel over the second dimension (num_ood_actions)
                # .state.apply_fn will be run once for each action in num_ood_actions dimension
            )(actions)
            chex.assert_shape(q_values, (self.config["critic_ensemble_size"], actions.shape[0], actions.shape[1]))
            return q_values   
        else:         
            return self.state.apply_fn(
                {"params": grad_params or self.state.params},
                observations, actions, name="critic",
                rngs={"dropout": rng} if train else {},
                train=train,
            )

    @staticmethod
    def _extract_obs_encoding(intermediates):
        """Extract obs_encoding from intermediates dict, handling variable Flax nesting."""
        def _find_obs_encoding(d):
            if isinstance(d, dict):
                if "obs_encoding" in d:
                    val = d["obs_encoding"]
                    return val[0] if isinstance(val, (tuple, list)) else val
                for v in d.values():
                    result = _find_obs_encoding(v)
                    if result is not None:
                        return result
            return None
        result = _find_obs_encoding(intermediates)
        assert result is not None, f"obs_encoding not found in intermediates. Keys: {jax.tree_util.tree_structure(intermediates)}"
        return result

    def forward_critic_with_embeddings(self, observations: Union[Data, Tuple[Data, Data]], actions: jax.Array, rng: PRNGKey,
                                       *, grad_params: Optional[Params] = None, train: bool = True):
        """Forward pass returning (q_values, obs_encoding) via sow/mutable intermediates."""
        assert rng is not None, "Must specify rng when training with embeddings"
        params = grad_params or self.state.params
        q_values, state_updates = self.state.apply_fn(
            {"params": params},
            observations, actions, name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
            mutable=["intermediates"],
        )
        obs_enc = self._extract_obs_encoding(state_updates["intermediates"])
        return q_values, obs_enc

    def forward_target_critic(self, observations: Union[Data, Tuple[Data, Data]], actions: jax.Array, rng: PRNGKey) -> jax.Array:
        """Forward pass for target critic network"""
        return self.forward_critic(observations, actions, rng=rng, grad_params=self.state.target_params, train=False)

    def _sample_ood_actions(self, batch, rng: PRNGKey, num_ood_actions: int = 10):
        """Sample Out-of-Distribution (OOD) actions for evaluation."""
        batch_size = batch["rewards"].shape[0]
        action_dim = batch["actions"].shape[-1]
        
        rng, ood_rng = jax.random.split(rng)
        
        if self.config.get("ood_action_sample_method", "uniform") == "uniform":
            ood_actions = jax.random.uniform(ood_rng, shape=(batch_size, num_ood_actions, action_dim), minval=-1.0, maxval=1.0)
        else:
            ood_actions = jax.random.normal(ood_rng, shape=(batch_size, num_ood_actions, action_dim))
        
        chex.assert_shape(ood_actions, (batch_size, num_ood_actions, action_dim))
        return ood_actions

    @staticmethod
    def _contrastive_loss(obs_enc, task_feasible, margin: float = 1.0):
        """Triplet contrastive loss with hard mining on L2-normalized embeddings.

        obs_enc: (ensemble_inner, batch, embed_dim) or (batch, embed_dim)
        task_feasible: (batch,) binary labels
        """
        # If obs_enc has ensemble_inner dim, take mean across inner ensemble
        if obs_enc.ndim == 3:
            obs_enc = jnp.mean(obs_enc, axis=0)  # (batch, embed_dim)

        # Gradient-safe L2 normalize: divide by sqrt(||x||^2 + eps) not (||x|| + eps).
        # This bounds the gradient at O(1/sqrt(eps))=O(1e4) instead of O(1/eps)=O(1e8),
        # preventing explosion when FiLM encoder produces near-zero embeddings at init.
        obs_enc = obs_enc / jnp.sqrt(jnp.sum(obs_enc ** 2, axis=-1, keepdims=True) + 1e-8)

        # Pairwise L2 distances: (batch, batch)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b ≈ 2 - 2*a.b (for unit vectors)
        similarity = jnp.matmul(obs_enc, obs_enc.T)  # (batch, batch)
        # +1e-8 inside sqrt: prevents inf gradient at sqrt(0), which occurs on the
        # diagonal (sim=1.0 for identical vectors) and propagates even through jnp.where masks.
        dists = jnp.sqrt(jnp.maximum(2.0 - 2.0 * similarity, 0.0) + 1e-8)

        feasible_mask = task_feasible > 0.5  # (batch,) bool
        infeasible_mask = ~feasible_mask

        num_feasible = jnp.sum(feasible_mask)
        num_infeasible = jnp.sum(infeasible_mask)

        # For each feasible anchor: hardest positive = furthest feasible, hardest negative = nearest infeasible
        # Mask out self-distances and wrong-class distances
        large_val = 1e6

        # Positive distances: only feasible-to-feasible (exclude self)
        pos_mask = feasible_mask[None, :] & feasible_mask[:, None]  # (batch, batch)
        eye_mask = jnp.eye(obs_enc.shape[0], dtype=bool)
        pos_mask = pos_mask & ~eye_mask
        pos_dists = jnp.where(pos_mask, dists, -large_val)  # -inf for invalid
        hardest_pos = jnp.max(pos_dists, axis=1)  # (batch,) furthest positive

        # Negative distances: feasible anchor to infeasible
        # neg_mask[i,j] = True when i is feasible anchor and j is infeasible
        neg_mask = feasible_mask[:, None] & infeasible_mask[None, :]
        neg_dists = jnp.where(neg_mask, dists, large_val)  # +inf for invalid
        hardest_neg = jnp.min(neg_dists, axis=1)  # (batch,) nearest negative

        # Triplet loss: only for feasible anchors where both pos and neg exist
        triplet_loss = jnp.maximum(0.0, hardest_pos - hardest_neg + margin)

        # Valid triplets: feasible anchors that found both a positive and a negative
        has_valid_pos = jnp.any(pos_mask, axis=1)  # (batch,)
        has_valid_neg = jnp.any(neg_mask, axis=1)  # (batch,)
        valid_anchors = feasible_mask & has_valid_pos & has_valid_neg
        num_valid = jnp.sum(valid_anchors)

        # Safe mean: avoid NaN when no valid triplets
        loss = jnp.where(num_valid > 0, jnp.sum(triplet_loss * valid_anchors) / (num_valid + 1e-8), 0.0)

        info = {
            "contrastive_loss": loss,
            "contrastive_mean_pos_dist": jnp.where(num_valid > 0, jnp.sum(hardest_pos * valid_anchors) / (num_valid + 1e-8), 0.0),
            "contrastive_mean_neg_dist": jnp.where(num_valid > 0, jnp.sum(hardest_neg * valid_anchors) / (num_valid + 1e-8), 0.0),
        }
        return loss, info

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(self, batch: Batch, *, pmap_axis: str = None, networks_to_update: frozenset[str] = frozenset({"critic"})):
        """Update Q-function using SARSA target + optional contrastive loss"""
        contrastive_weight = self.config["contrastive_weight"]
        contrastive_margin = self.config["contrastive_margin"]

        def loss_fn(params, rng):
            batch_size = batch["rewards"].shape[0]
            obs = self._include_goals_in_obs(batch, "observations")

            # Current Q-values: Q(s, a)
            # TODO(siglip-perf): With frozen encoder (stop_gradient), online and target critics
            # produce identical encoder outputs — only the MLP heads differ. Currently we run
            # the full encoder (e.g. SigLIP 400M params) twice per step. Fix: compute encoder
            # output once, pass pre-encoded features to both online and target MLP heads.
            # This would nearly halve per-step time for large frozen backbones.
            if contrastive_weight > 0:
                current_q, obs_enc = self.forward_critic_with_embeddings(
                    obs, batch["actions"], rng=rng, grad_params=params, train=True,
                )
            else:
                current_q = self.forward_critic(
                    obs, batch["actions"], rng=rng, grad_params=params, train=True,
                )  # (ensemble_size, batch_size)

            # Target Q-values: Q_target(s', a') using ACTUAL next actions (SARSA)
            # TODO(siglip-perf): Same encoder redundancy — next_observations also encoded twice.
            rng, target_rng = jax.random.split(rng)
            target_next_q = self.forward_target_critic(
                self._include_goals_in_obs(batch, "next_observations"),
                batch["next_actions"], rng=target_rng,  # Key: Use actual next actions
            )  # (ensemble_size, batch_size)

            # Subsample ensemble if requested
            if self.config["critic_subsample_size"] is not None:
                rng, subsample_key = jax.random.split(rng)
                subsample_idcs = jax.random.randint(
                    subsample_key, (self.config["critic_subsample_size"],),
                    0, self.config["critic_ensemble_size"],
                )
                target_next_q = target_next_q[subsample_idcs]
            
            # Take minimum across ensemble for target (reduce overestimation)
            if self.config.get("use_min_q", True):
                target_next_min_q = target_next_q.min(axis=0)
            else:
                target_next_min_q = target_next_q.mean(axis=0)
            
            # SARSA target: r + γ * mask * Q_target(s', a')
            targets = (
                batch["rewards"] + self.config["discount"] * batch["masks"] * target_next_min_q
            )
            chex.assert_shape(targets, (batch_size,))
            # Broadcast targets to match current_q shape
            targets = targets[None].repeat(current_q.shape[0], axis=0)
            
            # MSE loss
            td_error = current_q - targets
            critic_loss = jnp.mean(td_error ** 2)
            
            info = {
                "critic_loss": critic_loss,
                "online_q": jnp.mean(current_q),
                "target_q": jnp.mean(targets),
                "td_err": jnp.mean(jnp.abs(td_error)),
                "rewards": jnp.mean(batch["rewards"]),
            }

            # Optional contrastive loss
            if contrastive_weight > 0:
                cl_loss, cl_info = self._contrastive_loss(obs_enc, batch["task_feasible"], contrastive_margin)
                total_loss = critic_loss + contrastive_weight * cl_loss
                info.update(cl_info)
                info["total_loss"] = total_loss
            else:
                total_loss = critic_loss

            return total_loss, info
        
        # Setup loss functions
        loss_fns = {"critic": partial(loss_fn)}
        
        # Only compute gradients for specified networks
        assert networks_to_update.issubset(loss_fns.keys())
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        # Update RNG
        rng, new_rng = jax.random.split(self.state.rng) # TODO: check that in update step each ensemble member gets its own rng
        
        # Apply loss functions
        new_state, info = self.state.apply_loss_fns(loss_fns, pmap_axis=pmap_axis, has_aux=True)
        # apply_loss_fns returns aux with the pytree structure of loss_fns,
        # i.e. {"critic": actual_info_dict}. Unwrap to get flat metrics dict.
        info = info["critic"]

        # Update target network
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])
        
        # Update RNG
        new_state = new_state.replace(rng=new_rng)
        
        # Log learning rates
        # for name, opt_state in new_state.opt_states.items():
        #     if hasattr(opt_state, "hyperparams") and "learning_rate" in opt_state.hyperparams.keys():
        #         info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]
        
        return self.replace(state=new_state), info

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        """Debug metrics for validation"""
        rng = jax.random.PRNGKey(0) # TODO: should we use passed rng or a fixed one? cql.py also uses PRNGKey(0) here
        
        # Current Q-values
        current_q = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            batch["actions"], rng=rng, train=False,
        )
        
        # Target Q-values  
        target_q = self.forward_target_critic(
            self._include_goals_in_obs(batch, "next_observations"),
            batch["next_actions"], rng=rng,
        )
        
        # Action OOD Q-values evaluation
        rng, ood_rng = jax.random.split(rng)
        ood_actions = self._sample_ood_actions(batch, ood_rng, num_ood_actions=self.config.get("num_ood_actions", 10))
        ood_q = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            ood_actions, rng=rng, train=False,
        )

        # Held-out TD loss with the training target, so validation shows when the fit stops
        # generalising rather than only where Q sits.
        target_next = target_q.min(axis=0) if self.config.get("use_min_q", True) else target_q.mean(axis=0)
        td_error = current_q - (batch["rewards"] + self.config["discount"] * batch["masks"] * target_next)[None]

        return {
            "critic_loss": jnp.mean(td_error ** 2),
            "td_err": jnp.mean(jnp.abs(td_error)),
            "online_q": jnp.mean(current_q),
            "target_q": jnp.mean(target_q),
            # "q_std": jnp.std(current_q), check if needed
            "rewards": jnp.mean(batch["rewards"]), # added averaging here - for some reason in original script this was done implicitly while averaging over multiple iterations
            "ood_q": jnp.mean(ood_q),
        }

    @jax.jit
    def get_contrastive_metrics(self, batch):
        """Compute contrastive distance metrics for a labeled batch (eval use only)."""
        rng = jax.random.PRNGKey(0)
        obs = self._include_goals_in_obs(batch, "observations")
        _, obs_enc = self.forward_critic_with_embeddings(obs, batch["actions"], rng=rng, train=False)
        _, cl_info = self._contrastive_loss(obs_enc, batch["task_feasible"], self.config["contrastive_margin"])
        return cl_info

    def get_q_values(self, observations, goals, actions): # TODO: this method is not used anywhere, but already this way in cql.py
        """Get Q-values for given state-action pairs"""
        obs = (observations, goals) if self.config["goal_conditioned"] else observations
        q_values = self.state.apply_fn({"params": self.state.target_params}, obs, actions, name="critic")
        return jnp.min(q_values, axis=0)  # Take min across ensemble

    def compute_monte_carlo_returns(self, rewards, masks, discount=None):
        """
        Compute Monte Carlo returns (reward-to-go) for trajectory using JAX scan.        
        """
        if discount is None:
            discount = self.config.get("discount", 0.98)
        
        def scan_fn(carry, inputs):
            # carry: G_{t+1} (return from future timesteps)
            # inputs: (reward_t, mask_t)
            reward_t, mask_t = inputs
            
            # G_t = r_t + discount * mask_t * G_{t+1}
            # Same formula as TD: rewards + discount * masks * next_value
            monte_carlo_return = reward_t + discount * mask_t * carry
            
            return monte_carlo_return, monte_carlo_return
        
        # Scan backwards through trajectory
        rewards_rev = jnp.flip(rewards)
        masks_rev = jnp.flip(masks)
        
        # Initial carry is 0.0 (no future return beyond trajectory)
        _, returns_rev = jax.lax.scan(scan_fn, 0.0, (rewards_rev, masks_rev))
        
        # Flip back to get forward-time order
        monte_carlo_returns = jnp.flip(returns_rev)
        
        return monte_carlo_returns

    def get_eval_values(self, traj, seed, goals):
        """Evaluate Q-values over trajectory - FULL VERSION
        TODO(siglip-perf): Runs encoder 4x (online+target × actions+ood_actions) on same obs.
        With frozen encoder, could encode once and reuse for all critic forward passes.
        """
        obs = (traj["observations"], goals) if self.config["goal_conditioned"] else traj["observations"]
        
        # Current Q-values
        q_raw = self.forward_critic(obs, traj["actions"], seed, train=False)
        q_values = jnp.min(q_raw, axis=0)

        # Target Q-values
        target_q_raw = self.forward_target_critic(obs, traj["actions"], seed)
        target_q_values = jnp.min(target_q_raw, axis=0)

        # OOD Q-values
        rng, ood_rng = jax.random.split(seed) if seed is not None else (jax.random.PRNGKey(42), jax.random.PRNGKey(43))
        batch_like = {"rewards": traj["rewards"], "actions": traj["actions"]}
        ood_actions = self._sample_ood_actions(batch_like, ood_rng, num_ood_actions=self.config.get("num_ood_actions", 10))

        ood_q = self.forward_critic(obs, ood_actions, rng, train=False)
        ood_q_values = jnp.mean(jnp.min(ood_q, axis=0), axis=-1)

        ood_target_q = self.forward_target_critic(obs, ood_actions, rng)
        ood_target_q_values = jnp.mean(jnp.min(ood_target_q, axis=0), axis=-1)

        # Monte Carlo returns (ground truth Q-function approximation)
        monte_carlo_returns = self.compute_monte_carlo_returns(traj["rewards"], traj["masks"])

        return {
            "q": q_values, 
            "target_q": target_q_values, 
            "q_ood": ood_q_values,
            "target_q_ood": ood_target_q_values, 
            "q_monte_carlo": monte_carlo_returns,  # Ground truth comparison
            "rewards": traj["rewards"], 
            "masks": traj["masks"],
        }

    def get_trajectory_metrics_for_member(self, traj, seed, goals):
        """Get trajectory metrics for a single ensemble member - filters get_eval_values"""
        full_metrics = self.get_eval_values(traj, seed, goals)
        
        # Return only the subset needed for trajectory metrics
        subset_keys = ["q", "q_ood", "q_monte_carlo"]
        return {k: full_metrics[k] for k in subset_keys if k in full_metrics}

    @staticmethod
    def adapt_goal_length(goals, traj):
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
                        if hasattr(v, 'shape'):  # JAX array
                            rep = jnp.repeat(v[-1:], num_repeat, axis=0)
                            goals[k] = jnp.concatenate([v, rep], axis=0)
                        elif isinstance(v, list):
                            rep = [v[-1]] * num_repeat
                            goals[k] = v + rep
        return goals

    @staticmethod
    def _detect_injection_indices(language_strs):
        """Detect FP injection points from per-timestep language strings.

        Returns:
            (majority_prompt, injection_indices, injection_prompts)
        """
        from collections import Counter
        if not language_strs or len(set(language_strs)) <= 1:
            return language_strs[0] if language_strs else "", [], {}

        counts = Counter(language_strs)
        majority = counts.most_common(1)[0][0]

        injection_indices = []
        injection_prompts = {}  # idx -> prompt
        for i, p in enumerate(language_strs):
            if p != majority:
                injection_indices.append(i)
                injection_prompts[i] = p

        return majority, injection_indices, injection_prompts

    @staticmethod
    def plot_ensemble_trajectory_values(ensemble_agents, ensemble_size, traj, seeds, goals=None,
                                        show_injection_timeline=False, plot_label=None):
        """Plot ensemble trajectory metrics.

            show_injection_timeline: If True (IVA), show per-step bar chart + orange injection markers.
                                     Defaults to False (bridge/fractal): prompt shown as image subtitle only.
            plot_label: Override prompt label. E.g. "Original" or "Swapped". None = auto from goals.
        """
        using_random_goals = goals is not None
        if plot_label is None:
            plot_label = "Swapped/Mismatched" if using_random_goals else "Original"
        goals = SARSAEnsembleAgent.adapt_goal_length(goals, traj)

        all_member_metrics = []
        
        for member_idx in range(ensemble_size):
            single_agent = jax.tree_map(lambda x: x[member_idx], ensemble_agents)
            
            member_metrics = single_agent.get_trajectory_metrics_for_member(traj, seeds[member_idx], goals)
            all_member_metrics.append(member_metrics)

        # Compute ensemble statistics
        ensemble_stats = {}
        # Compute mean using tree_map - much cleaner!
        ensemble_stats.update({
            f"{k}_mean": v for k, v in 
            jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *all_member_metrics).items()
        })
        
        ensemble_stats.update({
            f"{k}_std": v for k, v in 
            jax.tree_map(lambda *xs: jnp.std(jnp.stack(xs), axis=0), *all_member_metrics).items()
        })        
        # Compute range (max - min) across ensemble members
        ensemble_stats.update({
            f"{k}_range": v for k, v in 
            jax.tree_map(lambda *xs: jnp.max(jnp.stack(xs), axis=0) - jnp.min(jnp.stack(xs), axis=0), *all_member_metrics).items()
        })
        ensemble_stats.update({
            f"{k}_min": v for k, v in 
            jax.tree_map(lambda *xs: jnp.min(jnp.stack(xs), axis=0), *all_member_metrics).items()
        })
        ensemble_stats.update({
            f"{k}_max": v for k, v in 
            jax.tree_map(lambda *xs: jnp.max(jnp.stack(xs), axis=0), *all_member_metrics).items()
        })

        # Create plot
        images = traj["observations"]["image"].squeeze()
        
        base_metrics = []
        for key in ensemble_stats.keys():
            if key.endswith('_mean'):
                base_metric = key[:-5]  # Remove '_mean'
                if f"{base_metric}_std" in ensemble_stats and f"{base_metric}_range" in ensemble_stats:  # Only if we have mean, std, and range
                    base_metrics.append(base_metric)
        base_metrics = sorted(list(set(base_metrics)))
        if "q_monte_carlo" in base_metrics:
                base_metrics.remove("q_monte_carlo")
                # Only show q_monte_carlo if not using random goals because reward structure should change
                if not using_random_goals:
                    base_metrics.append("q_monte_carlo")    # Hack to change order and add it at the end

        has_prompt_row = "language_str" in goals and show_injection_timeline
        num_metrics = len(base_metrics) + 1 + (1 if has_prompt_row else 0)
        fig, axs = plt.subplots(num_metrics, 1, figsize=(14, 3 * num_metrics))
        canvas = FigureCanvas(fig)
        
        current_row = 0
        
        # Get trajectory length info
        full_traj_length = images.shape[0]
        interval = max(1, images.shape[0] // 8)
        num_images_shown = len(range(0, images.shape[0], interval))

        # Plot images
        sel_images = images[::interval]
        sel_images = np.split(sel_images, sel_images.shape[0], 0)
        sel_images = [a.squeeze() for a in sel_images]
        sel_images = np.concatenate(sel_images, axis=1)
        axs[current_row].imshow(sel_images)
        dataset_name = ""
        if "dataset_name" in traj:
            entry = traj["dataset_name"][0]
            dataset_name = entry.decode("utf-8") if isinstance(entry, bytes) else str(entry)
        rr_score = ""
        if "roboreward_score" in traj:
            rr_score = f" [RR:{int(float(traj['roboreward_score'][0]))}/5]"
        img_title = f"[{dataset_name}]{rr_score} " if (dataset_name or rr_score) else ""
        img_title += f"Trajectory Images (showing {num_images_shown}/{full_traj_length} steps, every {interval}th frame)"
        # When no injection timeline, embed prompt as image subtitle
        if not show_injection_timeline and "language_str" in goals:
            prompt = goals["language_str"][0][:200]
            wrapped = textwrap.fill(prompt, width=80)
            img_title += f'\n[{plot_label}] "{wrapped}"'
        axs[current_row].set_title(img_title, fontsize=9)
        current_row += 1

        # Plot prompts with injection timeline (IVA only)
        if show_injection_timeline and "language_str" in goals:
            lang_strs = goals["language_str"]
            majority, inj_indices, inj_prompts = SARSAEnsembleAgent._detect_injection_indices(lang_strs)
            T = len(lang_strs)

            ax = axs[current_row]

            if using_random_goals:
                # Random/mismatched goals: entire trajectory is "wrong" -> all red
                ax.barh(0, T, height=0.5, color='#FF6B6B', label='Mismatched')
                title = f'({plot_label}) Prompt: "{majority[:50]}..."' if len(majority) > 50 else f'({plot_label}) Prompt: "{majority}"'
            else:
                # Original goals: green=matching, red=injection
                ax.barh(0, T, height=0.5, color='#90EE90', label='Original')
                for idx in inj_indices:
                    ax.barh(0, 1, left=idx, height=0.5, color='#FF6B6B')

                if len(majority) > 50:
                    title = f'{plot_label}: "{majority[:50]}..."'
                else:
                    title = f'{plot_label}: "{majority}"'
                if inj_indices:
                    inj_prompt = list(set(inj_prompts.values()))[0]
                    if len(inj_prompt) > 40:
                        inj_prompt = inj_prompt[:40] + "..."
                    title += f'\nInjection @ steps {inj_indices}: "{inj_prompt}"'

            ax.set_xlim(0, T)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('Timestep')
            ax.set_title(title, fontsize=10)
            ax.set_yticks([])
            current_row += 1

        # Detect injection indices for metric plots (IVA only)
        inj_indices = []
        if show_injection_timeline and "language_str" in goals:
            _, inj_indices, _ = SARSAEnsembleAgent._detect_injection_indices(goals["language_str"])

        # Plot ensemble metrics
        for base_metric in base_metrics:
            row = current_row
            current_row += 1

            mean_values = ensemble_stats[f"{base_metric}_mean"]
            std_values = ensemble_stats[f"{base_metric}_std"]
            range_values = ensemble_stats[f"{base_metric}_range"]
            min_values = ensemble_stats[f"{base_metric}_min"]
            max_values = ensemble_stats[f"{base_metric}_max"]

            time_steps = np.arange(len(mean_values))

            axs[row].plot(time_steps, mean_values, linestyle='-', marker='o', linewidth=2.5,
                        label=f'{base_metric} mean', color='blue')

            # Plot confidence band (mean ± std)
            axs[row].fill_between(time_steps,
                                mean_values - std_values,
                                mean_values + std_values,
                                alpha=0.3, color='blue', label=f'{base_metric} ± std')

            # Show min and max values as dashed lines
            axs[row].plot(time_steps, max_values, '--', alpha=0.6, color='red', linewidth=1, label=f'{base_metric} max')
            axs[row].plot(time_steps, min_values, '--', alpha=0.6, color='green', linewidth=1, label=f'{base_metric} min')

            axs[row].set_ylabel(base_metric)

            # Build title with overall + injection/non-injection stats
            title = f"Ensemble {base_metric}\n"
            title += f"overall: mean_std={np.mean(std_values):.3f}, mean_range={np.mean(range_values):.3f}"

            # Add injection vs non-injection breakdown if injections exist
            if inj_indices:
                inj_set = set(inj_indices)
                non_inj_indices = [i for i in range(len(std_values)) if i not in inj_set]

                inj_std = np.mean([std_values[i] for i in inj_indices if i < len(std_values)])
                non_inj_std = np.mean([std_values[i] for i in non_inj_indices]) if non_inj_indices else 0.0
                inj_range = np.mean([range_values[i] for i in inj_indices if i < len(range_values)])
                non_inj_range = np.mean([range_values[i] for i in non_inj_indices]) if non_inj_indices else 0.0

                title += f"\ninjection({len(inj_indices)}): std={inj_std:.3f}, range={inj_range:.3f}"
                title += f" | non-inj({len(non_inj_indices)}): std={non_inj_std:.3f}, range={non_inj_range:.3f}"

            axs[row].set_title(title, fontsize=9)
            axs[row].grid(True, alpha=0.3)
            axs[row].legend(loc='upper right', fontsize=8)

            # Mark injection steps with vertical lines (orange to distinguish from red max line)
            for idx in inj_indices:
                axs[row].axvline(x=idx, color='orange', linestyle='--', alpha=0.8, linewidth=2)

            # Color backgrounds based on metric type
            if 'ood' in base_metric:
                axs[row].set_facecolor('#fff5f5')  # Light red for OOD
            else:
                axs[row].set_facecolor('#f5fff5')  # Light green

        plt.tight_layout()
        canvas.draw()
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return out_image
    
    def plot_values(self, traj, seed=None, goals=None):
        """Plot Q-values over trajectory"""
        # from absl import logging
        goals = SARSAEnsembleAgent.adapt_goal_length(goals, traj)

        metrics = self.get_eval_values(traj, seed, goals)
        images = traj["observations"]["image"].squeeze()
        
        num_rows = len(metrics.keys()) + 2 # +1 for images, +1 for prompt
        fig, axs = plt.subplots(num_rows, 1, figsize=(10, 16))
        canvas = FigureCanvas(fig)
        
        current_row = 0
        full_traj_length = images.shape[0]
        interval = max(1, images.shape[0] // 8)
        num_images_shown = len(range(0, images.shape[0], interval))

        # Plot images
        sel_images = images[::interval]
        sel_images = np.split(sel_images, sel_images.shape[0], 0)
        sel_images = [a.squeeze() for a in sel_images]
        sel_images = np.concatenate(sel_images, axis=1)
        axs[current_row].imshow(sel_images)
        # axs[current_row].set_title("Trajectory Images")
        axs[current_row].set_title(f"Trajectory Images (showing {num_images_shown}/{full_traj_length} steps, every {interval}th frame)")
        current_row += 1

        # Plot prompts with injection timeline
        if "language_str" in goals:
            lang_strs = goals["language_str"]
            majority, inj_indices, inj_prompts = SARSAEnsembleAgent._detect_injection_indices(lang_strs)
            T = len(lang_strs)

            ax = axs[current_row]
            # Draw timeline bar: green=original, red=injection
            ax.barh(0, T, height=0.5, color='#90EE90', label='Original')
            for idx in inj_indices:
                ax.barh(0, 1, left=idx, height=0.5, color='#FF6B6B')

            ax.set_xlim(0, T)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('Timestep')

            # Build title with prompt info
            if len(majority) > 50:
                title = f'Original: "{majority[:50]}..."'
            else:
                title = f'Original: "{majority}"'
            if inj_indices:
                inj_prompt = list(set(inj_prompts.values()))[0]
                if len(inj_prompt) > 40:
                    inj_prompt = inj_prompt[:40] + "..."
                title += f'\nInjection @ steps {inj_indices}: "{inj_prompt}"'
            ax.set_title(title, fontsize=10)
            ax.set_yticks([])
            current_row += 1

        # Plot metrics
        for i, (key, values) in enumerate(metrics.items()):
            row = i + current_row
            axs[row].plot(values, linestyle='--', marker='o')
            axs[row].set_ylabel(key)
            axs[row].set_title(f"{key} over time")

            if key in ['q', 'target_q', 'q_ood', 'target_q_ood']:
                axs[row].set_xlabel('Time Step')
                if 'ood' in key:
                    axs[row].set_facecolor('#fff5f5')  # Light red for OOD
                else:
                    axs[row].set_facecolor('#f5fff5')  # Light green

            # Mark injection steps with vertical lines
            if "language_str" in goals:
                _, inj_indices, _ = SARSAEnsembleAgent._detect_injection_indices(goals["language_str"])
                for idx in inj_indices:
                    axs[row].axvline(x=idx, color='red', linestyle='--', alpha=0.6, linewidth=1.5)

        plt.tight_layout()
        canvas.draw()
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return out_image

    @classmethod
    def _create_encoder_def(cls, encoder_def: nn.Module, use_proprio: bool, enable_stacking: bool,
                          goal_conditioned: bool, early_goal_concat: bool, shared_goal_encoder: bool,
                          language_conditioned: bool, use_siglip_wrapper: bool = False,
                          freeze_siglip_backbone: bool = True):
        """Create encoder definition"""
        if goal_conditioned and not language_conditioned:
            if early_goal_concat:
                goal_encoder_def = None
            else:
                goal_encoder_def = encoder_def if shared_goal_encoder else copy.deepcopy(encoder_def)

            encoder_def = GCEncodingWrapper(
                encoder=encoder_def, goal_encoder=goal_encoder_def, use_proprio=use_proprio, stop_gradient=False,
            )
        elif language_conditioned and use_siglip_wrapper:
            # SigLIP: no FiLM — late fusion via concat + projection
            # freeze_siglip_backbone=True freezes via stop_gradient (Phase 1 PoC)
            encoder_def = SigLIPLCEncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=freeze_siglip_backbone,
                projection_dim=512,
            )
        elif language_conditioned:
            if shared_goal_encoder is not None or early_goal_concat is not None:
                raise ValueError("If language conditioned, shared_goal_encoder and early_goal_concat must not be set")
            encoder_def = LCEncodingWrapper(encoder=encoder_def, use_proprio=use_proprio, stop_gradient=False)
        else:
            encoder_def = EncodingWrapper(encoder_def, use_proprio=use_proprio, stop_gradient=False, enable_stacking=enable_stacking)

        return encoder_def

    # ==== VECTORIZED ENSEMBLE CREATION METHODS ====
    
    @staticmethod
    @partial(jax.vmap, in_axes=(0, None, None, None))
    def _create_single_agent_params(rng, network_input, actions, model_def):
        """
        Vectorized parameter initialization for ensemble members.
        This function is vmapped over different random seeds to create diverse parameters.
        """
        params = model_def.init(rng, critic=[network_input, actions])["params"]
        return params

    @staticmethod  
    @partial(jax.vmap, in_axes=(0, 0, None, None)) # each agent will have its own rng and params but share the same model_def and txs
    def _create_single_agent_state(rng, params, model_def, txs):
        """
        Vectorized state creation for ensemble members.
        Creates training states with the same optimizer but different parameters.
        """
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply, params=params, txs=txs,
            target_params=params, rng=rng,  # Initialize target = current
        )
        return state

    @classmethod
    def create_ensemble_vectorized(
        cls, base_rng: PRNGKey, ensemble_size: int, observations: Data, actions: jnp.ndarray,
        encoder_def: nn.Module, goals: Optional[Data] = None, **agent_kwargs
    ):
        """
        Fully vectorized ensemble creation using JAX vmap operations.
        This method creates all ensemble members in parallel, significantly reducing initialization time.
        """
        # from absl import logging
        # Split RNG for all ensemble members at once
        logging.info(f"base_rng: {base_rng}, ensemble_size: {ensemble_size}")
        agent_rngs = jax.random.split(base_rng, ensemble_size + 1)
        logging.info(f"agent_rngs: {agent_rngs}")
        init_rngs = agent_rngs[:-1]  # Use first N for initialization

        state_rngs = jax.random.split(agent_rngs[-1], ensemble_size)  # Use last for state creation
        logging.info(f"state_rngs: {state_rngs}")
        # Extract configuration
        goal_conditioned = agent_kwargs.get("goal_conditioned", False)
        language_conditioned = agent_kwargs.get("language_conditioned", False)
        
        if language_conditioned:
            goal_conditioned = True
            
        # Create encoder (same for all members)
        encoder_def = cls._create_encoder_def(
            encoder_def, use_proprio=False, enable_stacking=False, goal_conditioned=goal_conditioned,
            early_goal_concat=agent_kwargs.get("early_goal_concat", False),
            shared_goal_encoder=agent_kwargs.get("shared_goal_encoder", True),
            language_conditioned=language_conditioned,
            use_siglip_wrapper=agent_kwargs.get("use_siglip_wrapper", False),
            freeze_siglip_backbone=agent_kwargs.get("freeze_siglip_backbone", True),
        )

        # Create model definition (same for all members)
        network_kwargs = agent_kwargs.get("network_kwargs", {"hidden_dims": [256, 256], "activate_final": True, "use_layer_norm": False})
        critic_ensemble_size = agent_kwargs.get("critic_ensemble_size", 2)
        
        contrastive_weight = agent_kwargs["contrastive_weight"]
        sow_embeddings = contrastive_weight > 0 or agent_kwargs["sow_embeddings"]

        critic_backbone = partial(MLP, **network_kwargs)
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(name="critic_ensemble")
        critic_def = partial(Critic, encoder=encoder_def, network=critic_backbone, sow_embeddings=sow_embeddings)(name="critic")
        
        networks = {"critic": critic_def}
        model_def = ModuleDict(networks)

        # Create optimizer (same for all members)
        learning_rate = agent_kwargs.get("learning_rate", 3e-4)
        warmup_steps = agent_kwargs.get("warmup_steps", 1000)
        txs = {"critic": make_optimizer(learning_rate=learning_rate, warmup_steps=warmup_steps)}

        # Prepare network input
        network_input = (observations, goals) if goal_conditioned else observations

        # Vectorized parameter initialization - creates all ensemble member parameters in parallel
        ensemble_params = cls._create_single_agent_params(init_rngs, network_input, actions, model_def)
        # logging.info(f"ensemble_params: {jax.tree_util.tree_structure(ensemble_params)}")
        # Vectorized state creation - creates all training states in parallel
        ensemble_states = cls._create_single_agent_state(state_rngs, ensemble_params, model_def, txs)
        # logging.info(f"ensemble_states: {jax.tree_util.tree_structure(ensemble_states)}")
        # logging.info(f"ensemble_states shape: {jax.tree_map(lambda x: x.shape, ensemble_states)}")

        # Create configuration (same for all members)
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

        # Create ensemble by vmapping the agent constructor over states
        def create_single_agent(state):
            return cls(state=state, config=config)
        
        # Vectorized agent creation - all agents created simultaneously
        return jax.vmap(create_single_agent)(ensemble_states) # so ensemble is PyTree of agents, each with its own state