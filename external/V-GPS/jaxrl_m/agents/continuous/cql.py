# From V-GPS (https://github.com/nakamotoo/V-GPS), jaxrl_m of bridge_data_v2 (https://github.com/rail-berkeley/bridge_data_v2), MIT. Unmodified.
"""
Implementation of CQL in continuous action spaces.
"""
import copy
from functools import partial
from typing import Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from overrides import overrides

from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import *
from jaxrl_m.networks.actor_critic_nets import Critic, Policy, ensemblize
from jaxrl_m.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from jaxrl_m.networks.mlp import MLP, Scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class ContinuousCQLAgent(SACAgent):
    @overrides
    def _sample_negative_goals(self, batch, rng):
        """for calql, adjust the mc_returns for negative goals"""
        new_stats, neg_goal_mask = super()._sample_negative_goals(batch, rng)
        if self.config["use_calql"]:
            assert "mc_returns" in batch
            new_mc_returns = jnp.where(
                neg_goal_mask, -1 / (1 - self.config["discount"]), batch["mc_returns"]
            )
            return {
                **new_stats,
                "mc_returns": new_mc_returns,
            }, neg_goal_mask
        else:
            return new_stats, neg_goal_mask

    def forward_cql_alpha_lagrange(self, *, grad_params: Optional[Params] = None):
        """
        Forward pass for the CQL alpha Lagrange multiplier
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            name="cql_alpha_lagrange",
        )

    def forward_policy_and_sample(
        self,
        obs: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        repeat=None,
    ):
        rng, sample_rng = jax.random.split(rng)
        action_dist = super().forward_policy(obs, rng, grad_params=grad_params)
        if repeat:
            new_actions, log_pi = action_dist.sample_and_log_prob(
                seed=sample_rng, sample_shape=repeat
            )
            new_actions = jnp.transpose(
                new_actions, (1, 0, 2)
            )  # (batch, repeat, action_dim)
            log_pi = jnp.transpose(log_pi, (1, 0))  # (batch, repeat)
        else:
            new_actions, log_pi = action_dist.sample_and_log_prob(seed=sample_rng)
        return new_actions, log_pi

    def _get_cql_q_diff(
        self, batch, rng: PRNGKey, grad_params: Optional[Params] = None
    ):
        """
        most of the CQL loss logic is here
        It is needed for both critic_loss_fn and cql_alpha_loss_fn
        """
        batch_size = batch["rewards"].shape[0]
        q_pred = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            batch["actions"],
            rng,
            grad_params=grad_params,
        )
        chex.assert_shape(q_pred, (self.config["critic_ensemble_size"], batch_size))

        """sample random actions"""
        action_dim = batch["actions"].shape[-1]
        rng, action_rng = jax.random.split(rng)
        if self.config["cql_action_sample_method"] == "uniform":
            cql_random_actions = jax.random.uniform(
                action_rng,
                shape=(batch_size, self.config["cql_n_actions"], action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        elif self.config["cql_action_sample_method"] == "normal":
            cql_random_actions = jax.random.normal(
                action_rng,
                shape=(batch_size, self.config["cql_n_actions"], action_dim),
            )
        else:
            raise NotImplementedError

        rng, current_a_rng, next_a_rng = jax.random.split(rng, 3)
        cql_current_actions, cql_current_log_pis = self.forward_policy_and_sample(
            self._include_goals_in_obs(batch, "observations"),
            current_a_rng,
            repeat=self.config["cql_n_actions"],
        )
        chex.assert_shape(
            cql_current_log_pis, (batch_size, self.config["cql_n_actions"])
        )

        cql_next_actions, cql_next_log_pis = self.forward_policy_and_sample(
            self._include_goals_in_obs(batch, "next_observations"),
            next_a_rng,
            repeat=self.config["cql_n_actions"],
        )

        all_sampled_actions = jnp.concatenate(
            [
                cql_random_actions,
                cql_current_actions,
                cql_next_actions,
            ],
            axis=1,
        )

        """q values of randomly sampled actions"""
        rng, q_rng = jax.random.split(rng)
        cql_q_samples = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            all_sampled_actions,  # this is being vmapped over in sac.py
            q_rng,
            grad_params=grad_params,
            train=True,
        )
        chex.assert_shape(
            cql_q_samples,
            (
                self.config["critic_ensemble_size"],
                batch_size,
                self.config["cql_n_actions"] * 3,
            ),
        )

        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            cql_q_samples = cql_q_samples[subsample_idcs]

        """Cal-QL"""
        if self.config["use_calql"]:
            mc_lower_bound = jnp.repeat(
                batch["mc_returns"].reshape(-1, 1),
                self.config["cql_n_actions"] * 2,
                axis=1,
            )
            chex.assert_shape(
                mc_lower_bound, (batch_size, self.config["cql_n_actions"] * 2)
            )

            cql_q_pi = cql_q_samples[:, :, self.config["cql_n_actions"]:]
            num_vals = jnp.size(cql_q_pi)
            calql_bound_rate = jnp.sum(cql_q_pi < mc_lower_bound) / num_vals
            cql_q_pi = jnp.maximum(cql_q_pi, mc_lower_bound)
            cql_q_samples = jnp.concatenate(
                [
                    cql_q_samples[:, :, :self.config["cql_n_actions"]],
                    cql_q_pi,
                ],
                axis=-1,
            )


        if self.config["cql_importance_sample"]:
            random_density = jnp.log(0.5**action_dim)

            importance_prob = jnp.concatenate(
                [
                    jnp.broadcast_to(
                        random_density, (batch_size, self.config["cql_n_actions"])
                    ),
                    cql_current_log_pis,
                    cql_next_log_pis,  # this order matters, should match all_sampled_actions
                ],
                axis=1,
            )
            cql_q_samples = cql_q_samples - importance_prob  # broadcast over dim 0
        else:
            cql_q_samples = jnp.concatenate(
                [
                    cql_q_samples,
                    jnp.expand_dims(q_pred, -1),
                ],
                axis=-1,
            )
            cql_q_samples -= jnp.log(cql_q_samples.shape[-1]) * self.config["cql_temp"]
            chex.assert_shape(
                cql_q_samples,
                (
                    self.config["critic_ensemble_size"],
                    batch_size,
                    3 * self.config["cql_n_actions"] + 1,
                ),
            )

        """log sum exp of the ood actions"""
        cql_ood_values = (
            jax.scipy.special.logsumexp(
                cql_q_samples / self.config["cql_temp"], axis=-1
            )
            * self.config["cql_temp"]
        )
        chex.assert_shape(
            cql_ood_values, (self.config["critic_ensemble_size"], batch_size)
        )

        cql_q_diff = cql_ood_values - q_pred
        info = {
            "cql_ood_values": cql_ood_values.mean(),
        }
        if self.config["use_calql"]:
            info["calql_bound_rate"] = calql_bound_rate

        return cql_q_diff, info

    @overrides
    def _compute_next_actions(self, batch, rng):
        """
        compute the next actions but with repeat cql_n_actions times
        this should only be used when calculating critic loss using
        cql_max_target_backup
        """
        sample_n_actions = (
            self.config["cql_n_actions"]
            if self.config["cql_max_target_backup"]
            else None
        )
        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            self._include_goals_in_obs(batch, "next_observations"),
            rng,
            repeat=sample_n_actions,
        )
        return next_actions, next_actions_log_probs

    @overrides
    def _process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """add cql_max_target_backup option"""

        if self.config["cql_max_target_backup"]:
            max_target_indices = jnp.expand_dims(
                jnp.argmax(target_next_qs, axis=-1), axis=-1
            )
            target_next_qs = jnp.take_along_axis(
                target_next_qs, max_target_indices, axis=-1
            ).squeeze(-1)
            next_actions_log_probs = jnp.take_along_axis(
                next_actions_log_probs, max_target_indices, axis=-1
            ).squeeze(-1)

        target_next_qs = super()._process_target_next_qs(
            target_next_qs,
            next_actions_log_probs,
        )

        return target_next_qs

    @overrides
    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey, train=True):
        """add CQL loss on top of SAC loss"""
        if self.config["use_td_loss"]:
            td_loss, td_loss_info = super().critic_loss_fn(batch, params, rng, train=train)
        else:
            td_loss, td_loss_info = 0.0, {}

        cql_q_diff, cql_intermediate_results = self._get_cql_q_diff(batch, rng, params)

        """auto tune cql alpha"""
        if self.config["cql_autotune_alpha"]:
            alpha = self.forward_cql_alpha_lagrange()
            cql_loss = (cql_q_diff - self.config["cql_target_action_gap"]).mean()
        else:
            alpha = self.config["cql_alpha"]
            cql_loss = jnp.clip(
                cql_q_diff,
                self.config["cql_clip_diff_min"],
                self.config["cql_clip_diff_max"],
            ).mean()

        critic_loss = td_loss + alpha * cql_loss

        info = {
            **td_loss_info,
            "critic_loss": critic_loss,
            "td_err": td_loss,
            "cql_loss": cql_loss,
            "cql_alpha": alpha,
            "cql_diff": cql_q_diff.mean(),
            **cql_intermediate_results,
        }

        return critic_loss, info

    def cql_alpha_lagrange_penalty(
        self, qvals_diff, *, grad_params: Optional[Params] = None
    ):
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=qvals_diff,
            rhs=self.config["cql_target_action_gap"],
            name="cql_alpha_lagrange",
        )

    def cql_alpha_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """recompute cql_q_diff without gradients (not optimal for runtime)"""
        cql_q_diff, _ = self._get_cql_q_diff(batch, rng)

        cql_alpha_loss = self.cql_alpha_lagrange_penalty(
            qvals_diff=cql_q_diff.mean(),
            grad_params=params,
        )
        lmbda = self.forward_cql_alpha_lagrange()

        return cql_alpha_loss, {
            "cql_alpha_loss": cql_alpha_loss,
            "cql_alpha_lagrange_multiplier": lmbda,
        }

    @overrides
    def loss_fns(self, batch):
        losses = super().loss_fns(batch)
        if self.config["cql_autotune_alpha"]:
            losses["cql_alpha_lagrange"] = partial(self.cql_alpha_loss_fn, batch)

        return losses

    def update(
        self,
        batch: Batch,
        pmap_axis: str = None,
        networks_to_update: set = set({"actor", "critic"}),
    ):
        """update super() to perhaps include updating CQL lagrange multiplier"""
        if self.config["autotune_entropy"]:
            networks_to_update.add("temperature")
        if self.config["cql_autotune_alpha"]:
            networks_to_update.add("cql_alpha_lagrange")

        return super().update(
            batch, pmap_axis=pmap_axis, networks_to_update=frozenset(networks_to_update)
        )

    def update_cql_alpha(self, new_alpha):
        """update the CQL alpha. Used for finetuning online with a different alpha"""
        object.__setattr__(
            self, "config", self.config.copy({"cql_alpha": new_alpha})
        )  # hacky way to update self.config because self is a frozen dataclass


    @overrides
    def get_debug_metrics(self, batch, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
        _, critic_metrics = self.critic_loss_fn(batch, self.state.params, rng=jax.random.PRNGKey(0), train=False)

        metrics = {
            "log_probs": log_probs,
            "mse": mse,
            "pi_actions": pi_actions,
            **critic_metrics,
        }
        return metrics

    @jax.jit
    def get_q_values(self, observations, goals, actions):
        q = self.state.apply_fn(
            {"params": self.state.target_params},
            (observations, goals),
            actions,
            name="critic",
        )
        
        q = jnp.min(q.squeeze(), axis=0)
        return q


    # @jax.jit
    def get_eval_values(self, traj, seed, goals):
        actions = self.sample_actions(
            observations=traj["observations"], goals=goals, argmax=True
        )
        mse = ((actions - traj["actions"]) ** 2).sum((-1))
        q = self.forward_critic(
            (traj["observations"], goals),
            traj["actions"],
            seed,
            train=False,
            # distributional_critic_return_logits=self.config["distributional_critic"]
        )
        
        target_q = self.forward_target_critic(
            (traj["observations"], goals),
            traj["actions"],
            seed,
        )

        q = jnp.min(q, axis=0)
        target_q = jnp.min(target_q, axis=0)

        
        metrics = {
            "q": q,
            "target_q": target_q,
            "mse": mse,
            "rewards": traj["rewards"],
            "masks": traj["masks"],
        }
        return metrics

    def plot_values(self, traj, seed=None, goals=None):
        if goals is None:
            goals = traj["goals"]
        else:
            traj_len = traj["observations"]["image"].shape[0]

            if goals["language"].shape[0] > traj_len:
                goals = {k: v[:traj_len] for k, v in goals.items()}
            elif goals["language"].shape[0] < traj_len:
                num_repeat = traj_len - goals["language"].shape[0]
                for k, v in goals.items():
                    rep = jnp.repeat(v[-1:], num_repeat, axis=0)
                    goals[k] = jnp.concatenate([v, rep], axis=0)

        goals = traj["goals"] if goals is None else goals
        metrics = self.get_eval_values(traj, seed, goals)
        images = traj["observations"]["image"].squeeze() # (T, H, W, 3)

        num_rows = len(metrics.keys()) + 1

        fig, axs = plt.subplots(num_rows, 1, figsize=(8, 16))
        canvas = FigureCanvas(fig)
        plt.xlim(0, len(metrics["rewards"]))

        interval = images.shape[0] // 8
        interval = max(1, interval)
        sel_images = images[::interval]
        sel_images = np.split(sel_images, sel_images.shape[0], 0)
        sel_images = [a.squeeze() for a in sel_images]
        sel_images = np.concatenate(sel_images, axis=1) # (200, 8*200, 3)
        axs[0].imshow(sel_images)
        
        for i, (key, metric_val) in enumerate(metrics.items()):
            row = i + 1
            axs[row].plot(metric_val, linestyle='--', marker='o')
            axs[row].set_ylim([metric_val.min(), metric_val.max()])
            axs[row].set_ylabel(key)
        plt.tight_layout()
        canvas.draw()  # draw the canvas, cache the renderer
        out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return out_image



    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model arch
        encoder_def: nn.Module,
        shared_encoder: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        },
        # goals
        goals: Optional[Data] = None,
        early_goal_concat: bool = False,
        shared_goal_encoder: bool = True,
        **kwargs,
    ):
        # update algorithm config
        config = get_default_config(updates=kwargs)
        if config.language_conditioned:
            assert config.goal_conditioned, "Language conditioning requires goal conditioning"

        encoder_def = cls._create_encoder_def(
            encoder_def,
            use_proprio=False,
            enable_stacking=False,
            goal_conditioned=config.goal_conditioned,
            early_goal_concat=early_goal_concat,
            shared_goal_encoder=shared_goal_encoder,
            language_conditioned=config.language_conditioned,
        )

        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
                "critic": copy.deepcopy(encoder_def),
            }
            
        print("Encoder def:", encoder_def)

        # Define networks
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )
        critic_backbone = partial(MLP, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, config.critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")
        temperature_def = GeqLagrangeMultiplier(
            init_value=config.temperature_init,
            constraint_shape=(),
            name="temperature",
        )
        if config["cql_autotune_alpha"]:
            cql_alpha_lagrange_def = LeqLagrangeMultiplier(
                init_value=config.cql_alpha_lagrange_init,
                constraint_shape=(),
                name="cql_alpha_lagrange",
            )

        # model def
        networks = {
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }
        if config["cql_autotune_alpha"]:
            networks["cql_alpha_lagrange"] = cql_alpha_lagrange_def
        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**config.actor_optimizer_kwargs),
            "critic": make_optimizer(**config.critic_optimizer_kwargs),
            "temperature": make_optimizer(**config.temperature_optimizer_kwargs),
        }
        if config["cql_autotune_alpha"]:
            txs["cql_alpha_lagrange"] = make_optimizer(
                **config.cql_alpha_lagrange_otpimizer_kwargs
            )

        # init params
        rng, init_rng = jax.random.split(rng)
        extra_kwargs = {}
        if config["cql_autotune_alpha"]:
            extra_kwargs["cql_alpha_lagrange"] = []
        network_input = (
            (observations, goals) if config.goal_conditioned else observations
        )
        params = model_def.init(
            init_rng,
            actor=[network_input],
            critic=[network_input, actions],
            temperature=[],
            **extra_kwargs,
        )["params"]

        # create
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # config
        if config.target_entropy >= 0.0:
            config.target_entropy = -actions.shape[-1]
        config = flax.core.FrozenDict(config)

        return cls(state, config)


def get_default_config(updates=None):
    config = ConfigDict()
    config.discount = 0.98
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.soft_target_update_rate = 5e-3
    config.critic_ensemble_size = 2
    config.critic_subsample_size = None
    config.autotune_entropy = True
    config.temperature_init = 1.0
    config.actor_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 1e-4,
            "warmup_steps": 2000,
        }
    )
    config.critic_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
            "warmup_steps": 2000,
        }
    )
    config.temperature_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )

    config.cql_n_actions = 4
    config.cql_action_sample_method = "uniform"
    config.cql_max_target_backup = True
    config.cql_importance_sample = True
    config.cql_autotune_alpha = False
    config.cql_alpha_lagrange_init = 1.0
    config.cql_alpha_lagrange_otpimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )
    config.cql_target_action_gap = 1.0
    config.cql_temp = 1.0
    config.cql_alpha = 5.0
    config.cql_clip_diff_min = -np.inf
    config.cql_clip_diff_max = np.inf
    config.use_td_loss = True  # set this to False to essentially do BC

    # Cal-QL
    config.use_calql = False

    # Goal-conditioning
    config.goal_conditioned = False
    config.gc_kwargs = ConfigDict(
        {
            "negative_proportion": 0.0,
        }
    )

    config.early_goal_concat = False
    config.language_conditioned = False

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config
