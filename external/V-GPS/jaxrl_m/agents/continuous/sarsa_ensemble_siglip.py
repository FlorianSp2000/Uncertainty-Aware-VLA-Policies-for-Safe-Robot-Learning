# SARSA critic agent built on the CQL agent of V-GPS/jaxrl_m (https://github.com/nakamotoo/V-GPS).
"""SARSAEnsembleAgent subclass with chunked eval for large vision backbones (SigLIP)."""
import jax
import jax.numpy as jnp
from jaxrl_m.agents.continuous.sarsa_ensemble import SARSAEnsembleAgent

EVAL_CHUNK_SIZE = 32  # max traj frames per forward pass during eval


class SigLIPSARSAEnsembleAgent(SARSAEnsembleAgent):

    def _chunked_forward(self, forward_fn, obs, actions, seed, chunk_size=EVAL_CHUNK_SIZE):
        """Run forward_fn in chunks along traj dim to avoid OOM on long trajectories.

        forward_fn must accept (obs, actions, seed) — no train kwarg (eval only).
        """
        traj_len = actions.shape[0]
        if traj_len <= chunk_size:
            return forward_fn(obs, actions, seed)

        def _chunk_slice(o, start, end):
            if isinstance(o, tuple):
                return tuple(jax.tree.map(lambda x: x[start:end], part) for part in o)
            return jax.tree.map(lambda x: x[start:end], o)

        chunks = []
        for start in range(0, traj_len, chunk_size):
            end = min(start + chunk_size, traj_len)
            chunk_q = forward_fn(
                _chunk_slice(obs, start, end),
                actions[start:end],
                seed,
            )
            chunks.append(chunk_q)

        # concat along traj dim (axis 1 for 2D q, axis -2 for 3D ood_q)
        return jnp.concatenate(chunks, axis=1 if actions.ndim == 2 else -2)

    def _eval_critic(self, obs, actions, seed):
        """Eval-only wrapper for forward_critic (matches forward_target_critic signature)."""
        return self.forward_critic(obs, actions, seed, train=False)

    def get_eval_values(self, traj, seed, goals, **kwargs):
        """Chunked version — identical logic to base class but uses _chunked_forward."""
        obs = (traj["observations"], goals) if self.config["goal_conditioned"] else traj["observations"]

        q_raw = self._chunked_forward(self._eval_critic, obs, traj["actions"], seed)
        q_values = jnp.min(q_raw, axis=0)

        target_q_raw = self._chunked_forward(self.forward_target_critic, obs, traj["actions"], seed)
        target_q_values = jnp.min(target_q_raw, axis=0)

        rng, ood_rng = jax.random.split(seed) if seed is not None else (jax.random.PRNGKey(42), jax.random.PRNGKey(43))
        batch_like = {"rewards": traj["rewards"], "actions": traj["actions"]}
        ood_actions = self._sample_ood_actions(batch_like, ood_rng, num_ood_actions=self.config.get("num_ood_actions", 10))

        ood_q = self._chunked_forward(self._eval_critic, obs, ood_actions, rng)
        ood_q_values = jnp.mean(jnp.min(ood_q, axis=0), axis=-1)

        ood_target_q = self._chunked_forward(self.forward_target_critic, obs, ood_actions, rng)
        ood_target_q_values = jnp.mean(jnp.min(ood_target_q, axis=0), axis=-1)

        monte_carlo_returns = self.compute_monte_carlo_returns(traj["rewards"], traj["masks"])

        return {
            "q": q_values,
            "target_q": target_q_values,
            "q_ood": ood_q_values,
            "target_q_ood": ood_target_q_values,
            "q_monte_carlo": monte_carlo_returns,
            "rewards": traj["rewards"],
            "masks": traj["masks"],
        }
