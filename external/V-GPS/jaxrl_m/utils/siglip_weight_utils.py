# Loads pi0 (openpi, https://github.com/Physical-Intelligence/openpi, Apache-2.0) checkpoints; restore logic follows openpi.models.model.restore_params.
"""
Utility to load pretrained SigLIP weights from a Pi0 checkpoint and inject
them into the SARSA ensemble agent's parameter tree.

Pi0 checkpoint stores SigLIP params under params['PaliGemma']['img'].
The agent stores them under critic/encoder/encoder/siglip/ (per ensemble member).

Checkpoint loading is inlined from openpi.models.model.restore_params to avoid
pulling in openpi's full dependency tree (augmax, etc.).
"""
import pathlib

import jax
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from absl import logging
from flax import traverse_util


SIGLIP_PARAM_PREFIX = "modules_critic/encoder/encoder/siglip/"


def _restore_params(params_path: str) -> dict:
    """Load params from an orbax checkpoint as numpy arrays.

    Equivalent to openpi.models.model.restore_params(path, restore_type=np.ndarray)
    but without importing openpi (which drags in augmax etc.).
    """
    params_path = pathlib.Path(params_path).resolve()

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=None, restore_type=np.ndarray, dtype=None,
                    ),
                    item,
                ),
            ),
        )["params"]

    # Strip NNX "value" suffix if present (openpi saves via nnx.State)
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)


def load_siglip_params_from_pi0(checkpoint_path: str) -> dict:
    """Load SigLIP params from Pi0 checkpoint, filtering out head layer.

    Args:
        checkpoint_path: Path to Pi0 checkpoint params dir
            (e.g. /V-GPS/checkpoints/pi0_base/params)

    Returns:
        Flat dict of SigLIP param arrays (numpy), keys without head/* entries.
    """
    pi0_params = _restore_params(checkpoint_path)

    assert "PaliGemma" in pi0_params, (
        f"Expected 'PaliGemma' key in checkpoint, got: {list(pi0_params.keys())}"
    )
    assert "img" in pi0_params["PaliGemma"], (
        f"Expected 'img' key under PaliGemma, got: {list(pi0_params['PaliGemma'].keys())}"
    )

    siglip_params = pi0_params["PaliGemma"]["img"]
    flat = traverse_util.flatten_dict(siglip_params, sep="/")

    # Filter out head params (Pi0 projects to paligemma_width=2048; we use num_classes=None)
    filtered = {k: v for k, v in flat.items() if not k.startswith("head")}

    n_filtered = len(flat) - len(filtered)
    logging.info(
        f"Loaded {len(filtered)} SigLIP param arrays from Pi0 checkpoint "
        f"(filtered {n_filtered} head params)"
    )

    # Sanity check: embedding kernel should be (14, 14, 3, 1152) for So400m/14
    emb_key = "embedding/kernel"
    assert emb_key in filtered, f"Missing {emb_key} in SigLIP params"
    assert filtered[emb_key].shape == (14, 14, 3, 1152), (
        f"Unexpected embedding/kernel shape: {filtered[emb_key].shape}, "
        f"expected (14, 14, 3, 1152) for So400m/14"
    )

    return filtered


def inject_siglip_weights(ensemble_agents, siglip_flat: dict, ensemble_size: int):
    """Replace random SigLIP params in ensemble agent with pretrained weights.

    Args:
        ensemble_agents: Vmapped SARSA ensemble agent (params have leading ensemble dim)
        siglip_flat: Flat dict from load_siglip_params_from_pi0
        ensemble_size: Number of ensemble members

    Returns:
        Updated ensemble_agents with pretrained SigLIP weights.
    """
    agent_params = ensemble_agents.state.params
    flat_agent = traverse_util.flatten_dict(agent_params, sep="/")

    injected = 0
    mismatched = []

    for siglip_key, siglip_val in siglip_flat.items():
        agent_key = SIGLIP_PARAM_PREFIX + siglip_key

        if agent_key not in flat_agent:
            logging.warning(f"SigLIP key '{siglip_key}' not found in agent as '{agent_key}'")
            mismatched.append(siglip_key)
            continue

        # Broadcast: (shape) → (ensemble_size, shape)
        tiled = jnp.tile(
            jnp.array(siglip_val)[None, ...],
            [ensemble_size] + [1] * siglip_val.ndim,
        )
        assert flat_agent[agent_key].shape == tiled.shape, (
            f"Shape mismatch for {agent_key}: "
            f"agent={flat_agent[agent_key].shape}, pretrained={tiled.shape}"
        )
        flat_agent[agent_key] = tiled
        injected += 1

    assert injected > 0, (
        f"No SigLIP weights injected — param path mismatch. "
        f"Agent keys with 'siglip': {[k for k in flat_agent if 'siglip' in k][:5]}"
    )
    logging.info(f"Injected {injected}/{len(siglip_flat)} SigLIP param arrays into ensemble")

    if mismatched:
        logging.warning(f"Unmatched SigLIP keys ({len(mismatched)}): {mismatched[:5]}")

    new_params = traverse_util.unflatten_dict(flat_agent, sep="/")

    # Match original pytree type (FrozenDict vs dict) to avoid structure mismatch in jax.grad
    orig_type = type(agent_params)
    if hasattr(orig_type, "freeze") or "FrozenDict" in orig_type.__name__:
        import flax.core
        new_params = flax.core.freeze(new_params)

    # Update both params and target_params (target network starts from pretrained too)
    new_state = ensemble_agents.state.replace(
        params=new_params,
        target_params=new_params,
    )
    return ensemble_agents.replace(state=new_state)


def load_and_inject_siglip_weights(ensemble_agents, pi0_checkpoint_path: str, ensemble_size: int):
    """End-to-end: load Pi0 checkpoint → extract SigLIP → inject into ensemble.

    Args:
        ensemble_agents: Vmapped SARSA ensemble agent
        pi0_checkpoint_path: Path to Pi0 checkpoint params dir
        ensemble_size: Number of ensemble members

    Returns:
        Updated ensemble_agents with pretrained SigLIP backbone weights.
    """
    siglip_flat = load_siglip_params_from_pi0(pi0_checkpoint_path)
    ensemble_agents = inject_siglip_weights(ensemble_agents, siglip_flat, ensemble_size)

    # Log param norms for verification
    flat_agent = traverse_util.flatten_dict(ensemble_agents.state.params, sep="/")
    # Take first ensemble member for norm logging
    backbone_norms = []
    proj_norms = []
    mlp_norms = []
    for k, v in flat_agent.items():
        v0 = v[0]  # first ensemble member
        norm = float(jnp.linalg.norm(v0))
        if "siglip/" in k:
            backbone_norms.append(norm)
        elif "vision_language_proj" in k:
            proj_norms.append(norm)
        elif "critic_ensemble" in k:
            mlp_norms.append(norm)

    if backbone_norms:
        logging.info(f"SigLIP backbone mean param norm: {sum(backbone_norms)/len(backbone_norms):.2f}")
    if proj_norms:
        logging.info(f"Projection layer mean param norm: {sum(proj_norms)/len(proj_norms):.4f}")
    if mlp_norms:
        logging.info(f"Critic MLP mean param norm: {sum(mlp_norms)/len(mlp_norms):.4f}")

    return ensemble_agents
