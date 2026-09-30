"""Input perturbations and model scorers for `experiments/eval_perturbation_sensitivity.py`.

Language: near swaps (the most similar training instruction that is not a paraphrase).
Image: JPEG re-encoding matched to the inpainting's pixel change, and masked composites that
put the inpainted pixels only inside (local) or only outside (global) the edited region, so the
inpainting effect can be split into "the object changed" vs. "the whole image got re-encoded".
Scorers: one callable per model family returning an array (members, heads, frames) where
higher = the reading the model reports (Q for the ensemble, P(infeasible) for classifiers).
"""
from __future__ import annotations

import re
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# --------------------------------------------------------------------------- language

# Words that do not change what the instruction asks for. Verbs are canonicalised rather than
# dropped so that "open drawer" vs. "close drawer" stays a valid near swap.
FUNCTION_WORDS = frozenset(
    "the a an to of on in into onto from and it its at with for this that then there which is are be up".split())
VERB_CANON = {**{w: "put" for w in "put puts putting place placed places placing move moved moves moving set".split()},
              **{w: "pick" for w in "pick picked picks picking grab grabbed take took taking lift lifted".split()}}
NEAR_MAX_COSINE = 0.95
STEM_CHARS = 6


def _stem(w: str) -> str:
    """Crude stem so inflections ("close"/"closed") and derivations ("rectangle"/"rectangular")
    compare equal: drop one inflectional suffix and a final e, keep the first STEM_CHARS letters."""
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[:-len(suf)]
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w[:STEM_CHARS]


def content_words(s: str) -> frozenset[str]:
    """Stems of the words that carry the task: function words dropped, paraphrase verbs merged."""
    return frozenset(_stem(VERB_CANON.get(w, w)) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in FUNCTION_WORDS)


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-norm text embedding")
    return x / n


def is_substitution(own: frozenset[str], cand: frozenset[str]) -> bool:
    """Each side has a content word the other lacks. Pure additions/deletions ("put broccoli in
    pot" vs. "... in pot or pan") usually describe the same task and are rejected."""
    return bool(own - cand) and bool(cand - own)


def near_swaps(own: list[str], pool: list[str], text_processor, max_cosine: float) -> tuple[list[str], np.ndarray]:
    """For each own instruction the pool instruction with the highest MUSE cosine among those that
    (a) substitute at least one content word (is_substitution) and (b) have cosine <= max_cosine.
    Returns strings, cosines."""
    if not pool:
        raise ValueError("near-swap pool is empty")
    uniq = sorted(set(own))
    e_own = unit_rows(text_processor.encode(uniq))
    e_pool = unit_rows(np.concatenate([text_processor.encode(pool[i:i + 512]) for i in range(0, len(pool), 512)]))
    cos = e_own @ e_pool.T
    pool_words = [content_words(p) for p in pool]
    choice = {}
    for u, s in enumerate(uniq):
        w = content_words(s)
        valid = np.array([is_substitution(w, pw) for pw in pool_words]) & (cos[u] <= max_cosine)
        if not valid.any():
            raise ValueError(f"no valid near swap for {s!r}")
        j = int(np.argmax(np.where(valid, cos[u], -np.inf)))
        choice[s] = (pool[j], float(cos[u, j]))
    return [choice[s][0] for s in own], np.array([choice[s][1] for s in own], np.float32)


# --------------------------------------------------------------------------- image

# defined jax-free in image_perturbations.py (also imported by the host analysis); re-exported here
from experiments.utils.image_perturbations import (  # noqa: E402,F401
    IMAGE_CONDITION, change_mask, choose_jpeg_quality, image_conditions, is_image_condition, jpeg, mean_abs_delta, median_abs_delta)


def contact_sheet(trajs: list[dict], images: dict, masks: dict, radius: int, path: str, n: int = 12) -> None:
    """Rows: trajectories (first frame); columns: original, inpainted, every image condition, mask overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["original", "inpainted"] + list(images) + [f"mask r={radius} on original"]
    idx = np.linspace(0, len(trajs) - 1, min(n, len(trajs))).astype(int)
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(1.6 * len(cols), 1.7 * len(idx)), squeeze=False)
    for row, i in enumerate(idx):
        t = trajs[i]
        panels = [t["first_image"], t["first_image_inpainted"]] + [images[c][i, 0] for c in images]
        overlay = t["first_image"].astype(np.float32).copy()
        overlay[masks[radius][i, 0]] = 0.5 * overlay[masks[radius][i, 0]] + 0.5 * np.array([255, 0, 0])
        panels.append(overlay.astype(np.uint8))
        for col, img in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(cols[col], fontsize=6)
        axes[row][0].set_ylabel(f"{t['dataset'][:8]} {t['trajectory_id']}\n{t['language'][:30]}", fontsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- scorers

@partial(jax.vmap, in_axes=(0, None, 0))
def _members_q_and_features(agent, batch, rng):
    obs = agent._include_goals_in_obs(batch, "observations")
    return agent.forward_critic_with_embeddings(obs, batch["actions"], rng=rng, train=False)


@partial(jax.vmap, in_axes=(0, None, 0))
def _members_q(agent, batch, rng):
    obs = agent._include_goals_in_obs(batch, "observations")
    return agent.forward_critic(obs, batch["actions"], rng=rng, train=False)


def ensemble_scorer(agents, ensemble_size: int, text_processor, with_features: bool):
    """(images (F,H,W,3), actions (F,7), language [F]) -> (q (members, 2, F), features (members, F, D) | None)."""
    rngs = jax.random.split(jax.random.PRNGKey(0), ensemble_size)
    fwd = jax.jit(_members_q_and_features if with_features else _members_q)

    def score(images, actions, language):
        batch = {"actions": actions.astype(np.float32), "observations": {"image": images},
                 "goals": {"language": text_processor.encode(language)}}
        out = fwd(agents, batch, rngs)
        q, feat = (out if with_features else (out, None))
        q = np.asarray(q)
        if q.shape != (ensemble_size, 2, images.shape[0]):
            raise ValueError(f"unexpected critic output shape {q.shape}")
        if feat is not None:
            feat = np.asarray(feat)
            if feat.shape[:2] != (ensemble_size, images.shape[0]) or feat.ndim != 3:
                raise ValueError(f"unexpected encoder feature shape {feat.shape}")
        return q, feat
    return score


def classifier_scorer(agent, text_processor):
    """ResNet+MUSE binary classifier. Returns P(infeasible) as (1, 1, F), no features."""
    fwd = jax.jit(lambda a, obs_goals: a.forward_classifier(obs_goals, rng=None, train=False))

    def score(images, actions, language):
        logits = np.asarray(fwd(agent, ({"image": images}, {"language": text_processor.encode(language)})))
        if logits.shape != (images.shape[0],):
            raise ValueError(f"unexpected classifier logits shape {logits.shape}")
        p_feasible = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        return (1.0 - p_feasible).astype(np.float32)[None, None, :], None
    return score


def octo_scorer(model):
    """Octo with a feasibility head, each frame scored on its own (window 1, as in training with
    data_config.py:baseline_classifier). Returns P(infeasible) as (1, 1, F), no features."""
    def _logits(params, observations, tasks):
        bound = model.module.bind({"params": params})
        emb = bound.octo_transformer(observations, tasks, observations["timestep_pad_mask"], train=False)
        return bound.heads["feasibility"](emb, train=False)
    fwd = jax.jit(_logits)

    def score(images, actions, language):
        f = images.shape[0]
        obs = {"image_primary": jnp.asarray(images)[:, None],
               "timestep_pad_mask": jnp.ones((f, 1), bool),
               "pad_mask_dict": {"image_primary": jnp.ones((f, 1), bool)}}
        logits = np.asarray(fwd(model.params, obs, model.create_tasks(texts=list(language))))
        if logits.shape != (f, 1, 1):
            raise ValueError(f"unexpected Octo feasibility logits shape {logits.shape}")
        p_feasible = 1.0 / (1.0 + np.exp(-logits.reshape(f).astype(np.float64)))
        return (1.0 - p_feasible).astype(np.float32)[None, None, :], None
    return score
