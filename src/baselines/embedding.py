# ledoit_wolf re-implements sklearn.covariance.ledoit_wolf_shrinkage (BSD-3, https://github.com/scikit-learn/scikit-learn); Ledoit & Wolf (2004).
"""Embedding-distance baselines for "is the Q-ensemble just an OOD detector?".

Two families, both training-free w.r.t. the detection task:
  novelty    distance of a frame's feature to a reference set of in-distribution features
             (Mahalanobis with Ledoit-Wolf shrinkage, or cosine k-th nearest neighbour),
             fitted per group (source dataset) and scored as the minimum over groups.
             Features in, scores out: works on CLIP features and on the critic's own encoder.
  alignment  -cos(image embedding, instruction embedding) in CLIP space.

Scores follow the repo convention: higher = more anomalous / infeasible. Frames are the
(first, last) pair every perturbation dump carries; "both" is the mean of the two frame scores.
Pure numpy except `clip_embed` (lazy torch/transformers import). Drivers:
src/scripts/baselines/embedding_baselines.py (CLIP), say_no.perturbation.novelty (critic encoder).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from say_no.cp.metrics import _auroc_rows, _average_ranks, auroc_neg_pos, paired_auroc_ci
from say_no.perturbation.scores import FRAMES


def ledoit_wolf(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrunk covariance of rows of x (same estimator as sklearn.covariance.LedoitWolf)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"need (n>=2, d) samples, got {x.shape}")
    n, p = x.shape
    xc = x - x.mean(axis=0)
    emp = xc.T @ xc / n
    mu = np.trace(emp) / p
    x2 = xc ** 2
    beta_ = np.sum(x2.T @ x2) / n
    delta_ = np.sum(emp ** 2)
    beta = (beta_ - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * np.trace(emp) + p * mu ** 2) / p
    beta = min(beta, delta)
    shrink = 0.0 if beta == 0 else beta / delta
    return (1.0 - shrink) * emp + shrink * mu * np.eye(p), float(shrink)


class FeatureNovelty:
    """Distance of x to a reference feature set; per-group fit, min over groups at score time."""

    METHODS = ("mahalanobis", "knn")

    def __init__(self, method: str, k: int = 5):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.method, self.k = method, k
        self.fitted_: dict | None = None

    def fit(self, ref: np.ndarray, groups: np.ndarray) -> "FeatureNovelty":
        ref = np.asarray(ref, dtype=np.float64)
        groups = np.asarray(groups)
        if ref.ndim != 2 or groups.shape != (ref.shape[0],):
            raise ValueError(f"ref (n, d) and groups (n,) expected, got {ref.shape}, {groups.shape}")
        if not np.all(np.isfinite(ref)):
            raise ValueError("non-finite reference features")
        self.fitted_ = {}
        for g in np.unique(groups):
            r = ref[groups == g]
            if self.method == "mahalanobis":
                cov, shrink = ledoit_wolf(r)
                self.fitted_[g] = {"mean": r.mean(axis=0), "prec": np.linalg.inv(cov), "shrinkage": shrink, "n": len(r)}
            else:
                if len(r) < self.k:
                    raise ValueError(f"group {g!r} has {len(r)} < k={self.k} reference samples")
                self.fitted_[g] = {"unit": _unit(r), "n": len(r)}
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.fitted_ is None:
            raise RuntimeError("fit before score")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x must be (n, d), got {x.shape}")
        per_group = []
        for st in self.fitted_.values():
            if self.method == "mahalanobis":
                d = x - st["mean"]
                per_group.append(np.sqrt(np.einsum("nd,de,ne->n", d, st["prec"], d)))
            else:
                dist = 1.0 - _unit(x) @ st["unit"].T
                per_group.append(np.sort(dist, axis=1)[:, self.k - 1])
        return np.min(per_group, axis=0)

    def summary(self) -> dict:
        if self.fitted_ is None:
            raise RuntimeError("fit before summary")
        return {str(g): {k: v for k, v in st.items() if k in ("shrinkage", "n")} for g, st in self.fitted_.items()}


def _unit(x: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(nrm == 0):
        raise ValueError("zero-norm feature vector")
    return x / nrm


def alignment_score(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    """Row-wise -cos(image, text): higher = instruction fits the frame worse."""
    img_emb, txt_emb = np.asarray(img_emb, np.float64), np.asarray(txt_emb, np.float64)
    if img_emb.shape != txt_emb.shape:
        raise ValueError(f"paired embeddings expected, got {img_emb.shape} vs {txt_emb.shape}")
    return -np.sum(_unit(img_emb) * _unit(txt_emb), axis=-1)


def frame_novelty(model_factory, ref: np.ndarray, ref_groups: np.ndarray, feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Novelty per frame for each condition.

    ref: (n_ref, 2, D) or (n_ref, M, 2, D); feats[cond]: (n, 2, D) or (n, M, 2, D) with the same M.
    With a member axis (the critic's own encoder, one per ensemble member) each member gets its
    own reference fit and the per-frame score is the member mean. Returns cond -> (n, 2).
    ref_groups: (n_ref,), the source dataset of each reference trajectory (both frames inherit it).
    """
    ref = np.asarray(ref)
    if ref.ndim == 3:
        ref, feats = ref[:, None], {c: np.asarray(v)[:, None] for c, v in feats.items()}
    if ref.ndim != 4 or ref.shape[2] != 2:
        raise ValueError(f"ref must be (n, 2, D) or (n, M, 2, D), got {ref.shape}")
    n_ref, m = ref.shape[:2]
    groups = np.repeat(np.asarray(ref_groups), 2)
    if groups.shape != (2 * n_ref,):
        raise ValueError(f"ref_groups must be ({n_ref},), got {np.shape(ref_groups)}")
    out = {c: np.zeros(v.shape[:1] + (2,)) for c, v in feats.items()}
    for c, v in feats.items():
        if v.ndim != 4 or v.shape[1:] != ref.shape[1:]:
            raise ValueError(f"{c}: feature shape {v.shape} does not match ref {ref.shape}")
    for j in range(m):
        model = model_factory().fit(ref[:, j].reshape(2 * n_ref, -1), groups)
        for c, v in feats.items():
            out[c] += model.score(v[:, j].reshape(-1, v.shape[-1])).reshape(-1, 2) / m
    return out


def by_frame(s: np.ndarray) -> dict[str, np.ndarray]:
    """(n, 2) per-frame scores -> {first, last, both (= mean of the two frame scores)}."""
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError(f"expected (n, 2) per-frame scores, got {s.shape}")
    return dict(zip(FRAMES, (s[:, 0], s[:, 1], s.mean(axis=1))))


def score_key(name: str, frame: str) -> str:
    """'both' keeps the bare name so it lines up with the ensemble's frame-averaged row."""
    return name if frame == "both" else f"{name} ({frame} frame)"


def auroc_rows(name: str, neg: np.ndarray, pos: np.ndarray, n_boot: int = 2000) -> dict:
    """AUROC (+ paired CI) of pos vs neg per frame, keys in the perturbation_sensitivity.json schema."""
    rows = {}
    for frame, (a, b) in zip(FRAMES, zip(by_frame(neg).values(), by_frame(pos).values())):
        key = score_key(name, frame)
        rows[f"auroc {key}"] = auroc_neg_pos(a, b)
        rows[f"auroc {key} ci95"] = list(paired_auroc_ci(a, b, n_boot=n_boot))
    return rows


def paired_auroc_diff_ci(neg_a, pos_a, neg_b, pos_b, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """AUROC(a) - AUROC(b) and its percentile CI, resampling trajectories jointly for both scores."""
    arrs = [np.asarray(x, np.float64) for x in (neg_a, pos_a, neg_b, pos_b)]
    if len({x.shape for x in arrs}) != 1 or arrs[0].ndim != 1:
        raise ValueError(f"four equal-length 1-D arrays expected, got {[x.shape for x in arrs]}")
    na, pa, nb, pb = arrs
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, na.size, (n_boot, na.size))
    d = _auroc_rows(na[idx], pa[idx]) - _auroc_rows(nb[idx], pb[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return auroc_neg_pos(na, pa) - auroc_neg_pos(nb, pb), float(lo), float(hi)


def spearman(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.shape != b.shape or a.size < 3:
        raise ValueError(f"need equal-length >= 3 vectors, got {a.shape} vs {b.shape}")
    return float(np.corrcoef(_average_ranks(a), _average_ranks(b))[0, 1])


def spearman_ci(a, b, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, (n_boot, a.size))
    vals = np.array([spearman(a[i], b[i]) for i in idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return spearman(a, b), float(lo), float(hi)


def rows_for_dump(trajs: list[dict], dataset: np.ndarray, trajectory_ids: np.ndarray) -> list[dict]:
    """pkl trajectories in a dump's row order, matched on (dataset, trajectory_id) (unique, asserted)."""
    index = {(t["dataset"], int(t["trajectory_id"])): t for t in trajs}
    if len(index) != len(trajs):
        raise ValueError("(dataset, trajectory_id) is not unique in the pkl")
    keys = list(zip([str(d) for d in dataset], [int(i) for i in trajectory_ids]))
    missing = [k for k in keys if k not in index]
    if missing:
        raise ValueError(f"{len(missing)} dump rows not in the pkl, e.g. {missing[:3]}")
    return [index[k] for k in keys]


def image_key(img: np.ndarray) -> str:
    img = np.ascontiguousarray(img)
    return hashlib.sha1(str(img.shape).encode() + img.tobytes()).hexdigest()


def clip_embed(images: list[np.ndarray], texts: list[str], model_id: str, revision: str,
               batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalised CLIP image (n, D) and text (m, D) embeddings, CPU, eval mode."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id, revision=revision).eval()
    proc = CLIPProcessor.from_pretrained(model_id, revision=revision)

    def _tensor(out):  # transformers >= 5 may wrap projected features in a model-output object
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    img_out, txt_out = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = proc(images=[np.asarray(x) for x in images[i:i + batch_size]], return_tensors="pt")
            img_out.append(_tensor(model.get_image_features(pixel_values=batch["pixel_values"])).numpy())
        for i in range(0, len(texts), batch_size):
            batch = proc(text=list(texts[i:i + batch_size]), return_tensors="pt", padding=True, truncation=True)
            txt_out.append(_tensor(model.get_text_features(input_ids=batch["input_ids"],
                                                           attention_mask=batch["attention_mask"])).numpy())
    dim = model.config.projection_dim
    img = np.concatenate(img_out) if img_out else np.zeros((0, dim), np.float32)
    txt = np.concatenate(txt_out) if txt_out else np.zeros((0, dim), np.float32)
    return _unit(img).astype(np.float32), _unit(txt).astype(np.float32)


def cached_clip_embed(images: list[np.ndarray], texts: list[str], cache: Path, model_id: str,
                      revision: str) -> tuple[np.ndarray, np.ndarray]:
    """clip_embed with an .npz cache keyed by image sha1 / exact text; embeds only what is missing."""
    cache = Path(cache)
    img_store, txt_store = {}, {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["model_id"]) != model_id or str(z["revision"]) != revision:
            raise ValueError(f"cache {cache} was built with {z['model_id']}@{z['revision']}, not {model_id}@{revision}")
        img_store = dict(zip(z["img_keys"].tolist(), z["img_emb"]))
        txt_store = dict(zip(z["txt_keys"].tolist(), z["txt_emb"]))
    keys = [image_key(x) for x in images]
    new_img = {k: x for k, x in zip(keys, images) if k not in img_store}
    new_txt = sorted({t for t in texts if t not in txt_store})
    if new_img or new_txt:
        ie, te = clip_embed(list(new_img.values()), new_txt, model_id, revision)
        img_store.update(zip(new_img, ie))
        txt_store.update(zip(new_txt, te))
        cache.parent.mkdir(parents=True, exist_ok=True)
        ik, tk = list(img_store), list(txt_store)
        np.savez(cache, model_id=model_id, revision=revision,
                 img_keys=np.array(ik), img_emb=np.stack([img_store[k] for k in ik]),
                 txt_keys=np.array(tk, dtype=str), txt_emb=np.stack([txt_store[k] for k in tk]))
    dim = next(iter(img_store.values())).shape[-1]
    return (np.stack([img_store[k] for k in keys]) if keys else np.zeros((0, dim), np.float32),
            np.stack([txt_store[t] for t in texts]) if texts else np.zeros((0, dim), np.float32))


def read_features(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """feat/<cond> (n, M, 2, D) of an eval_perturbation_sensitivity.py --dump_features HDF5,
    plus its dataset and trajectory_ids."""
    import h5py

    with h5py.File(path) as f:
        if "feat" not in f:
            raise ValueError(f"{path}: no feat/ group (dump with --dump_features)")
        feats = {c: f[f"feat/{c}"][()].astype(np.float64) for c in f["feat"].keys()}
        dataset = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in f["dataset"][()]])
        ids = f["trajectory_ids"][()]
    if "orig" not in feats:
        raise ValueError(f"{path}: feat/orig missing")
    for c, v in feats.items():
        if v.ndim != 4 or v.shape[0] != len(ids) or v.shape[2] != 2 or not np.all(np.isfinite(v)):
            raise ValueError(f"{path}: feat/{c} shape {v.shape} or values invalid, expected ({len(ids)}, M, 2, D)")
    return feats, dataset, ids


def markdown_table(results: dict) -> str:
    """One table per label: every score (ensemble std_Q first) with AUROC [CI] and its paired
    difference to std_Q [CI]; Spearman rows listed below."""
    lines = []
    for lab, res in results.items():
        lines += [f"\n## {lab}  (n={res['meta'].get('n_trajectories', '?')})\n",
                  "| condition | score | AUROC [95% CI] | AUROC(std_Q) - AUROC(score) [95% CI] |", "|---|---|---|---|"]
        for c, row in res["conditions"].items():
            names = [k[len("auroc "):] for k in row if k.startswith("auroc ") and not k.endswith("ci95")
                     and not k.startswith("auroc diff ")]
            paired_with_ensemble = "auroc std_Q" in row
            for nm in names:
                lo, hi = row[f"auroc {nm} ci95"]
                dcell = ""
                if paired_with_ensemble and not nm.startswith("std_Q"):
                    d, (dlo, dhi) = row[f"auroc diff std_Q - {nm}"], row[f"auroc diff std_Q - {nm} ci95"]
                    dcell = f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
                lines.append(f"| {c} | {nm} | {row[f'auroc {nm}']:.3f} [{lo:.3f}, {hi:.3f}] | {dcell} |")
            for k, v in row.items():
                if k.startswith("spearman ") and not k.endswith("ci95"):
                    lo, hi = row[f"{k} ci95"]
                    lines.append(f"| {c} | {k} | {v:+.3f} [{lo:+.3f}, {hi:+.3f}] | |")
        if "note" in res["meta"]:
            lines.append(f"\nnote: {res['meta']['note']}")
    meta = next(iter(results.values()))["meta"]
    head = (f"# Embedding-distance baselines (CLIP), rendered {meta['rendered']}\n\n"
            f"CLIP: {meta['clip_model']}@{meta['clip_revision']} · kNN k={meta['k']} · bootstrap {meta['n_boot']} "
            f"(paired, trajectories) · reference: {meta['reference']} · 'both': {meta['both']}\n")
    return head + "\n".join(lines) + "\n"
