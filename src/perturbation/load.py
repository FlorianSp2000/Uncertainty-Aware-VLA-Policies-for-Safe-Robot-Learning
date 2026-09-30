"""Readers for the Bridge/Fractal perturbation dumps.

Canonical input: HDF5 of `eval_perturbation_sensitivity.py`, one per (checkpoint, pkl), with
`trajectory_ids`, `dataset`, `language`, `swap_language/<condition>` and either
  q/<condition>  (n, members, heads, frames=2)  Q-ensemble -> ModelDump kind "ensemble"
  p/<condition>  (n, 1, 1, frames=2)            classifier / Octo, P(infeasible) -> kind "detector"
Host-side detectors (CLIP scores, encoder-feature novelty) are kind "detector" too: one score per
(trajectory, frame), higher = more infeasible, `readout` names it. Several files for the same
model are merged by label; they must describe the same trajectories in the same order.

Legacy input: the Sept-2025 `inference_results_*.hdf5` of `evaluate_inpainting_uncertainty.py`
(scalars per trajectory, spread with ddof=0), kept for the superseded 09-28 morning outputs.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

N_FRAMES = 2  # (first, last)


@dataclass
class ModelDump:
    label: str
    kind: str  # "ensemble" | "detector"
    paths: list[str]
    meta: list[dict]
    trajectory_ids: np.ndarray
    dataset: np.ndarray
    language: np.ndarray
    q: dict[str, np.ndarray] = field(default_factory=dict)      # ensemble: (n, M, H, 2)
    score: dict[str, np.ndarray] = field(default_factory=dict)  # detector: (n, 2) per frame, or (n, 1) = both only
    readout: str = ""                                           # detector: name of its score
    swap_language: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def conditions(self) -> list[str]:
        return list(self.q if self.kind == "ensemble" else self.score)

    @property
    def frames(self) -> tuple[str, ...]:
        if self.kind == "ensemble" or next(iter(self.score.values())).shape[1] == N_FRAMES:
            return ("first", "last", "both")
        return ("both",)

    @property
    def n_members(self) -> int:
        assert self.kind == "ensemble"
        return self.q["orig"].shape[1]


def restrict(m: ModelDump, mask: np.ndarray) -> ModelDump:
    """The rows of `m` where `mask` is True (e.g. one source dataset)."""
    mask = np.asarray(mask, bool)
    if mask.shape != m.trajectory_ids.shape or not mask.any():
        raise ValueError(f"{m.label}: mask shape {mask.shape} or empty")
    sub = lambda d: {c: v[mask] for c, v in d.items()}
    return ModelDump(label=m.label, kind=m.kind, paths=m.paths, meta=m.meta, trajectory_ids=m.trajectory_ids[mask],
                     dataset=m.dataset[mask], language=m.language[mask], q=sub(m.q), score=sub(m.score),
                     readout=m.readout, swap_language=sub(m.swap_language))


def _str_array(ds) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[()]])


def _read_one(path: Path) -> dict:
    with h5py.File(path) as f:
        if ("q" in f) == ("p" in f):
            raise ValueError(f"{path}: need exactly one of q/ (ensemble) or p/ (classifier); keys {list(f.keys())}")
        kind, group = ("ensemble", "q") if "q" in f else ("detector", "p")
        vals = {c: f[f"{group}/{c}"][()].astype(np.float64) for c in f[group].keys()}
        if kind == "detector":
            for c, v in vals.items():
                if v.ndim != 4 or v.shape[1:3] != (1, 1) or v.shape[3] != N_FRAMES:
                    raise ValueError(f"{path}: p/{c} shape {v.shape}, expected (n, 1, 1, {N_FRAMES})")
                if v.min() < 0 or v.max() > 1:
                    raise ValueError(f"{path}: p/{c} outside [0, 1]")
            vals = {c: v[:, 0, 0, :] for c, v in vals.items()}
        out = {
            "kind": kind, "vals": vals,
            "meta": json.loads(f.attrs["metadata"]),
            "trajectory_ids": f["trajectory_ids"][()],
            "dataset": _str_array(f["dataset"]),
            "language": _str_array(f["language"]),
            "swap_language": ({c: _str_array(f[f"swap_language/{c}"]) for c in f["swap_language"].keys()}
                              if "swap_language" in f else {}),
        }
    n = len(out["trajectory_ids"])
    for c, v in vals.items():
        if kind == "ensemble" and (v.ndim != 4 or v.shape[0] != n or v.shape[3] != N_FRAMES):
            raise ValueError(f"{path}: q/{c} shape {v.shape}, expected ({n}, M, H, {N_FRAMES})")
        if kind == "detector" and v.shape != (n, N_FRAMES):
            raise ValueError(f"{path}: p/{c} shape {v.shape}, expected ({n}, {N_FRAMES})")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{path}: non-finite values in {c}")
    return out


def load_models(items: list[tuple[str, Path]]) -> dict[str, ModelDump]:
    """(label, path) pairs -> one ModelDump per label, conditions merged across files."""
    models: dict[str, ModelDump] = {}
    for label, path in items:
        r = _read_one(path)
        if label not in models:
            models[label] = ModelDump(label=label, kind=r["kind"], paths=[str(path)], meta=[r["meta"]],
                                      trajectory_ids=r["trajectory_ids"], dataset=r["dataset"],
                                      language=r["language"],
                                      readout="p_infeasible" if r["kind"] == "detector" else "")
        m = models[label]
        if m.paths[-1] != str(path):
            m.paths.append(str(path)); m.meta.append(r["meta"])
        if r["kind"] != m.kind:
            raise ValueError(f"{label}: mixes {m.kind} and {r['kind']} dumps")
        if not (np.array_equal(r["trajectory_ids"], m.trajectory_ids) and np.array_equal(r["dataset"], m.dataset)):
            raise ValueError(f"{label}: {path} describes different trajectories than {m.paths[0]}")
        target = m.q if m.kind == "ensemble" else m.score
        for c, v in r["vals"].items():
            if c in target:
                if c != "orig":
                    raise ValueError(f"{label}: condition {c} appears in two files")
                if not np.allclose(target[c], v, atol=1e-3):
                    raise ValueError(f"{label}: 'orig' differs between {m.paths[0]} and {path}")
                continue
            target[c] = v
        m.swap_language.update(r["swap_language"])
    for m in models.values():
        if "orig" not in m.conditions:
            raise ValueError(f"{m.label}: no 'orig' condition")
    return models


def _legacy(path: Path) -> tuple[dict, dict]:
    with h5py.File(path) as f:
        meta = json.loads(f.attrs["metadata"])
        x = {k: f[k][()] for k in f.keys() if f[k].ndim > 0 and f[k].shape[0] > 0}
    return x, meta


def legacy_originals(path: Path) -> dict[str, np.ndarray]:
    """Per-trajectory scores of the ORIGINAL frames of a legacy dump, higher = more infeasible.

    Ensemble: spread rescaled from ddof=0 to the canonical ddof=1, and -mean Q (both frames).
    Classifier: P(infeasible) = 1 - P(feasible)."""
    x, meta = _legacy(path)
    if meta.get("model_type", "ensemble") == "classifier":
        return {"p_infeasible": 1.0 - x["original_probs"].astype(np.float64)}
    m = int(meta["ensemble_size"])
    return {"spread": x["original_uncertainties"].astype(np.float64) * np.sqrt(m / (m - 1)),
            "neg_mean_q": -x["original_q_means"].astype(np.float64)}


def reviewed_rows(pkl_path: Path) -> tuple[list[dict], np.ndarray]:
    """Trajectories of an inpainting pkl that carry both inpainted frames, and the mask of
    those that passed the manual review (not in `failed_ids`) — the dumps' row order."""
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    trajs = [t for t in data["trajectories"]
             if t.get("first_image_inpainted") is not None and t.get("last_image_inpainted") is not None]
    failed = {(f["dataset"], f["trajectory_id"]) for f in data["failed_ids"]}
    keep = np.array([(t["dataset"], t["trajectory_id"]) not in failed for t in trajs])
    return trajs, keep


def legacy_classifier_inpainting(path: Path, pkl_path: Path, label: str) -> ModelDump:
    """Legacy classifier inpainting dump restricted to the reviewed pairs, as a ModelDump with
    conditions orig / inpaint."""
    x, meta = _legacy(path)
    if meta.get("model_type") != "classifier":
        raise ValueError(f"{path}: not a classifier dump ({meta.get('model_type')})")
    trajs, keep = reviewed_rows(pkl_path)
    if not np.array_equal(x["trajectory_ids"], np.array([t["trajectory_id"] for t in trajs])):
        raise ValueError(f"{path}: rows do not line up with {pkl_path}")
    kept = [t for t, k in zip(trajs, keep) if k]
    return ModelDump(label=label, kind="detector", paths=[str(path)], meta=[meta],
                     trajectory_ids=x["trajectory_ids"][keep],
                     dataset=np.array([t["dataset"] for t in kept]),
                     language=np.array([t["language"] for t in kept]), readout="p_infeasible",
                     score={c: 1.0 - x[k][keep].astype(np.float64)[:, None]
                            for c, k in (("orig", "original_probs"), ("inpaint", "inpainted_probs"))})


def reviewed_frames(pkl_path: Path, trajectory_ids: np.ndarray, dataset: np.ndarray) -> list[dict]:
    """Images of the reviewed inpainting pairs, aligned with a dump's rows (asserted)."""
    trajs, keep = reviewed_rows(pkl_path)
    kept = [t for t, k in zip(trajs, keep) if k]
    if [(t["dataset"], t["trajectory_id"]) for t in kept] != list(zip(dataset, trajectory_ids.tolist())):
        raise ValueError(f"{pkl_path}: reviewed rows do not line up with the dump")
    return kept


def write_detector_scores(path: Path, meta: dict, ref: ModelDump, scores: dict[str, dict[str, np.ndarray]]) -> None:
    """Host-side detector scores on the rows of dump `ref`: score/<readout>/<cond> = (n, 2)."""
    n = len(ref.trajectory_ids)
    with h5py.File(path, "w") as f:
        f.attrs["metadata"] = json.dumps(meta)
        f["trajectory_ids"] = ref.trajectory_ids
        f["dataset"] = ref.dataset.astype("S")
        f["language"] = np.array(ref.language, dtype=h5py.string_dtype())
        for rd, conds in scores.items():
            if "orig" not in conds:
                raise ValueError(f"{rd}: no 'orig' scores")
            for c, v in conds.items():
                v = np.asarray(v, np.float64)
                if v.shape != (n, N_FRAMES) or not np.all(np.isfinite(v)):
                    raise ValueError(f"{rd}/{c}: shape {v.shape} or values invalid, expected ({n}, {N_FRAMES})")
                f[f"score/{rd}/{c}"] = v


def load_detector_scores(path: Path, labels: dict[str, str]) -> dict[str, ModelDump]:
    """`write_detector_scores` file -> one detector ModelDump per readout in `labels` (readout -> label)."""
    with h5py.File(path) as f:
        meta = json.loads(f.attrs["metadata"])
        missing = set(labels) - set(f["score"])
        if missing:
            raise ValueError(f"{path}: readouts {missing} absent; has {list(f['score'])}")
        out = {}
        for rd, lab in labels.items():
            out[lab] = ModelDump(label=lab, kind="detector", paths=[str(path)], meta=[meta],
                                 trajectory_ids=f["trajectory_ids"][()], dataset=_str_array(f["dataset"]),
                                 language=_str_array(f["language"]), readout=rd,
                                 score={c: f[f"score/{rd}/{c}"][()] for c in f[f"score/{rd}"]})
    return out
