"""Seen vs unseen instructions: which evaluation trajectories carry an instruction the critic trained on.

Input is the per-trajectory instruction walk of `external/V-GPS/experiments/extract_instruction_vocabulary.py`
(one HDF5 group per `<tfds name>/<split>`). "Seen" = the instruction string occurs in at least one
training-split trajectory of the same source dataset. One unique string counts as one task; paraphrases
of a training instruction therefore count as unseen, so "unseen instruction" is an upper bound on
"unseen task". Scenes and objects are not compared.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

TFDS_NAME = {"bridge": "bridge_dataset", "fractal": "fractal20220817_data"}
NORMALISATIONS = ("exact", "normalised")


def normalise(s: str, how: str) -> str:
    """'normalised': case, surrounding whitespace, repeated inner whitespace and trailing '.' ignored."""
    if how == "exact":
        return s
    if how == "normalised":
        return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".").strip()
    raise ValueError(f"unknown normalisation {how!r}")


def load_walk(path: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """{'bridge'|'fractal': {'train'|'val': {language, traj_len, first_action}}}."""
    out = {}
    with h5py.File(path) as f:
        if f.attrs["max_trajectories"] != 0:
            raise ValueError(f"{path}: capped walk (max_trajectories={f.attrs['max_trajectories']}), not the full splits")
        for ds, tfds_name in TFDS_NAME.items():
            out[ds] = {}
            for split in ("train", "val"):
                g = f[f"{tfds_name}/{split}"]
                lang = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["language"][()]])
                if (lang == "").any():
                    raise ValueError(f"{path}: empty instruction in {tfds_name}/{split}")
                out[ds][split] = {"language": lang, "traj_len": g["traj_len"][()], "first_action": g["first_action"][()]}
    return out


def overlap_counts(walk: dict, how: str) -> dict:
    """Per source dataset: trajectory and unique-instruction counts of both splits, and their overlap."""
    out = {}
    for ds, splits in walk.items():
        tr = {normalise(s, how) for s in splits["train"]["language"]}
        va_rows = [normalise(s, how) for s in splits["val"]["language"]]
        va = set(va_rows)
        out[ds] = {
            "train_trajectories": int(len(splits["train"]["language"])),
            "val_trajectories": len(va_rows),
            "train_unique": len(tr), "val_unique": len(va),
            "unique_in_both": len(tr & va), "unique_val_only": len(va - tr), "unique_train_only": len(tr - va),
            "val_trajectories_with_seen_instruction": int(sum(s in tr for s in va_rows)),
        }
    return out


def seen_mask(language: np.ndarray, dataset: np.ndarray, walk: dict, how: str) -> np.ndarray:
    """True where the row's instruction occurs in the training split of its own source dataset."""
    vocab = {ds: {normalise(s, how) for s in walk[ds]["train"]["language"]} for ds in walk}
    unknown = set(dataset) - set(vocab)
    if unknown:
        raise ValueError(f"unknown source datasets {unknown}")
    return np.array([normalise(s, how) in vocab[d] for s, d in zip(language, dataset)])


def training_count(language: np.ndarray, dataset: np.ndarray, walk: dict) -> np.ndarray:
    """Number of training-split trajectories of the row's own source dataset with exactly this instruction."""
    counts = {ds: Counter(walk[ds]["train"]["language"]) for ds in walk}
    return np.array([counts[d][s] for s, d in zip(language, dataset)])


def nearest_training(language: np.ndarray, dataset: np.ndarray, walk: dict) -> tuple[np.ndarray, np.ndarray]:
    """Closest training-split instruction of the row's own source dataset and its character-level
    similarity (rapidfuzz `ratio`, 0-100; 100 = identical string). Separates paraphrases and typos of a
    training instruction (high) from new object/place combinations (low)."""
    from rapidfuzz import fuzz, process

    vocab = {ds: sorted(set(walk[ds]["train"]["language"])) for ds in walk}
    hits = [process.extractOne(s, vocab[d], scorer=fuzz.ratio) for s, d in zip(language, dataset)]
    return np.array([h[0] for h in hits]), np.array([h[1] for h in hits])


def _fingerprints(split: dict) -> set[tuple]:
    return {(int(n), tuple(np.round(np.asarray(a, np.float32), 4))) for n, a in zip(split["traj_len"], split["first_action"])}


def check_pkl_split(pkl_path: Path, walk: dict, split: str) -> dict:
    """Every pkl trajectory must be one of `split`'s trajectories (length + first action) and not of the other split.

    The pkls were extracted with 4 files read in parallel, so their order differs from the walk; the
    (length, first action) fingerprint identifies the source trajectory instead of the index.
    """
    other = {"train": "val", "val": "train"}[split]
    rows = pickle.load(open(pkl_path, "rb"))["trajectories"]
    counts = {}
    for ds in walk:
        fp_in, fp_out = _fingerprints(walk[ds][split]), _fingerprints(walk[ds][other])
        mine = [r for r in rows if r["dataset"] == ds]
        keys = [(int(r["traj_length"]), tuple(np.round(np.asarray(r["first_action"], np.float32), 4))) for r in mine]
        missing = [r["trajectory_id"] for r, k in zip(mine, keys) if k not in fp_in]
        if missing:
            raise ValueError(f"{pkl_path.name} {ds}: trajectories {missing[:10]} not found in the {split} split")
        counts[ds] = {"trajectories": len(mine), "fingerprint_also_in_" + other: int(sum(k in fp_out for k in keys))}
    return counts
