"""Novelty of a frame in the critic's OWN encoder features, as a detector next to the ensemble.

Answers "is spread = novelty in the ensemble's own representation?": `feat/<cond>` (n, members,
frames, D) of a dump is scored against the same checkpoint's features of training-split originals
(Mahalanobis with Ledoit-Wolf shrinkage, or cosine k-th NN; fit per member and source dataset,
min over datasets, mean over members — say_no.baselines.embedding.frame_novelty).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from say_no.baselines.embedding import FeatureNovelty, frame_novelty, read_features
from say_no.perturbation.load import ModelDump

READOUTS = {"mahalanobis": "enc_mahalanobis", "knn": "enc_knn"}


def encoder_novelty(ens: ModelDump, ref_path: Path, k: int) -> dict[str, ModelDump]:
    """One detector ModelDump per novelty method, on the rows and conditions of `ens` that carry
    features. Labels: '<ens label> | <readout>'."""
    feats, dataset, ids = {}, None, None
    for p in ens.paths:
        f, d, i = read_features(Path(p))
        if dataset is None:
            dataset, ids = d, i
        if not (np.array_equal(d, dataset) and np.array_equal(i, ids)):
            raise ValueError(f"{p}: feature rows differ between files of {ens.label}")
        feats.update(f)
    if not (np.array_equal(ids, ens.trajectory_ids) and np.array_equal(dataset, ens.dataset)):
        raise ValueError(f"{ens.label}: feature rows do not line up with its Q rows")
    ref, ref_ds, _ = read_features(ref_path)
    if set(ref) != {"orig"}:
        raise ValueError(f"{ref_path}: reference must hold originals only, has {list(ref)}")
    if ref["orig"].shape[1] != feats["orig"].shape[1]:
        raise ValueError(f"{ref_path}: {ref['orig'].shape[1]} members vs {feats['orig'].shape[1]} in {ens.label}")
    out = {}
    for method, rd in READOUTS.items():
        s = frame_novelty(lambda method=method: FeatureNovelty(method, k=k), ref["orig"], ref_ds, feats)
        lab = f"{ens.label} | {rd}"
        out[lab] = ModelDump(label=lab, kind="detector", paths=ens.paths + [str(ref_path)],
                             meta=[{"reference": str(ref_path), "method": method, "k": k}],
                             trajectory_ids=ens.trajectory_ids, dataset=ens.dataset, language=ens.language,
                             readout=rd, score=s)
    return out
