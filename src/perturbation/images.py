"""The image conditions of the cluster dumps, rebuilt on the host from the same code.

SSOT: external/V-GPS/experiments/utils/image_perturbations.py (jax-free; the cluster eval imports it
through experiments/utils/perturbations.py). Loaded by file path because the V-GPS tree is not an
installed package on the host. `rebuild_like_dump` reads the mask threshold and the JPEG quality
from the dump's metadata, rebuilds every image condition, and asserts per-frame median |delta| and
mask area equal the values the cluster logged — so host-side baselines see the identical pixels.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

SSOT = Path(__file__).resolve().parents[2] / "external/V-GPS/experiments/utils/image_perturbations.py"


def _load():
    if not SSOT.exists():
        raise FileNotFoundError(SSOT)
    spec = importlib.util.spec_from_file_location("vgps_image_perturbations", SSOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IP = _load()


def rebuild_like_dump(trajs: list[dict], dump: Path, n_rows: int | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Image conditions of `dump` for `trajs` (the dump's rows, same order; or its first `n_rows`)
    -> images[cond] (n, 2, H, W, 3) incl. 'inpaint', and the parameters used. Raises if any
    per-frame statistic differs from the one stored in the dump."""
    rows = slice(None) if n_rows is None else slice(0, n_rows)
    with h5py.File(dump) as f:
        meta = json.loads(f.attrs["metadata"])
        conds = [c for c in meta["conditions"] if IP.is_image_condition(c)]
        logged_delta = {c: f[f"image_median_abs_delta/{c}"][rows] for c in f["image_median_abs_delta"]}
        logged_area = {k: f[f"mask_area_fraction/{k}"][rows] for k in f["mask_area_fraction"]}
        ids = f["trajectory_ids"][rows]
    if len(trajs) != len(ids) or [int(t["trajectory_id"]) for t in trajs] != ids.tolist():
        raise ValueError(f"trajectories do not line up with {dump}")
    st = meta["image_stats"]
    quality, threshold = int(st["jpeg"]["chosen_quality"]), int(st["mask_threshold"])
    images, masks, _ = IP.image_conditions(trajs, conds, [quality], threshold)
    orig = np.stack([np.stack([t["first_image"], t["last_image"]]) for t in trajs])
    images["inpaint"] = np.stack([np.stack([t["first_image_inpainted"], t["last_image_inpainted"]]) for t in trajs])
    for c, im in images.items():
        d = np.array([[IP.median_abs_delta(o, x) for o, x in zip(oo, xx)] for oo, xx in zip(orig, im)])
        if not np.array_equal(d.astype(np.float32), logged_delta[c]):
            raise ValueError(f"{c}: per-frame median |delta| differs from {dump} "
                             f"({int((d != logged_delta[c]).sum())} frames)")
    for r, m in masks.items():
        area = m.reshape(*m.shape[:2], -1).mean(axis=-1)
        if not np.allclose(area, logged_area[f"r{r}"], atol=1e-6):
            raise ValueError(f"mask r{r}: area differs from {dump}")
    return images, {"jpeg_quality": quality, "mask_threshold": threshold, "conditions": conds,
                    "verified_against": str(dump)}
