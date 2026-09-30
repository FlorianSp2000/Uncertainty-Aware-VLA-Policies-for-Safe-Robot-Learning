"""Image perturbations of `experiments/eval_perturbation_sensitivity.py`, jax-free so the host
analysis (say_no.perturbation.images) builds the identical images for its CLIP baselines.

JPEG re-encoding matched to the inpainting's pixel change, and masked composites that put the
inpainted pixels only inside (local) or only outside (global) the edited region.
"""
from __future__ import annotations

import io
import re

import numpy as np
from PIL import Image
from scipy import ndimage

IMAGE_CONDITION = re.compile(r"^(jpeg_orig|composite_local_r(\d+)|composite_global_r(\d+))$")


def is_image_condition(c: str) -> bool:
    return bool(IMAGE_CONDITION.match(c))


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    if img.dtype != np.uint8 or img.ndim != 3:
        raise ValueError(f"expected HxWx3 uint8, got {img.shape} {img.dtype}")
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def median_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Per frame: median over pixels and channels of |a - b| in grey levels."""
    return float(np.median(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def mean_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def change_mask(orig: np.ndarray, edited: np.ndarray, threshold: int, radius: int) -> np.ndarray:
    """Pixels whose max-channel |edited - orig| exceeds threshold, dilated by a disk of `radius` px."""
    raw = np.abs(edited.astype(np.int16) - orig.astype(np.int16)).max(axis=-1) > threshold
    if radius == 0:
        return raw
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return ndimage.binary_dilation(raw, structure=(yy ** 2 + xx ** 2) <= radius ** 2)


def choose_jpeg_quality(origs: np.ndarray, inpaints: np.ndarray, qualities: list[int]) -> tuple[int, dict]:
    """Quality whose median (over frames) per-frame median |delta| is closest to the inpaintings'
    (frames: (N, H, W, 3)). The per-frame median is integer-valued, so ties are broken by the
    per-frame mean |delta|, which also counts the large in-mask changes."""
    target = float(np.median([median_abs_delta(o, p) for o, p in zip(origs, inpaints)]))
    target_mean = float(np.median([mean_abs_delta(o, p) for o, p in zip(origs, inpaints)]))
    enc = {int(q): [jpeg(o, q) for o in origs] for q in qualities}
    per_q = {q: float(np.median([median_abs_delta(o, j) for o, j in zip(origs, js)])) for q, js in enc.items()}
    per_q_mean = {q: float(np.median([mean_abs_delta(o, j) for o, j in zip(origs, js)])) for q, js in enc.items()}
    best = min(per_q, key=lambda q: (abs(per_q[q] - target), abs(per_q_mean[q] - target_mean), -q))
    return best, {"target_median_abs_delta_inpaint": target, "target_mean_abs_delta_inpaint": target_mean,
                  "median_abs_delta_by_quality": per_q, "mean_abs_delta_by_quality": per_q_mean, "chosen_quality": best}


def image_conditions(trajs: list[dict], conditions: list[str], qualities: list[int], threshold: int) -> tuple[dict, dict, dict]:
    """Builds every requested image condition for all trajectories.

    Returns images[cond] (n, 2, H, W, 3), masks[r] (n, 2, H, W) bool, stats (json-able)."""
    wanted = [c for c in conditions if is_image_condition(c)]
    if not wanted:
        return {}, {}, {}
    orig = np.stack([np.stack([t["first_image"], t["last_image"]]) for t in trajs])
    inp = np.stack([np.stack([t["first_image_inpainted"], t["last_image_inpainted"]]) for t in trajs])
    if orig.shape != inp.shape or orig.dtype != np.uint8 or inp.dtype != np.uint8:
        raise ValueError(f"original / inpainted frames disagree: {orig.shape} {orig.dtype} vs {inp.shape} {inp.dtype}")
    flat_o, flat_i = orig.reshape(-1, *orig.shape[2:]), inp.reshape(-1, *inp.shape[2:])
    images, masks, stats = {}, {}, {"mask_threshold": threshold}
    stats["median_abs_delta_inpaint_per_frame"] = [median_abs_delta(o, p) for o, p in zip(flat_o, flat_i)]
    if "jpeg_orig" in wanted:
        q, qstats = choose_jpeg_quality(flat_o, flat_i, qualities)
        stats["jpeg"] = qstats
        images["jpeg_orig"] = np.stack([jpeg(o, q) for o in flat_o]).reshape(orig.shape)
        stats["median_abs_delta_jpeg_per_frame"] = [median_abs_delta(o, j) for o, j in
                                                    zip(flat_o, images["jpeg_orig"].reshape(flat_o.shape))]
    radii = sorted({int(m.group(2) or m.group(3)) for c in wanted if (m := IMAGE_CONDITION.match(c)).group(1) != "jpeg_orig"})
    for r in radii:
        m = np.stack([change_mask(o, p, threshold, r) for o, p in zip(flat_o, flat_i)]).reshape(orig.shape[:-1])
        masks[r] = m
        area = m.reshape(m.shape[0] * 2, -1).mean(axis=1)
        stats[f"mask_area_fraction_r{r}"] = {"median": float(np.median(area)), "p10": float(np.percentile(area, 10)),
                                            "p90": float(np.percentile(area, 90)), "min": float(area.min()),
                                            "max": float(area.max()), "fraction_empty": float((area == 0).mean())}
        if f"composite_local_r{r}" in wanted:
            images[f"composite_local_r{r}"] = np.where(m[..., None], inp, orig)
        if f"composite_global_r{r}" in wanted:
            images[f"composite_global_r{r}"] = np.where(m[..., None], orig, inp)
    return images, masks, stats
