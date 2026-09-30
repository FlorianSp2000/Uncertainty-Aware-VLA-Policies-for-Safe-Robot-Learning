"""Frame decoding for iva_cp pickle records.

Two dump schemas exist and both must keep working:

  legacy  record["images"]         = uint8 array (K, H, W, 3), K = strided subset
          record["image_stride"]   = stride used
          record["image_encoding"] = absent

  current record["images"]         = list of PNG-encoded bytes, one per frame
          record["image_encoding"] = "png"

`decode_frames` normalises both to a (K, H, W, 3) uint8 array; `frame_indices`
reports which original timesteps those K frames correspond to, so callers can
align a filmstrip to the timestep axis without assuming a stride.
"""

import numpy as np
from PIL import Image
import io


def frame_indices(record: dict) -> np.ndarray:
    """Original timestep index of each stored frame."""
    if record["images"] is None:
        return np.empty(0, dtype=int)
    n = len(record["images"])
    stride = record.get("image_stride", 1) if record.get("image_encoding") is None else 1
    return np.arange(n) * stride


def decode_frames(record: dict, stride: int = 1) -> np.ndarray:
    """Stored frames as a (K, H, W, 3) uint8 array.

    `stride` subsamples the stored frames further (for filmstrips); it composes
    with whatever stride the dump already applied.
    """
    imgs = record["images"]
    assert imgs is not None, (
        f"record {record['task']}#{record['episode_idx']} has no images — "
        "rerun inference with --keep_images"
    )
    if record.get("image_encoding") == "png":
        frames = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in imgs[::stride]]
        out = np.stack(frames, axis=0)
    else:
        out = np.asarray(imgs)[::stride]
    assert out.ndim == 4 and out.shape[-1] == 3, f"unexpected frame array {out.shape}"
    return out.astype(np.uint8)
