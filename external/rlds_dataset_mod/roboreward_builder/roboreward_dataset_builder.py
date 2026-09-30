"""Convert RoboReward HuggingFace dataset to RLDS/TFDS format.

Produces a TFDS dataset schema aligned with bridge_dataset:
- observation/image_0: uint8 (256,256,3) JPEG
- observation/state:   float32 (7,)  -- zeros placeholder
- action:              float32 (7,)  -- zeros placeholder
- reward:              float32 scalar -- 1.0 on last step (RLDS convention, overwritten at load time)
- discount:            float32 scalar -- always 1.0
- is_first/last/terminal: bool scalar
- language_instruction: string
- episode_metadata: file_path, has_language, has_image_*, episode_id,
                    roboreward_score, source_dataset

No language_embedding stored -- MUSE recomputes it at train time.
No image_1/2/3 -- RoboReward has one camera per trajectory.

Usage (set env vars, then run):
    export ROBOREWARD_SNAPSHOT_DIR="/path/to/snapshots/469b9af..."
    export ROBOREWARD_SOURCE="bridge"        # sub-dataset prefix in metadata.jsonl
    export ROBOREWARD_N_WORKERS="10"
    export ROBOREWARD_MAX_IN_MEMORY="100"    # must be divisible by N_WORKERS
    export ROBOREWARD_N_EPISODES="-1"        # -1 = all, >0 = subset for testing

    python roboreward_dataset_builder.py /path/to/output/dir
"""

import os
import sys

# ── path setup: MUST come before rlds_dataset_mod import ──────────────────
_BUILDER_DIR = os.path.dirname(os.path.abspath(__file__))
_MOD_DIR     = os.path.dirname(_BUILDER_DIR)
if _MOD_DIR not in sys.path:
    sys.path.insert(0, _MOD_DIR)

import json
import re
import types

import cv2
import numpy as np
import tensorflow_datasets as tfds

from rlds_dataset_mod.multithreaded_adhoc_tfds_builder import MultiThreadedAdhocDatasetBuilder

# ── Config from environment ────────────────────────────────────────────────
SNAPSHOT_DIR   = os.environ["ROBOREWARD_SNAPSHOT_DIR"]
SOURCE_DATASET = os.environ.get("ROBOREWARD_SOURCE", "bridge")
N_WORKERS      = int(os.environ.get("ROBOREWARD_N_WORKERS", "10"))
MAX_IN_MEMORY  = int(os.environ.get("ROBOREWARD_MAX_IN_MEMORY", "100"))
N_EPISODES     = int(os.environ.get("ROBOREWARD_N_EPISODES", "-1"))  # -1 = all

assert MAX_IN_MEMORY % N_WORKERS == 0, \
    f"MAX_IN_MEMORY ({MAX_IN_MEMORY}) must be divisible by N_WORKERS ({N_WORKERS})"

IMAGE_SIZE = (256, 256)

# metadata.jsonl uses "val"; TFDS uses "validation"
_SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}

# ── Feature spec ───────────────────────────────────────────────────────────
FEATURES = tfds.features.FeaturesDict({
    "steps": tfds.features.Dataset({
        "observation": tfds.features.FeaturesDict({
            "image_0": tfds.features.Image(
                shape=IMAGE_SIZE + (3,),
                dtype=np.uint8,
                encoding_format="jpeg",
                doc="Primary camera RGB observation resized to 256x256.",
            ),
            "state": tfds.features.Tensor(
                shape=(7,),
                dtype=np.float32,
                doc="Placeholder robot state (zeros — no proprioception in RoboReward).",
            ),
        }),
        "action": tfds.features.Tensor(
            shape=(7,),
            dtype=np.float32,
            doc="Placeholder action (zeros — no actions in RoboReward).",
        ),
        "reward": tfds.features.Scalar(
            dtype=np.float32,
            doc="1.0 on final step, 0.0 otherwise.",
        ),
        "discount": tfds.features.Scalar(
            dtype=np.float32,
            doc="Discount factor, always 1.0.",
        ),
        "is_first": tfds.features.Scalar(
            dtype=np.bool_,
            doc="True on first step of the episode.",
        ),
        "is_last": tfds.features.Scalar(
            dtype=np.bool_,
            doc="True on last step of the episode.",
        ),
        "is_terminal": tfds.features.Scalar(
            dtype=np.bool_,
            doc="True on last step of the episode.",
        ),
        "language_instruction": tfds.features.Text(
            doc="Language task instruction from RoboReward.",
        ),
    }),
    "episode_metadata": tfds.features.FeaturesDict({
        "file_path": tfds.features.Text(
            doc="Relative MP4 path within RoboReward snapshot directory.",
        ),
        "has_language": tfds.features.Scalar(
            dtype=np.bool_,
            doc="True if language instruction is non-empty.",
        ),
        "has_image_0": tfds.features.Scalar(
            dtype=np.bool_,
            doc="Always True — primary image present.",
        ),
        "has_image_1": tfds.features.Scalar(
            dtype=np.bool_,
            doc="Always False — no secondary camera.",
        ),
        "has_image_2": tfds.features.Scalar(
            dtype=np.bool_,
            doc="Always False.",
        ),
        "has_image_3": tfds.features.Scalar(
            dtype=np.bool_,
            doc="Always False.",
        ),
        "episode_id": tfds.features.Scalar(
            dtype=np.int32,
            doc="Zero-based index of this episode within its split.",
        ),
        "roboreward_score": tfds.features.Scalar(
            dtype=np.int32,
            doc="Human quality rating 1-5 from RoboReward dataset.",
        ),
        "source_dataset": tfds.features.Text(
            doc="Original OXE source dataset name (e.g. 'bridge').",
        ),
    }),
})


# ── Frame extraction ───────────────────────────────────────────────────────
def mp4_to_frames(mp4_path: str) -> list:
    """Decode all frames from an MP4, resize to IMAGE_SIZE, return list of uint8 arrays."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[:2] != IMAGE_SIZE:
            frame_rgb = cv2.resize(frame_rgb, (IMAGE_SIZE[1], IMAGE_SIZE[0]),
                                   interpolation=cv2.INTER_AREA)
        frames.append(frame_rgb.astype(np.uint8))
    cap.release()
    assert frames, f"No frames decoded from {mp4_path}"
    return frames


# ── Per-episode processing (top-level for multiprocessing pickling) ────────
def _build_episode(row: dict, episode_id: int, split_dir: str) -> dict:
    """Convert one metadata.jsonl row into an RLDS episode dict."""
    mp4_path = os.path.join(split_dir, row["file_name"])
    frames   = mp4_to_frames(mp4_path)
    n        = len(frames)
    task     = row["task"]
    source   = row["file_name"].split("/")[0]

    steps = [
        {
            "observation": {
                "image_0": frames[i],
                "state":   np.zeros(7, dtype=np.float32),
            },
            "action":               np.zeros(7, dtype=np.float32),
            "reward":               np.float32(i == n - 1),
            "discount":             np.float32(1.0),
            "is_first":             np.bool_(i == 0),
            "is_last":              np.bool_(i == n - 1),
            "is_terminal":          np.bool_(i == n - 1),
            "language_instruction": task,
        }
        for i in range(n)
    ]

    return {
        "steps": steps,
        "episode_metadata": {
            "file_path":        row["file_name"],
            "has_language":     np.bool_(bool(task)),
            "has_image_0":      np.bool_(True),
            "has_image_1":      np.bool_(False),
            "has_image_2":      np.bool_(False),
            "has_image_3":      np.bool_(False),
            "episode_id":       np.int32(episode_id),
            "roboreward_score": np.int32(row["reward"]),
            "source_dataset":   source,
        },
    }


# ── Metadata loading ───────────────────────────────────────────────────────
def _load_rows(tfds_split: str) -> list:
    """Load metadata.jsonl rows for a split, filtered to SOURCE_DATASET."""
    meta_split = _SPLIT_MAP[tfds_split]
    meta_file  = os.path.join(SNAPSHOT_DIR, meta_split, "metadata.jsonl")
    rows = [json.loads(line) for line in open(meta_file)]
    rows = [r for r in rows if r["file_name"].startswith(SOURCE_DATASET + "/")]
    if N_EPISODES > 0:
        rows = rows[:N_EPISODES]
    return rows


# ── Generator function for the builder ────────────────────────────────────
# Must be a top-level function (not lambda/closure) for multiprocessing pickle.
def roboreward_generator(split: str):
    """Called by ParallelSplitBuilder as roboreward_generator(split="train[0:100]").

    Parses the TFDS slice string, loads the corresponding metadata rows,
    and yields plain episode dicts (key assigned externally by the builder).
    """
    m = re.fullmatch(r"(\w+)(?:\[(\d+):(\d+)?\])?", split)
    assert m, f"Unexpected split string format: {split!r}"
    split_name = m.group(1)
    start      = int(m.group(2)) if m.group(2) else 0
    end        = int(m.group(3)) if m.group(3) else None

    rows       = _load_rows(split_name)
    rows_slice = rows[start:end]
    split_dir  = os.path.join(SNAPSHOT_DIR, _SPLIT_MAP[split_name])

    for local_i, row in enumerate(rows_slice):
        episode_id = start + local_i
        yield _build_episode(row, episode_id, split_dir)


# ── Builder subclass ───────────────────────────────────────────────────────
# IMPORTANT: subclassing (not instantiating) MultiThreadedAdhocDatasetBuilder
# so that inspect.getfile(RoboRewardDataset) returns THIS file's path — a real
# filesystem path — instead of the rlds_dataset_mod package path which resolves
# to a MultiplexedPath that lacks .parts and crashes TFDS's code_path property.
class RoboRewardDataset(MultiThreadedAdhocDatasetBuilder):
    pass


def build(target_dir: str) -> None:
    dataset_name = f"roboreward_{SOURCE_DATASET.replace('/', '_')}"
    print(f"Building '{dataset_name}' → {target_dir}")
    print(f"  Source:     {SOURCE_DATASET}")
    print(f"  Snapshot:   {SNAPSHOT_DIR}")
    print(f"  N_WORKERS:  {N_WORKERS}")
    print(f"  MAX_IN_MEM: {MAX_IN_MEMORY}")

    split_counts = {}
    for tfds_split in ("train", "validation", "test"):
        n = len(_load_rows(tfds_split))
        split_counts[tfds_split] = n
        print(f"  {tfds_split:12s}: {n} episodes")

    # SimpleNamespace provides .num_examples used by ParallelSplitBuilder
    split_datasets = {
        split: types.SimpleNamespace(num_examples=n)
        for split, n in split_counts.items()
        if n > 0
    }

    builder = RoboRewardDataset(
        name=dataset_name,
        version=tfds.core.Version("1.0.0"),
        features=FEATURES,
        split_datasets=split_datasets,
        data_dir=target_dir,
        description=(
            f"RoboReward trajectories converted to RLDS format. "
            f"Source OXE sub-dataset: {SOURCE_DATASET}. "
            f"Actions and proprio are zero placeholders."
        ),
        generator_fcn=roboreward_generator,
        n_workers=N_WORKERS,
        max_episodes_in_memory=MAX_IN_MEMORY,
    )
    builder.download_and_prepare()
    print(f"\nDone. Dataset written to {target_dir}/{dataset_name}/")


if __name__ == "__main__":
    assert len(sys.argv) == 2, f"Usage: python {sys.argv[0]} <target_dir>"
    build(target_dir=sys.argv[1])
