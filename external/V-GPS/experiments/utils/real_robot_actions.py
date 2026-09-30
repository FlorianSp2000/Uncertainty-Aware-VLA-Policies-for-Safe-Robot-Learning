"""Action normalisation for the real-robot dataset, on the Bridge/Fractal recipe.

The Bridge/Fractal ensemble was fitted on actions mapped to [-1, 1] with per-dimension
``p001``/``p999`` bounds, gripper binarised first. That mapping lives in
``make_dataset_from_rlds``, which the real-robot path does not go through -- it builds
trajectories from the ``.npz`` cache and wraps them with ``from_generator``. So the same three
octo functions are called here instead of being reimplemented:
``binarize_gripper_actions``, ``get_dataset_statistics``, ``normalize_action_and_proprio``.

Training and evaluation agree because both call :func:`action_statistics` with the same
arguments and hit the same cached JSON, not because two call sites were kept in step by hand.

Statistics are taken over every usable teleoperated demonstration, train and validation
together -- octo's analogue is ``split="all"`` -- and never over the rollouts or the
leave-one-out scenes, which are test data.

Used by ``octo/data/real_robot.py`` and ``experiments/eval_real_robot.py``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import dlimp as dl
import numpy as np
import tensorflow as tf

from octo.data.utils.data_utils import (
    NormalizationType,
    binarize_gripper_actions,
    get_dataset_statistics,
    normalize_action_and_proprio,
    tree_map,
)

ACTION_DIM = 7  # x, y, z, rx, ry, rz, gripper
GRIPPER_DIM = 6


def resolve_normalization(name: str) -> NormalizationType | None:
    """Config and sbatch carry the scheme as a plain string, because ``NormalizationType`` is a
    ``str`` Enum that ml_collections will not accept from the command line."""
    if name is None or str(name).lower() == "none":
        return None
    try:
        return NormalizationType(str(name).lower())
    except ValueError:
        raise ValueError(
            f"unknown normalization {name!r}; expected one of "
            f"{[t.value for t in NormalizationType] + ['none']}") from None


def teleop_rows(data_dir: str | Path, pool: str = "teleop") -> list[dict]:
    """The training episodes that define this robot's action space -- the teleoperated
    demonstrations for the lab's robot, the ``finetune`` successes for SAFE's rollouts.

    Rows the converter marked corrupt are excluded: a broken recording would drag the
    percentile bounds, and it is not part of the training population either.
    """
    index_path = Path(data_dir) / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found -- run experiments/convert_real_robot.py first")
    rows = [r for r in json.loads(index_path.read_text())
            if r["pool"] == pool and not r["corrupt"]]
    if not rows:
        raise ValueError(f"no usable {pool} episodes in {index_path}")
    return rows


def _raw_actions(data_dir: str | Path, rows: list[dict]) -> np.ndarray:
    """Every demonstration action, unstrided. Reads only the ``actions`` member of each cache
    entry, so the frames stay on disk."""
    data_dir = Path(data_dir)
    out = []
    for row in rows:
        payload = np.load(data_dir / row["pool"] / f"{row['key']}.npz", allow_pickle=True)
        actions = payload["actions"]
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"{row['key']}: actions {actions.shape}, expected (T, {ACTION_DIM})")
        out.append(actions)
    return np.concatenate(out)


def gripper_open_width(data_dir: str | Path, rows: list[dict]) -> float:
    """Commanded width of the fully open gripper, in the recorder's own units.

    The column is an absolute width rather than the {0, 1} Bridge/Fractal emit, so
    ``binarize_gripper_actions`` -- which tests against 0.95 and 0.05 -- needs the scale that
    makes "fully open" equal 1. Taken from the data, not asserted, so a recording made with
    different hardware limits does not silently binarise everything as closed.
    """
    widths = _raw_actions(data_dir, rows)[:, GRIPPER_DIM]
    width = float(widths.max())
    if width <= 0.0:
        raise ValueError(f"gripper column never opens (max {width}) -- wrong column or dead channel")
    return width


def binarize_episode(actions: np.ndarray, open_width: float) -> np.ndarray:
    """Snap the gripper column to fully-closed or fully-open, as Bridge/Fractal does.

    Run over the **whole recorded episode, before any striding**. Binarisation carries the state
    reached after a transition backwards through the intermediate frames, so its result depends
    on the sequence it sees: binarising each phase-offset slice separately would make the gripper
    column depend on the stride, and would make the training loader disagree with
    ``experiments/eval_real_robot.py`` on exactly those frames. Bridge/Fractal binarise the
    recorded episode; so do we.

    Scaled into [0, 1] and back because ``binarize_gripper_actions`` tests against fixed 0.95 and
    0.05 thresholds.
    """
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions {actions.shape}, expected (T, {ACTION_DIM})")
    unit = tf.constant(actions[:, GRIPPER_DIM] / open_width, dtype=tf.float32)
    out = np.array(actions, dtype=np.float32, copy=True)
    out[:, GRIPPER_DIM] = np.asarray(binarize_gripper_actions(unit)) * open_width
    return out


def _actions_only_dataset(data_dir: str | Path, rows: list[dict], frame_stride: int,
                          open_width: float, causal_gripper: bool) -> dl.DLataset:
    """The trajectories the statistics are taken over: actions alone, strided and phase-shifted
    exactly as ``octo/data/real_robot.py`` strides them, so the bounds describe the frames
    training actually sees."""
    data_dir = Path(data_dir)

    def gen():
        for row in rows:
            payload = np.load(data_dir / row["pool"] / f"{row['key']}.npz", allow_pickle=True)
            actions = binarize(payload["actions"].astype(np.float32), open_width, causal_gripper)
            for offset in range(frame_stride):
                idx = np.arange(offset, len(actions), frame_stride)
                if len(idx) == 0:
                    continue
                # The empty observation is required, not decorative: get_dataset_statistics maps
                # over traj["observation"] to pick up proprio, and raises KeyError without it.
                yield {"action": actions[idx], "observation": {}}

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={"action": tf.TensorSpec(shape=(None, ACTION_DIM), dtype=tf.float32),
                          "observation": {}},
    )
    dataset.__class__ = type("DLataset", (dl.DLataset, type(dataset)), dl.DLataset.__dict__.copy())
    dataset.is_flattened = False
    return dataset


def action_statistics(data_dir: str | Path, frame_stride: int, causal_gripper: bool,
                      force_recompute: bool = False, pool: str = "teleop") -> dict:
    """Per-dimension bounds for the real-robot action space, cached beside the ``.npz`` cache.

    Written by octo's own ``get_dataset_statistics``, so the file format and the percentiles are
    the ones ``normalize_action_and_proprio`` expects.

    The stride is part of the cache key even though the bounds do not depend on it: emitting all
    ``frame_stride`` phase offsets uses every recorded frame exactly once whatever the stride, so
    only the trajectory count moves. Keying on it anyway means a config change cannot silently
    reuse an entry written under different settings.
    """
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    rows = teleop_rows(data_dir, pool)
    open_width = gripper_open_width(data_dir, rows)
    stats = get_dataset_statistics(
        _actions_only_dataset(data_dir, rows, frame_stride, open_width, causal_gripper),
        hash_dependencies=(
            f"real_robot_{pool}",
            str(Path(data_dir).resolve()),
            f"episodes={len(rows)}",
            f"stride={frame_stride}",
            f"gripper_open_width={open_width:.6f}",
            f"causal_gripper={causal_gripper}",
        ),
        save_dir=str(data_dir),
        force_recompute=force_recompute,
    )
    # The cache round-trips through JSON, and normalize_action_and_proprio does arithmetic on
    # these; make_dataset_from_rlds converts the same way at dataset.py:512.
    stats = tree_map(np.array, stats)
    logging.info(
        f"real_robot action statistics: {stats['num_trajectories']} trajectories, "
        f"{stats['num_transitions']} frames, gripper open width {open_width:.4f}\n"
        f"  p001 {np.round(stats['action']['p001'], 4).tolist()}\n"
        f"  p999 {np.round(stats['action']['p999'], 4).tolist()}"
    )
    return stats


def load_statistics(path: str | Path) -> dict:
    """Fixed statistics from an octo ``dataset_statistics_*.json`` -- for a policy that already
    acts in a pretraining dataset's units, e.g. OpenVLA on the WidowX un-normalising with the
    Bridge statistics. Then the critic's first layer sees exactly the mapping it was fitted on,
    and nothing is estimated from the finetune pool."""
    stats = tree_map(np.array, json.loads(Path(path).read_text()))
    for k in ("p001", "p999", "max"):
        if stats["action"][k].shape != (ACTION_DIM,):
            raise ValueError(f"{path}: action/{k} has shape {stats['action'][k].shape}")
    if open_width_of(stats) <= 0.0:
        raise ValueError(f"{path}: gripper max {open_width_of(stats)} -- never opens")
    logging.info(f"fixed action statistics from {path}\n"
                 f"  p001 {np.round(stats['action']['p001'], 4).tolist()}\n"
                 f"  p999 {np.round(stats['action']['p999'], 4).tolist()}")
    return stats


def resolve_statistics(data_dir: str | Path, frame_stride: int, statistics_path: str | None,
                       pool: str, causal_gripper: bool, force_recompute: bool = False) -> dict:
    """The one place training and evaluation get their statistics from: a fixed file when one
    is configured, else the percentile bounds of the training pool."""
    if statistics_path:
        return load_statistics(statistics_path)
    return action_statistics(data_dir, frame_stride=frame_stride, causal_gripper=causal_gripper,
                             force_recompute=force_recompute, pool=pool)


def open_width_of(stats: dict) -> float:
    """Recover the open width from computed statistics -- binarising preserves the maximum, so
    it survives into ``action['max']`` and no second file has to be kept in step."""
    return float(stats["action"]["max"][GRIPPER_DIM])


def normalize_dataset(dataset: dl.DLataset, stats: dict,
                      normalization_type: NormalizationType) -> dl.DLataset:
    """Map actions to [-1, 1]. The gripper must already be binarised -- see
    :func:`binarize_episode` for why that cannot happen here."""
    return dataset.traj_map(
        lambda t: normalize_action_and_proprio(t, metadata=stats,
                                               normalization_type=normalization_type)
    )


def binarize_causal(actions: np.ndarray, open_width: float) -> np.ndarray:
    """Per-step gripper snap for a monitor running online: open iff the command is at least half
    of the open width. ``binarize_episode`` resolves an intermediate command by the state the
    episode reaches LATER, which a live monitor cannot know (look-ahead found on LIBERO). On two-valued grippers (SAFE-WidowX: 0 / 0.996) both give the same result."""
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions {actions.shape}, expected (T, {ACTION_DIM})")
    out = np.array(actions, dtype=np.float32, copy=True)
    out[:, GRIPPER_DIM] = np.where(actions[:, GRIPPER_DIM] >= 0.5 * open_width, open_width, 0.0)
    return out


def binarize(actions: np.ndarray, open_width: float, causal: bool) -> np.ndarray:
    """The one switch between the two gripper treatments, so training, statistics and
    evaluation cannot pick different ones."""
    return (binarize_causal if causal else binarize_episode)(actions, open_width)


def normalize_array(actions: np.ndarray, stats: dict, normalization_type: NormalizationType,
                    causal_gripper: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The same mapping for a single episode held in memory, which is how
    ``experiments/eval_real_robot.py`` meets the rollouts.

    Returns the normalised actions and the per-pose-dimension fraction of frames the clip
    flattened: the bounds come from the demonstrations, so a policy that acts outside them
    saturates here rather than reaching the critic, and that has to be visible in the log
    instead of silent.
    """
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions {actions.shape}, expected (T, {ACTION_DIM})")
    traj = {"observation": {},
            "action": tf.constant(binarize(actions, open_width_of(stats), causal_gripper))}
    traj = normalize_action_and_proprio(traj, metadata=stats,
                                        normalization_type=normalization_type)
    out = np.asarray(traj["action"])
    clipped = np.mean(np.abs(out[:, :GRIPPER_DIM]) >= 1.0 - 1e-6, axis=0)
    return out, clipped
