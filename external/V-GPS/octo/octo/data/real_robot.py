"""Rollout-cache loader for Q-ensemble training (thesis: SAFE WidowX / LIBERO rollouts, pool ``finetune``).

(The loader also supports a real-robot demonstration dataset from preliminary experiments that are not part of the thesis.)

Reads the per-episode cache written by ``experiments/convert_real_robot.py`` and yields
trajectories in the same shape as ``octo/data/iva.py``, so the same trajectory and frame
transforms apply.

Only the successful teleoperated demonstrations are training data, minus whatever the converter
marked corrupt. They are the same demonstrations the GR00T policy was finetuned on, which is
what makes the critic a monitor for *that* policy rather than for a different one.

Frame rate
----------
The demonstrations are recorded at 20 Hz, but the policy rollouts we score store one
observation per policy call -- one per 20 control steps. Training on every 20 Hz frame would
therefore fit the critic on a timestep the evaluation never shows it, and would stretch a
~217-step episode far past the horizon of ``gamma = 0.98``, so the discounted step count
saturates at its -50 floor for most of the episode and stops separating anything. Striding the
demonstrations by ``frame_stride`` fixes both: at the default 20 a demonstration becomes ~11
frames at the rollouts' own rate, and Q spans roughly -10 to 0.

Striding by k would discard (k-1)/k of the data, so all k phase offsets are emitted as separate
trajectories: every recorded frame is used exactly once, in a trajectory whose consecutive
frames are one policy call apart.

Actions
-------
The cache holds raw recorder units; the mapping to [-1, 1] happens here, so that this path and
``experiments/eval_real_robot.py`` cannot drift apart and so that changing the action treatment
does not mean re-running the conversion job.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import dlimp as dl
import numpy as np
import tensorflow as tf

# Layering note: the container puts /V-GPS on PYTHONPATH (train-q.def:24), so the action-space
# definition can live next to the converter that writes the cache rather than inside octo.
from experiments.utils.real_robot_actions import (
    binarize, normalize_dataset, open_width_of, resolve_normalization,
    resolve_statistics)

ACTION_DIM = 7
# Trailing frames treated as success (reward 0, no bootstrap). The window is a property of
# the recording -- the task is done for the last fraction of a second of a 20 Hz capture, which
# is the 3 frames the Bridge/Fractal and IVA loaders use. After striding, one frame already
# spans that and more, so the count must shrink with the stride or a third of every episode
# would be labelled terminal.
DEFAULT_NUM_TERMINAL = 1

TASKS = [
    "green_cup_to_blue", "green_cup_to_green", "green_cup_to_red",
    "white_cup_to_blue", "white_cup_to_green", "white_cup_to_red",
]


def _wrap_tf_dataset_as_dlimp(tf_dataset: tf.data.Dataset) -> dl.DLataset:
    """Swap in DLataset so .traj_map()/.frame_map() work, as octo/data/iva.py does."""
    if not isinstance(tf_dataset, dl.DLataset) and isinstance(tf_dataset, tf.data.Dataset):
        tf_dataset.__class__ = type(
            "DLataset", (dl.DLataset, type(tf_dataset)), dl.DLataset.__dict__.copy()
        )
        tf_dataset.is_flattened = False
    return tf_dataset


def _compute_mc_return(rewards: np.ndarray, masks: np.ndarray, discount: float) -> np.ndarray:
    mc = np.zeros_like(rewards)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + discount * running * masks[i]
        mc[i] = running
    return mc


def _episode_rewards(traj_len: int, step_reward: float, num_terminal: int) -> tuple[np.ndarray, np.ndarray]:
    """Successful demonstration: a per-step cost, then a terminal run with no bootstrap.

    Mirrors ``octo/data/iva.py:create_rewards_with_penalties`` for a trajectory with no false
    premise, which is every trajectory here -- the demonstrations are all successful.
    """
    rewards = np.full(traj_len, step_reward, dtype=np.float32)
    n_term = min(num_terminal, traj_len)
    rewards[traj_len - n_term:] = 0.0
    td_mask = np.concatenate([
        np.ones(traj_len - n_term, dtype=np.float32),
        np.zeros(n_term, dtype=np.float32),
    ])
    return rewards, td_mask


def load_index(data_dir: str) -> List[dict]:
    index_path = Path(data_dir) / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found -- run experiments/convert_real_robot.py first"
        )
    return json.loads(index_path.read_text())


def pool_rows(data_dir: str, pool: str) -> list:
    rows = [r for r in load_index(data_dir) if r["pool"] == pool and not r["corrupt"]]
    if not rows:
        raise ValueError(f"no usable {pool} episodes in {data_dir}/index.json")
    return rows


def teleop_split(data_dir: str, val_fraction: float = 0.1, seed: int = 42,
                 pool: str = "teleop") -> tuple[list, list]:
    """Episode-level train/val split of the demonstrations, stratified by task.

    Split on episodes and not on frames: two frames of one demonstration are near-duplicates,
    so a frame-level split would put the same scene on both sides.
    """
    rows = [r for r in load_index(data_dir) if r["pool"] == pool and not r["corrupt"]]
    if not rows:
        raise ValueError(f"no usable {pool} episodes in {data_dir}/index.json")
    rng = np.random.default_rng(seed)
    train, val = [], []
    for task in sorted({r["condition"] for r in rows}):
        task_rows = sorted([r for r in rows if r["condition"] == task], key=lambda r: r["key"])
        order = rng.permutation(len(task_rows))
        n_val = max(1, int(round(val_fraction * len(task_rows))))
        for rank, idx in enumerate(order):
            (val if rank < n_val else train).append(task_rows[idx])
    return train, val


def _build_trajectories(rows: List[dict], data_dir: str, frame_stride: int,
                        step_reward: float, discount: float,
                        num_terminal: int = DEFAULT_NUM_TERMINAL,
                        gripper_open_width: Optional[float] = None,
                        causal_gripper: bool = False) -> List[dict]:
    trajectories = []
    dropped_offsets = 0
    for row in rows:
        payload = np.load(Path(data_dir) / row["pool"] / f"{row['key']}.npz", allow_pickle=True)
        images, actions = payload["images"], payload["actions"]
        if len(images) != len(actions):
            raise ValueError(f"{row['key']}: {len(images)} frames vs {len(actions)} actions")
        if gripper_open_width is not None:
            # Before striding, on purpose -- see real_robot_actions.binarize_episode.
            actions = binarize(actions.astype(np.float32), gripper_open_width, causal_gripper)

        kept_here = 0
        for offset in range(frame_stride):
            idx = np.arange(offset, len(images), frame_stride)
            if len(idx) < num_terminal + 2:  # too short to carry a bootstrapped transition
                dropped_offsets += 1
                continue
            kept_here += 1
            rewards, td_mask = _episode_rewards(len(idx), step_reward, num_terminal)
            trajectories.append({
                "observation": {"image_primary": np.asarray(images[idx], dtype=np.bytes_)},
                "action": actions[idx].astype(np.float32),
                "task": {"language_instruction": np.array([row["instruction"]] * len(idx),
                                                          dtype=np.bytes_)},
                "reward": rewards,
                "td_mask": td_mask,
                "mc_return": _compute_mc_return(rewards, td_mask, discount),
                "task_feasible": np.ones(len(idx), dtype=bool),
                "task_id": np.array([row["condition"]] * len(idx), dtype=np.bytes_),
                "episode_id": np.full(len(idx), abs(hash(row["key"])) % (2**31), dtype=np.int32),
                "step_id": idx.astype(np.int32),
            })
        if kept_here == 0:
            raise ValueError(
                f"{row['key']}: {len(images)} frames yield no trajectory at stride "
                f"{frame_stride} with num_terminal={num_terminal}. Either the episode is "
                f"broken and the converter should mark it corrupt, or the stride is too coarse")
    if dropped_offsets:
        # Expected: the tail phase offsets of an episode whose length is not a multiple of the
        # stride are one frame shorter. Logged so a stride that is quietly discarding a large
        # share of the data cannot pass unnoticed.
        logging.info(
            f"real_robot: dropped {dropped_offsets} short phase offsets of "
            f"{len(rows) * frame_stride} at stride {frame_stride}")
    return trajectories


def make_real_robot_dataset(
    data_dir: str,
    split: str = "train",
    seed: int = 42,
    discount: float = 0.99,
    step_reward: float = -1.0,
    frame_stride: int = 20,
    num_terminal: int = DEFAULT_NUM_TERMINAL,
    val_fraction: float = 0.1,
    tasks: Optional[List[str]] = None,
    normalization: str = "bounds",
    force_recompute_statistics: bool = False,
    train_pool: str = "teleop",
    statistics_path: Optional[str] = None,
    causal_gripper: bool = False,
    val_pool: Optional[str] = None,
) -> dl.DLataset:
    """Demonstrations as a DLataset ready for apply_trajectory_transforms.

    Args:
        data_dir: the cache directory written by experiments/convert_real_robot.py
        split: "train" or "val" -- an episode-level split of the demonstrations
        step_reward: per-step cost. -1.0 makes Q a discounted step count. 0.0 removes it,
            which only carries signal when negative demonstrations are present; with none,
            every member fits Q = 0 and both the magnitude and the spread collapse.
        frame_stride: keep every k-th frame, emitting all k phase offsets (see module docstring)
        num_terminal: trailing frames given reward 0 and no bootstrap (see DEFAULT_NUM_TERMINAL)
        tasks: restrict to these task folders (None = all six)
        normalization: "bounds" reproduces the Bridge/Fractal action space the pretrained
            critic was fitted on. "none" trains on raw recorder units.
        force_recompute_statistics: recompute the cached percentile bounds instead of reusing
            the JSON beside the cache
        train_pool: the index ``pool`` whose episodes are training data ("teleop" for the lab's
            demonstrations, "finetune" for SAFE's successful rollouts)
        statistics_path: fixed octo statistics JSON; None = bounds of ``train_pool``
        causal_gripper: binarise the gripper per step (what an online monitor sees) instead of
            over the whole episode; see ``real_robot_actions.binarize_causal``
        val_pool: index pool used as validation as a whole (``holdout`` of
            ``split_vla_rollouts.py``); None = carve ``val_fraction`` out of ``train_pool``
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

    if val_pool is None:
        train_rows, val_rows = teleop_split(data_dir, val_fraction=val_fraction, seed=seed,
                                            pool=train_pool)
    else:
        train_rows, val_rows = pool_rows(data_dir, train_pool), pool_rows(data_dir, val_pool)
    rows = train_rows if split == "train" else val_rows
    if tasks is not None:
        rows = [r for r in rows if r["condition"] in tasks]
        if not rows:
            raise ValueError(f"no episodes left after filtering to tasks={tasks}")

    # Statistics first: they carry the open width the gripper binarisation needs, and they are
    # taken over every usable demonstration rather than over this split.
    normalization_type = resolve_normalization(normalization)
    stats = None
    if normalization_type is not None:
        stats = resolve_statistics(data_dir, frame_stride, statistics_path, train_pool,
                                   causal_gripper, force_recompute=force_recompute_statistics)
    else:
        logging.warning(
            "real_robot: action normalization is OFF -- the Bridge/Fractal checkpoint was "
            "fitted on actions in [-1, 1], so its first layer will be fed a different scale")

    trajectories = _build_trajectories(
        rows, data_dir, frame_stride, step_reward, discount, num_terminal=num_terminal,
        gripper_open_width=None if stats is None else open_width_of(stats),
        causal_gripper=causal_gripper)
    logging.info(
        f"real_robot {split}: {len(rows)} episodes -> {len(trajectories)} trajectories "
        f"(stride {frame_stride}), {sum(len(t['action']) for t in trajectories)} frames"
    )

    def gen():
        for traj in trajectories:
            yield traj

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "observation": {"image_primary": tf.TensorSpec(shape=(None,), dtype=tf.string)},
            "action": tf.TensorSpec(shape=(None, ACTION_DIM), dtype=tf.float32),
            "task": {"language_instruction": tf.TensorSpec(shape=(None,), dtype=tf.string)},
            "reward": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "td_mask": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "mc_return": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "task_feasible": tf.TensorSpec(shape=(None,), dtype=bool),
            "task_id": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "episode_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "step_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
    )
    dataset = dataset.shuffle(len(trajectories), seed=seed)
    dataset = _wrap_tf_dataset_as_dlimp(dataset)
    if stats is not None:
        dataset = normalize_dataset(dataset, stats, normalization_type)
    return dataset
