"""Read the real-robot dataset of preliminary experiments (not part of the thesis) from the unpacked directory tree.

Imported by convert_real_robot.py; not used for the SAFE / LIBERO rollouts.

Input is ``datasets/real_robot/raw_data_recomposed`` -- the nine delivered archives merged into
one tree (420 files, verified against the archives).

Two record formats live in the tree, both pickles despite the ``.npy`` suffix on the teleop
files (the dataset's own readme says so):

* teleop  -- ``[(RobotState, RobotAction), ...]`` at 20 Hz, one entry per control step
* rollout -- ``[(obs_dict, action_chunk[, task]), ...]``, one entry per GR00T policy call

Only ``ood_trajectories_normal`` carries ``descriptions.txt``; the rollouts hold their prompt
per frame inside each record, because it can change mid-episode.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_DIM = 7  # x, y, z, rx, ry, rz, gripper

TELEOP_DIR = "ood_trajectories_normal"
SCENE_DIR = "ood_trajectories"
ROLLOUT_DIRS = ("ood_trajectories_normal_rollout", "ood_trajectories_exchange_rollout",
                "ood_trajectories_missing_rollout", "ood_trajectories_half_missing_rollout")
INFEASIBLE = ("missing", "half_missing")


@dataclass
class RobotState:
    time_stamp: float = 0.0
    q: np.ndarray = None
    dq: np.ndarray = None
    eef_pose: np.ndarray = None
    gripper_width: float = 0.0
    wrist_camera_rgb_image: np.ndarray = None
    wrist_camera_depth_image: np.ndarray = None
    external_camera_rgb_image: np.ndarray = None
    external_camera_depth_image: np.ndarray = None


@dataclass
class RobotAction:
    time_stamp: float = 0.0
    eef_action: np.ndarray = None
    gripper_action: np.ndarray = None


class _Unpickler(pickle.Unpickler):
    """Rebind the recorder's classes, and bridge the numpy 2 -> numpy 1 module rename.

    Episodes were pickled where numpy >= 2 writes array reconstructors under ``numpy._core``;
    the training container pins numpy 1.24, where they live under ``numpy.core``.
    """

    def find_class(self, module, name):
        if name in ("RobotState", "RobotAction"):
            return globals()[name]
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def index(root: str | Path) -> list[dict]:
    """One row per episode: where it is and what the path says it is.

    Labels here are what the *directory layout* asserts. Whether the scene agrees -- whether a
    `missing` episode really lacks the object its prompt names -- is not recorded anywhere in
    the dataset and is not decided here.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in (".npy", ".pkl"):
            continue
        rel = p.relative_to(root)
        head = rel.parts[0]
        if head == TELEOP_DIR:
            rows.append(dict(pool="teleop", condition=rel.parts[1], path=p, rel=str(rel),
                             success=True, feasible=True))
        elif head == SCENE_DIR:
            rows.append(dict(pool="scene", condition=rel.parts[1], path=p, rel=str(rel),
                             success=None, feasible=False))
        elif head in ROLLOUT_DIRS:
            cond = head.replace("ood_trajectories_", "").replace("_rollout", "")
            rows.append(dict(pool="rollout", condition=cond, path=p, rel=str(rel),
                             success=rel.parts[-2] == "1", feasible=cond not in INFEASIBLE))
    if not rows:
        raise ValueError(f"no episode files under {root}")
    return rows


def instruction_of(root: str | Path, task: str) -> str:
    """The task's prompt, from its own descriptions.txt (teleop tasks only)."""
    return (Path(root) / TELEOP_DIR / task / "descriptions.txt").read_text().strip()


def load(path: str | Path, with_wrist: bool = True) -> dict:
    """One episode, in whichever of the two formats it is stored.

    Returns a dict with a common core -- ``kind``, ``external``, ``actions`` (T, 7),
    ``prompts`` -- plus the fields only one format has. ``with_wrist=False`` leaves the second
    camera unstacked, which halves peak memory for callers that only train on one view.
    """
    path = Path(path)
    with open(path, "rb") as f:
        steps = _Unpickler(f).load()
    if not steps:
        raise ValueError(f"{path} unpickles to an empty sequence")

    if isinstance(steps[0][0], RobotState):
        eef = np.stack([s[1].eef_action for s in steps])
        grip = np.stack([np.atleast_1d(s[1].gripper_action)[0] for s in steps])[:, None]
        return dict(
            kind="teleop", n=len(steps),
            external=np.stack([s[0].external_camera_rgb_image for s in steps]),
            wrist=np.stack([s[0].wrist_camera_rgb_image for s in steps]) if with_wrist else None,
            actions=np.concatenate([eef, grip], axis=1).astype(np.float32),
            q=np.stack([s[0].q for s in steps]),
            gripper_width=np.array([float(s[0].gripper_width) for s in steps]),
            timestamps=np.array([s[0].time_stamp for s in steps]),
            prompts=None,
        )

    obs = [s[0] for s in steps]
    chunks = np.stack([np.asarray(s[1]) for s in steps])  # (T, horizon, 10)
    return dict(
        kind="rollout", n=len(steps),
        external=np.stack([o["observation.images.external"] for o in obs]),
        wrist=np.stack([o["observation.images.wrist"] for o in obs]) if with_wrist else None,
        actions=chunks[:, 0, :ACTION_DIM].astype(np.float32),  # executed at this observation
        chunks=chunks,
        state=np.stack([o["observation.state"] for o in obs]),
        prompts=[o["task"] for o in obs],
        timestamps=None,
    )


def segments(prompts: list[str]) -> list[tuple[int, int, str]]:
    """(start, stop, prompt) runs of constant prompt within one rollout."""
    sw = [i for i in range(1, len(prompts)) if prompts[i] != prompts[i - 1]]
    b = [0, *sw, len(prompts)]
    return [(a, e, prompts[a]) for a, e in zip(b[:-1], b[1:])]


def short(prompt: str) -> str:
    """``Pick up the green cup and place it on the blue circle.`` -> ``green_cup_to_blue``."""
    m = re.search(r"the (\w+) cup and place it on the (\w+) circle", prompt)
    return f"{m.group(1)}_cup_to_{m.group(2)}" if m else prompt
