"""IVA (Instruct, Verify, Act) dataset loader for Q-ensemble training.

This module provides functions to load the IVA dataset in a format compatible
with Q-ensemble (SARSA ensemble) training.

IVA Dataset Structure:
- Directory: iva_dataset/{Train,Eval}/
- Files: iva_{split}_fp.json (FP injections) or iva_{split}_tp.json (TP only)
- Entry format: {id, image, conversations, task, episode, step, is_false_premise}
- Actions: 8D [J1, ..., J7, gripper]
- Language: Extracted via "The task is \"...\""
- Images: 128x128 PNG at {task}/episode{episode}/{step}.png

Data Loading Strategies (use_tp_only):
- False (default): Load _fp files
- True: Load _tp files (no FP), useful for train-clean/eval-OOD experiments

False Premise Strategies (fp_strategy):
- "copy_prev": FP steps reuse the previous step's action (default)
- "zero":      FP steps get a zero action
ONLY REACHES ANYTHING WHEN _fp FILES ARE LOADED, i.e. use_tp_only=False. _tp
files have an action on every step

Image Preprocessing:
- Images loaded as encoded PNG bytes during dataset creation (load_images_as_bytes=True)
- Source images are 128x128, upsampled to 256x256 during frame transforms
- Upsampling uses TensorFlow's bilinear interpolation via dl.transforms.resize_image
- This matches the pretrained ensemble's 256x256 input size
- Format compatible with apply_frame_transforms() (same as bridge/fractal tfrecords)

Goal Relabeling:
- Not used for Q-learning (goal_relabeling_strategy=None in config)
- Goal conditioning comes from language instructions only

Usage:
    # TP-only training (no FP), eval on FP for OOD detection
    train_data = make_iva_dataset("iva_dataset", split="train", use_tp_only=True)
    val_data = make_iva_dataset("iva_dataset", split="eval", use_tp_only=False)
"""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
from tqdm import tqdm


# Single source of truth for the IVA task names. Both splits contain all nine;
# `make_iva_dataset(tasks=None)` auto-detects from filenames, so use this only
# where an explicit, ordered list is needed (CLI defaults, table rows).
IVA_TASKS = [
    "close_jar",
    "meat_off_grill",
    "open_drawer",
    "push_buttons",
    "put_money_in_safe",
    "reach_and_drag",
    "slide_block_to_color_target",
    "sweep_to_dustpan_of_size",
    "turn_tap",
]

# Per-frame identity carried alongside the transition data so a scored trajectory
# can be joined back to its source JSON entry. Tiled to traj_len rather than kept
# as scalars, matching `task_feasible` here and `dataset_name` in
# octo/data/dataset.py -- every top-level leaf shares the same leading axis.
IVA_META_KEYS = ("task_id", "episode_id", "step_id")

# The instruction is quoted inside the human turn's prose. One definition, used
# both by the trajectory builder and by `instruction_inventory`.
_TASK_INSTRUCTION_RE = re.compile(r'The task is "(.*?)"')


# =============================================================================
# Helper Functions
# =============================================================================

def _wrap_tf_dataset_as_dlimp(tf_dataset: tf.data.Dataset) -> dl.DLataset:
    """Wrap tf.data.Dataset as dl.DLataset using dynamic class swapping.

    This mimics the internal _wrap logic from dlimp.dataset to enable
    .traj_map() and .frame_map() methods on the dataset.

    Args:
        tf_dataset: Standard TensorFlow dataset from from_generator()

    Returns:
        dl.DLataset with all dlimp methods available
    """
    if not isinstance(tf_dataset, dl.DLataset) and isinstance(tf_dataset, tf.data.Dataset):
        # Create new type inheriting from both DLataset and the tf dataset class
        tf_dataset.__class__ = type(
            "DLataset",
            (dl.DLataset, type(tf_dataset)),
            dl.DLataset.__dict__.copy()
        )
        tf_dataset.is_flattened = False
    return tf_dataset


# =============================================================================
# Q-Ensemble Compatible Loader
# =============================================================================

def make_iva_dataset(
    data_dir: str,
    split: str = "train",
    tasks: Optional[List[str]] = None,
    seed: int = 42,
    load_images_as_bytes: bool = True,
    discount: float = 0.99,
    use_tp_only: bool = False,
    fp_strategy: str = "copy_prev",
    step_reward: float = -1.0,
) -> dl.DLataset:
    """Load IVA dataset as DLataset compatible with Q-ensemble training.

    This returns trajectories that can be passed through apply_trajectory_transforms
    and apply_frame_transforms, just like the OXE datasets.

    Args:
        data_dir: Path to iva_dataset directory (containing Train/ and Eval/)
        split: "train" for training, "eval" for validation
        tasks: List of tasks to include (None = all 9 tasks)
        seed: Random seed for shuffling
        load_images_as_bytes: If True (default), read images from disk and store as
            encoded PNG bytes (compatible with apply_frame_transforms). If False,
            store file paths (requires custom image loading in pipeline).
        discount: Discount factor for MC return computation
        use_tp_only: If True, load _tp files (no false premises). Default False.
        fp_strategy: Which action accompanies a false-premise step. Reward is
            -1.0 either way; only the action differs. Inert unless _fp files are
            loaded (use_tp_only=False) -- see the module docstring.
            - "copy_prev": reuse the previous step's action (default)
            - "zero":      zero action
        step_reward: per-step reward on non-terminal true-premise steps.
            -1.0 (default) makes Q a discounted step count -- a clock.
            0.0 removes the clock: Q of a feasible episode is 0 everywhere and
            only the -1.0/mask=1 false-premise branch moves it. Ablation knob of the
            thesis step-reward comparison.

    Returns:
        dl.DLataset of trajectories with structure:
        {
            "observation": {"image_primary": tf.string},  # Encoded PNG bytes
            "action": tf.Tensor,                          # 8D actions
            "task": {"language_instruction": tf.string},  # Language (may vary per step)
            "reward": tf.Tensor,     # Reward based on fp_strategy
            "td_mask": tf.Tensor,    # 1 for valid steps
            "mc_return": tf.Tensor,  # Monte Carlo returns (gamma=discount)
            "task_feasible": tf.Tensor,  # True for feasible, False for false premise
        }
    """
    if fp_strategy not in ("copy_prev", "zero"):
        raise ValueError(f"fp_strategy must be 'copy_prev' or 'zero', got: {fp_strategy}")
    data_path = Path(data_dir) / split.capitalize()
    if not data_path.exists():
        raise FileNotFoundError(f"IVA data directory not found: {data_path}")

    # Determine file suffix based on use_tp_only
    file_suffix = "tp" if use_tp_only else "fp"
    logging.info(f"Loading IVA {split} split with use_tp_only={use_tp_only}, fp_strategy={fp_strategy}")

    # Load data
    data = []

    if split == "train":
        # Train: single consolidated file
        data_file = data_path / f"iva_train_{file_suffix}.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Train {file_suffix.upper()} file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} {file_suffix.upper()} entries from {data_file}")

    elif split == "eval":
        # Eval: per-task files
        if tasks is None:
            # Auto-detect all tasks from filenames
            # Files are named: iva_eval_{task}_{fp|tp}.json
            task_files = list(data_path.glob(f"iva_eval_*_{file_suffix}.json"))
            tasks = []
            for f in task_files:
                # Remove "iva_eval_" prefix and "_{fp|tp}" suffix
                task_name = f.stem.replace("iva_eval_", "", 1).replace(f"_{file_suffix}", "", 1)
                tasks.append(task_name)
            logging.info(f"Auto-detected eval tasks: {tasks}")

        for task in tasks:
            task_file = data_path / f"iva_eval_{task}_{file_suffix}.json"
            if not task_file.exists():
                logging.warning(f"Eval {file_suffix.upper()} file not found for task '{task}': {task_file}")
                continue

            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            data.extend(task_data)
            logging.info(f"Loaded {len(task_data)} {file_suffix.upper()} entries for task '{task}'")

    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'eval'")

    # Build trajectories from data
    trajectories = _build_trajectories(
        data,
        data_path,
        tasks,
        load_images_as_bytes,
        discount=discount,
        fp_strategy=fp_strategy,
        step_reward=step_reward,
    )
    logging.info(f"Created {len(trajectories)} trajectories from {split} split")

    # Convert to TensorFlow dataset
    def gen():
        for traj in trajectories:
            yield traj

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "observation": {
                "image_primary": tf.TensorSpec(shape=(None,), dtype=tf.string),
            },
            "action": tf.TensorSpec(shape=(None, 8), dtype=tf.float32),
            "task": {
                "language_instruction": tf.TensorSpec(shape=(None,), dtype=tf.string),
            },
            "reward": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "td_mask": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "mc_return": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "task_feasible": tf.TensorSpec(shape=(None,), dtype=bool),  # NEW
            # IVA_META_KEYS
            "task_id": tf.TensorSpec(shape=(None,), dtype=tf.string),
            "episode_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "step_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
    )

    # Shuffle
    dataset = dataset.shuffle(len(trajectories), seed=seed)

    # Wrap as DLataset for compatibility with .traj_map() and .frame_map()
    dataset = _wrap_tf_dataset_as_dlimp(dataset)

    return dataset


def _build_single_trajectory(
    args: Tuple[Tuple[str, int], List[dict], Path, bool, float, str]
) -> Optional[dict]:
    """Build a single trajectory from entries. Used for parallel processing.

    Args:
        args: Tuple of (key, entries, data_dir, load_images_as_bytes, discount, fp_strategy,
            step_reward)
            - key: (task, episode) tuple
            - entries: List of JSON entries for this trajectory
            - data_dir: Path to data directory
            - load_images_as_bytes: Whether to load images as bytes
            - discount: Discount factor for MC returns
            - fp_strategy: "copy_prev" or "zero"
    """
    ((task, episode), entries, data_dir, load_images_as_bytes, discount, fp_strategy,
     step_reward) = args

    # Sort entries by step
    entries = sorted(entries, key=lambda x: x["step"])

    # Parse each entry
    parsed_steps = []
    false_premise_indices = []
    languages = []

    for step_idx, entry in enumerate(entries):
        parsed = parse_iva_conversations(entry)
        parsed_steps.append(parsed)

        if parsed['is_false_premise']:
            false_premise_indices.append(step_idx)
        languages.append(parsed['task_name'])

    if len(parsed_steps) == 0:
        return None

    traj_len = len(parsed_steps)

    # Extract actions, images
    actions = np.array([p['action'] for p in parsed_steps], dtype=np.float32)
    image_paths = [str(data_dir / entry['image']) for entry in entries]

    # Load images as bytes if requested
    if load_images_as_bytes:
        image_bytes_list = []
        for img_path in image_paths:
            with open(img_path, 'rb') as f:
                image_bytes_list.append(f.read())
        image_data = np.array(image_bytes_list, dtype=np.bytes_)
    else:
        image_data = np.array(image_paths, dtype=np.bytes_)

    # Fill in steps that carry no action at all. After the parser reads the last
    # gpt turn, the only such steps are terminal refusals -- the episode ends at
    # the injection, so there is no next action. Three cases, in order:
    #   1. step has an action              -> use it (the common case)
    #   2. no action, a previous step exists -> fp_strategy decides
    #   3. no action, and it is step 0     -> nothing to copy from; drop the
    #                                         trajectory rather than invent one
    # Case 3 is a single-frame episode that is both first and last (1 of 225 eval
    # episodes). Under fp_strategy="zero" a zero action is the intent, so no drop.
    for i, parsed in enumerate(parsed_steps):
        if parsed['has_action']:
            continue
        if fp_strategy == "zero":
            actions[i] = np.zeros(8, dtype=np.float32)
        elif i > 0:
            # copy_prev: keep the action in-distribution. A zero action is one
            # the critic never saw in training, so a low Q would reflect an
            # unseen action rather than the language/image contradiction.
            actions[i] = actions[i - 1]
        else:
            logging.warning(
                f"dropping {task}/episode{episode}: step 0 is a false-premise refusal "
                f"with no action and no previous step to copy from (traj_len={traj_len})"
            )
            return None

    # Reward does NOT depend on fp_strategy: FP steps get -1.0 under both,
    # identical to a regular step. Only the action above differs.
    fp_reward = -1.0

    rewards = create_rewards_with_penalties(
        traj_len=traj_len,
        false_premise_indices=false_premise_indices,
        penalty=fp_reward,
        num_final_success=3,
        step_reward=step_reward,
    )

    # Compute td_mask
    last_step_is_false_premise = (traj_len - 1) in false_premise_indices
    if last_step_is_false_premise:
        td_mask = np.ones(traj_len, dtype=np.float32)
    else:
        num_terminal = min(3, traj_len)
        td_mask = np.concatenate([
            np.ones(traj_len - num_terminal, dtype=np.float32),
            np.zeros(num_terminal, dtype=np.float32)
        ])

    # Compute mc_return
    #TODO: Dont hardcode discount
    mc_return = _compute_mc_return(rewards, td_mask, discount=discount)

    # Create task_feasible array
    task_feasible = np.ones(traj_len, dtype=bool)
    for idx in false_premise_indices:
        if 0 <= idx < traj_len:
            task_feasible[idx] = False

    return {
        "observation": {"image_primary": image_data},
        "action": actions,
        "task": {"language_instruction": np.array(languages, dtype=np.bytes_)},
        "reward": rewards,
        "td_mask": td_mask,
        "mc_return": mc_return,
        "task_feasible": task_feasible,
        # IVA_META_KEYS -- identity only, never consumed by the agent. step_id is
        # the source JSON `step` field, so an episode truncated at a terminal
        # injection is visible as a short run and any gap in the source
        # enumeration is detectable downstream.
        # np.array(..., dtype=np.bytes_) sizes the itemsize from the content;
        # np.full with the same dtype would truncate to one byte.
        "task_id": np.array([task] * traj_len, dtype=np.bytes_),
        "episode_id": np.full(traj_len, episode, dtype=np.int32),
        "step_id": np.array([entry["step"] for entry in entries], dtype=np.int32),
    }


def _build_trajectories(
    data: List[dict],
    data_dir: Path,
    tasks: Optional[List[str]] = None,
    load_images_as_bytes: bool = True,
    num_workers: Optional[int] = None,
    discount: float = 0.99,
    fp_strategy: str = "copy_prev",
    step_reward: float = -1.0,
) -> List[dict]:
    """Build trajectories from data with configurable false premise handling.

    Uses parallel processing to speed up image loading (~10-20x faster).

    Args:
        data: List of JSON entries (may contain false premise injections)
        data_dir: Path to the data directory
        tasks: List of tasks to include (None = all)
        load_images_as_bytes: If True, read images from disk as encoded bytes
        num_workers: Number of parallel workers (default: 2x CPU cores for I/O-bound work)
        discount: Discount factor for MC return computation
        fp_strategy: "copy_prev" or "zero" for false premise action handling
        step_reward: per-step reward on true-premise steps (-1.0 = clock, 0.0 = no clock)

    Returns:
        List of trajectory dicts with task_feasible field
    """
    # Set default workers: 2x CPU cores (I/O-bound benefits from oversubscription)
    if num_workers is None:
        num_workers = min(32, (os.cpu_count() or 4) * 2)
    logging.info(f"num_workers: {num_workers}")
    # Group by (task, episode)
    groups = defaultdict(list)
    for entry in data:
        key = (entry["task"], entry["episode"])
        groups[key].append(entry)

    # Filter groups by task if specified
    filtered_groups = {k: v for k, v in groups.items() if tasks is None or k[0] in tasks}

    # sorted() gives a canonical (task, episode) order. This matters because
    # make_iva_dataset ends with a seeded .shuffle(): a seeded shuffle is a
    # deterministic permutation OF ITS INPUT, so the seed only means something if
    # the input order is fixed too.
    args_list = [
        (key, filtered_groups[key], data_dir, load_images_as_bytes, discount, fp_strategy,
         step_reward)
        for key in sorted(filtered_groups)
    ]

    # executor.map yields results in INPUT order, not completion order, so the
    # parallelism does not leak into the ordering.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = tqdm(
            executor.map(_build_single_trajectory, args_list),
            total=len(args_list),
            desc="Building trajectories",
        )
        # None = an episode _build_single_trajectory refused to build (see there).
        return [traj for traj in results if traj is not None]


def _compute_mc_return(rewards: np.ndarray, masks: np.ndarray, discount: float = 0.99) -> np.ndarray:
    """Compute Monte Carlo returns from rewards and masks.

    Computed backwards: mc_return[t] = reward[t] + discount * mc_return[t+1] * masks[t+1]

    This matches Octo's tf.scan implementation:
        mc_return = tf.scan(
            lambda prev_return, x: x[0] + discount * prev_return * x[1],
            [reward, td_mask],
            initializer=0.0,
            reverse=True
        )

    Args:
        rewards: (T,) array of rewards
        masks: (T,) array of masks (1 for valid, 0 for padded/terminal)
        discount: Discount factor (gamma)

    Returns:
        (T,) array of Monte Carlo returns
    """
    traj_len = len(rewards)
    mc_returns = np.zeros(traj_len, dtype=np.float32)

    # Compute returns backwards (matching tf.scan behavior)
    # At step t, we use masks[t] which corresponds to masks[t+1] in the forward view
    # because tf.scan processes elements and prev_return contains mc_return[t+1]
    running_return = 0.0
    for t in reversed(range(traj_len)):
        # masks[t] is applied to running_return (which is mc_return[t+1])
        mc_returns[t] = rewards[t] + discount * running_return * masks[t]
        running_return = mc_returns[t]

    return mc_returns


def _split_fp_json_paths(data_dir: str, split: str, tasks: Optional[List[str]] = None) -> List[Path]:
    """Raw `_fp` JSON file(s) for a split.

    Mirrors the file layout `make_iva_dataset` resolves inline (train = one
    consolidated file, eval = one file per task). Kept separate rather than
    factored out of `make_iva_dataset` so that reading instruction metadata
    cannot perturb the loader that training runs depend on.
    """
    data_path = Path(data_dir) / split.capitalize()
    if not data_path.exists():
        raise FileNotFoundError(f"IVA data directory not found: {data_path}")
    if split == "train":
        return [data_path / "iva_train_fp.json"]
    if split != "eval":
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'eval'")
    names = tasks if tasks is not None else IVA_TASKS
    paths = [data_path / f"iva_eval_{t}_fp.json" for t in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing IVA eval fp files: {missing}")
    return paths


def instruction_inventory(
    data_dir: str, split: str, tasks: Optional[List[str]] = None
) -> Tuple[Dict[str, "Counter"], Dict[Tuple[str, int], str]]:
    """Which instruction strings a split contains, and each episode's true one.

    Reads the raw `_fp` JSON rather than the trajectory builder: this is string
    metadata, and building trajectories would decode every PNG for it.

    A false-premise entry carries the WRONG instruction in `conversations[0]` and
    the corrected, true one in `conversations[2]` (the 4-turn shape documented in
    `parse_iva_conversations`). Terminal refusals have no correction turn, so an
    episode's true instruction is resolved from its other entries.

    Returns:
        counts: task -> Counter of true-premise instruction strings, i.e. how
            familiar each string is in this split.
        episode_true: (task, episode) -> the episode's single true instruction.
            Episodes whose entries carry no true instruction anywhere (the
            1-frame terminal-refusal episode in eval `open_drawer`) are absent.
    """
    counts: Dict[str, Counter] = defaultdict(Counter)
    per_episode: Dict[Tuple[str, int], Counter] = defaultdict(Counter)

    for path in _split_fp_json_paths(data_dir, split, tasks):
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            key = (entry["task"], entry["episode"])
            if entry.get("is_false_premise", False):
                # Only the correction turn states the true task here.
                if len(entry["conversations"]) >= 3:
                    per_episode[key][_instruction_of(entry, turn=2)] += 1
            else:
                true_instruction = _instruction_of(entry, turn=0)
                counts[entry["task"]][true_instruction] += 1
                per_episode[key][true_instruction] += 1

    episode_true = {}
    for key, variants in per_episode.items():
        if not variants:
            continue
        if len(variants) != 1:
            raise ValueError(
                f"episode {key} has {len(variants)} distinct true instructions: {dict(variants)}"
            )
        episode_true[key] = next(iter(variants))
    return dict(counts), episode_true


def _instruction_of(entry: dict, turn: int) -> str:
    """The instruction string quoted in one conversation turn."""
    value = entry["conversations"][turn]["value"]
    match = _TASK_INSTRUCTION_RE.search(value)
    if match is None:
        raise ValueError(
            f"no task instruction in turn {turn} of IVA entry id={entry['id']} "
            f"{entry['task']}/ep{entry['episode']}/step{entry['step']}: {value[:200]!r}"
        )
    return match.group(1)


def parse_iva_conversations(entry: dict) -> dict:
    """Parse IVA conversation entry into structured format.

    Extracts task name, action, step history, and false premise status
    from a single IVA JSON entry.

    Args:
        entry: IVA JSON entry with keys {id, image, conversations, task, episode, step}

    Returns:
        dict with keys:
            - task_name: str - The task instruction
            - action: np.ndarray (8D) - The robot action
            - step_history: list - Previous 5 steps joint positions (7D each)
            - is_false_premise: bool - Whether this is a false premise injection
            - has_action: bool - Whether any gpt turn provided an action

    Conversation shapes:
        true premise (2 turns):  [human(task), gpt(action)]
        false premise (4 turns): [human(WRONG task), gpt(refusal),
                                  human(correction, true task), gpt(action)]
        terminal refusal (2):    [human(WRONG task), gpt(refusal)]  -- no action

    task_name is always taken from conversations[0], i.e. the WRONG task on a
    false-premise step. That mismatch against the image is the false premise.
    """
    # Extract false premise status (sparse field - only present when True)
    is_false_premise = entry.get('is_false_premise', False)

    # Extract conversations
    conversations = entry.get('conversations', [])

    # Extract task name from human message. An unparseable prompt used to fall
    # back to "" and flow silently into the language encoder, producing a
    # meaningless Q value with no warning. 
    if not conversations:
        raise ValueError(f"IVA entry has no conversations: id={entry['id']} "
                         f"{entry['task']}/ep{entry['episode']}/step{entry['step']}")
    human_msg = conversations[0]['value']
    task_match = _TASK_INSTRUCTION_RE.search(human_msg)
    if task_match is None:
        raise ValueError(f"no task instruction in IVA entry: id={entry['id']} "
                         f"{entry['task']}/ep{entry['episode']}/step{entry['step']}: {human_msg[:200]!r}")
    task_name = task_match.group(1)

    # Extract step history (previous 5 steps)
    step_history = []
    history_match = re.search(r'previous five.*steps are \[\[(.*?)\]\]', human_msg)
    if history_match:
        # Parse nested arrays [[x,y,z,rx,ry,rz,gripper], ...]
        history_str = '[' + history_match.group(1) + ']'
        try:
            step_history = json.loads(history_str)
        except json.JSONDecodeError:
            step_history = []

    # The only entries genuinely without an action are terminal refusals, where
    # the episode ends at the injection (46 eval / 172 train).
    action = np.zeros(8, dtype=np.float32)
    has_action = False

    for turn in reversed(conversations):
        action_match = re.search(r'The next action step: \[([^\]]+)\]', turn['value'])
        if action_match is None:
            continue
        values = [float(x.strip()) for x in action_match.group(1).split(",")]
        if len(values) >= 8:
            action = np.array(values[:8], dtype=np.float32)
            has_action = True
        elif len(values) == 7:
            # 7D joint delta without gripper; IVA actions are 8D [J1-J7, gripper].
            action = np.array(values + [1.0], dtype=np.float32)
            has_action = True
        break

    return {
        'task_name': task_name,
        'action': action,
        'step_history': step_history,
        'is_false_premise': is_false_premise,
        'has_action': has_action,
    }


def create_rewards_with_penalties(
    traj_len: int,
    false_premise_indices: List[int],
    penalty: float = -1.0,
    num_final_success: int = 3,
    step_reward: float = -1.0,
) -> np.ndarray:
    """Create rewards with false premise handling.

    Args:
        traj_len: Length of trajectory
        false_premise_indices: List of step indices that are false premises
        penalty: Reward value for false premise steps (default -1.0, same as regular)
        num_final_success: Number of final steps with reward=0 for success

    Returns:
        rewards: Array of shape (traj_len,)

    Rules:
    1. False premise steps → penalty (default -1.0)
    2. If trajectory ends with true premise: last 3 true premise steps → 0.0 (success)
    3. If trajectory ends with false premise (cut off): NO success rewards
    4. All other steps → -1.0
    """
    # Initialize all rewards to the per-step reward (-1.0 = discounted clock).
    rewards = np.full(traj_len, step_reward, dtype=np.float32)

    # Apply penalty to false premise steps
    false_premise_set = set(false_premise_indices)
    for idx in false_premise_indices:
        if 0 <= idx < traj_len:
            rewards[idx] = penalty

    # Check if trajectory ends with false premise (cut off episode)
    last_step_is_false_premise = (traj_len - 1) in false_premise_set

    # Only give success rewards if trajectory completed successfully (last step is true premise)
    if not last_step_is_false_premise:
        # Find true premise indices (not false premises)
        true_premise_indices = [i for i in range(traj_len) if i not in false_premise_set]

        # Last 3 true premise steps get reward=0 (success signal)
        if len(true_premise_indices) > 0:
            num_success = min(num_final_success, len(true_premise_indices))
            last_success_indices = true_premise_indices[-num_success:]
            for idx in last_success_indices:
                rewards[idx] = 0.0

    return rewards


# =============================================================================
# Utilities
# =============================================================================

def get_iva_statistics(data_dir: str, split: str = "train") -> Dict:
    """Get statistics about the IVA dataset (FP-only).

    Args:
        data_dir: Path to iva_dataset directory
        split: "train" or "eval"

    Returns:
        Dict with keys: num_trajectories, num_steps, num_tasks, task_names,
                       num_false_premise, trajectory_lengths
    """
    data_path = Path(data_dir) / split.capitalize()
    if not data_path.exists():
        raise FileNotFoundError(f"IVA data directory not found: {data_path}")

    # Load FP-only data
    fp_data = []

    if split == "train":
        # Train: single consolidated FP file
        fp_file = data_path / "iva_train_fp.json"
        if fp_file.exists():
            fp_data = json.load(open(fp_file, "r", encoding="utf-8"))

    elif split == "eval":
        # Eval: per-task FP files
        task_files = list(data_path.glob("iva_eval_*_fp.json"))

        for fp_file in task_files:
            # Extract task name from filename
            task_name = fp_file.stem.replace("iva_eval_", "", 1).replace("_fp", "", 1)
            if fp_file.exists():
                fp_data.extend(json.load(open(fp_file, "r", encoding="utf-8")))

    # Group by episode
    episodes = set((e["task"], e["episode"]) for e in fp_data)

    # Get task names
    tasks = set(e["task"] for e in fp_data)

    # Count false premise steps
    num_false_premise = sum(1 for e in fp_data if e.get('is_false_premise', False))

    # Trajectory lengths
    fp_groups = defaultdict(list)
    for entry in fp_data:
        key = (entry["task"], entry["episode"])
        fp_groups[key].append(entry)

    traj_lengths = [len(steps) for steps in fp_groups.values()]

    return {
        "num_trajectories": len(episodes),
        "num_steps": len(fp_data),
        "num_false_premise_steps": num_false_premise,
        "num_tasks": len(tasks),
        "task_names": sorted(list(tasks)),
        "trajectory_length_mean": np.mean(traj_lengths) if traj_lengths else 0,
        "trajectory_length_std": np.std(traj_lengths) if traj_lengths else 0,
    }


def visualize_trajectory(traj: dict, max_frames: int = 20, show: bool = True):
    """Visualize a trajectory for debugging/inspection.

    Loads images and displays key information about the trajectory including
    actions, rewards, language instructions, and false premise injections.

    Frames are sampled equidistantly across the trajectory to show progression.

    Args:
        traj: Trajectory dict from make_iva_dataset() with keys:
              observation, action, task, reward, td_mask, mc_return, task_feasible
        max_frames: Maximum number of frames to visualize (default 20)
        show: Whether to display the plot (default True)

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        raise ImportError("matplotlib required for visualization. Install with: pip install matplotlib")

    # Decode trajectory data
    traj_len = len(traj['action'])
    actions = traj['action'].numpy() if hasattr(traj['action'], 'numpy') else traj['action']
    rewards = traj['reward'].numpy() if hasattr(traj['reward'], 'numpy') else traj['reward']
    td_mask = traj['td_mask'].numpy() if hasattr(traj['td_mask'], 'numpy') else traj['td_mask']
    task_feasible = traj['task_feasible'].numpy() if hasattr(traj['task_feasible'], 'numpy') else traj['task_feasible']

    # Decode image paths and language
    image_paths = traj['observation']['image_primary'].numpy()
    image_paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in image_paths]

    languages = traj['task']['language_instruction'].numpy()
    languages = [l.decode('utf-8') if isinstance(l, bytes) else l for l in languages]

    # Sample equidistant frames across trajectory
    if traj_len <= max_frames:
        # Show all frames if trajectory is short enough
        frame_indices = list(range(traj_len))
    else:
        # Sample equidistantly
        frame_indices = np.linspace(0, traj_len - 1, max_frames, dtype=int)

    viz_len = len(frame_indices)

    # Calculate grid layout (prefer roughly square grids)
    n_cols = int(np.ceil(np.sqrt(viz_len)))
    n_rows = int(np.ceil(viz_len / n_cols))

    # Create figure with grid layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    fig.suptitle(f'Trajectory Visualization ({traj_len} steps total, showing {viz_len} frames)', fontsize=14, fontweight='bold')

    # Find unique language instructions (to detect false premises)
    unique_langs = set(languages)
    has_false_premise = len(unique_langs) > 1 or not all(task_feasible)
    base_lang = languages[0] if languages else ""

    # Plot each sampled frame
    for plot_idx, step_idx in enumerate(frame_indices):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]

        # Load image
        try:
            from PIL import Image
            img = Image.open(image_paths[step_idx])
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"IMG ERR\n{image_paths[step_idx].split('/')[-1][:15]}",
                   ha='center', va='center', fontsize=8, color='red')

        # Step info (use actual step index)
        step_text = f"Step {step_idx}"
        if not task_feasible[step_idx]:
            step_text += "\n⚠ FALSE PREMISE"
            ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=3))
        elif step_idx >= traj_len - 3:
            step_text += "\n✓ SUCCESS"
            ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='green', linewidth=2))

        # Action info
        action_str = f"Action: [{actions[step_idx][0]:.2f}, ...,{actions[step_idx][-1]:.2f}]"
        reward_str = f"Reward: {rewards[step_idx]:+.1f}"

        # Language (truncated)
        lang = languages[step_idx] if step_idx < len(languages) else base_lang
        if len(lang) > 25:
            lang = lang[:22] + "..."
        lang_str = f'"{lang}"'

        # Combine text
        info_text = f"{step_text}\n{reward_str}\n{lang_str}"
        ax.set_title(info_text, fontsize=7, fontweight='bold' if not task_feasible[step_idx] else 'normal')

        ax.axis('off')

    # Hide empty subplots
    for i in range(viz_len, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
        axes[row, col].set_visible(False)

    plt.tight_layout()

    # Print trajectory summary
    print(f"\n{'='*60}")
    print(f"TRAJECTORY SUMMARY ({traj_len} steps)")
    print(f"{'='*60}")
    print(f"Unique instructions: {len(unique_langs)}")
    print(f"False premise steps: {sum(~task_feasible)} / {traj_len}")
    print(f"Rewards: min={rewards.min():+.1f}, max={rewards.max():+.1f}")
    print(f"Action shape: {actions.shape}")
    print(f"\nLanguage instructions:")
    for i, lang in enumerate(sorted(unique_langs)):
        marker = "✓" if lang == base_lang else "⚠"
        print(f"  {marker} \"{lang}\"")

    if has_false_premise:
        fp_steps = np.where(~task_feasible)[0]
        print(f"\nFalse premise steps (first 10): {fp_steps[:10]}")

    print(f"\nLast 3 true premise rewards: {rewards[task_feasible][-3:]}")
    print(f"Expected: [0. 0. 0.]")
    print(f"{'='*60}\n")

    if show:
        plt.show()

    return fig
