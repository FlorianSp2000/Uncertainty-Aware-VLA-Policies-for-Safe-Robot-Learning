# Config adapted from V-GPS (https://github.com/nakamotoo/V-GPS) experiments/configs/train_config.py.
"""Configuration for IVA ensemble training.

This config extends train_ensemble_config.py with IVA-specific overrides.

Modes:
- ensemble_sarsa: Train from scratch on IVA
- ensemble_sarsa_finetune: Finetune from pretrained bridge/fractal ensemble

False Premise Strategies (iva_fp_strategy):
- "copy_prev": FP steps reuse the previous step's action (default)
- "zero":      FP steps get a zero action
Reward is -1.0 under both -- the choice affects the action only. Inert unless
_fp files are loaded (iva_use_tp_only=False).

Data Loading (iva_use_tp_only):
- False: Load _fp files (contains ~84% TP + sparse FP injections) (default)
- True: Load _tp files (no FP), eval on _fp for OOD detection

Note: Finetune mode uses lower LR (1e-4). Consider increasing to 3e-4 if
training on cleaner data (TP-only) or with different reward structure.
"""
import importlib.util
import os
from copy import deepcopy

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

# Load base ensemble config (ml_collections doesn't support relative imports)
_spec = importlib.util.spec_from_file_location(
    "train_ensemble_config",
    os.path.join(os.path.dirname(__file__), "train_ensemble_config.py")
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
get_ensemble_config = _module.get_config


def get_config(config_string):
    """IVA ensemble configuration extending base ensemble config.

    Imports base config from train_ensemble_config.py and adds IVA-specific
    settings like data_dir and tasks.

    Note: The IVA dataloader automatically merges TP and FP files and detects
    false premise steps by comparing instructions step-by-step. No separate
    negative demo configuration needed.
    """
    # Get base config from ensemble (includes all agent, encoder, text processing settings)
    base_cfg = deepcopy(get_ensemble_config("ensemble_sarsa"))

    # IVA-specific overrides
    iva_overrides = dict(
        iva_data_dir=placeholder(str),  # Path to iva_dataset directory
        iva_tasks=None,  # None = all 9 tasks, or list like ["turn_tap", "close_jar"]
        resume_path=placeholder(str),  # Path to checkpoint (auto-detects 7D vs 8D actions)
        # Data loading strategy
        iva_use_tp_only=False,  # True: train on _tp (no FP), eval on _fp
        # False premise handling (only relevant when loading _fp files)
        iva_fp_strategy="copy_prev",  # action on FP steps: "copy_prev" | "zero".
                                      # Reward is -1.0 either way.
        # V2 negative demo swapping (bridge/fractal style)
        create_negative_demos=False,  # Enable V2 per-batch random prompt swapping
        negative_demo_ratio=0.10,     # 10% of trajectories get random wrong prompt
        # Per-step reward on true-premise steps. -1.0 = the SARSA clock (default,
        # Q = discounted steps-to-termination). 0.0 = clock removed: a feasible
        # episode has Q=0 everywhere, so only the false-premise / negative-demo
        # branch (reward -1, td_mask 1) moves Q. Ablation for the disagreement-floor
        # hypothesis.
        iva_step_reward=-1.0,
    )

    # Trajectory and frame transform kwargs
    traj_transform_kwargs = dict(
        window_size=1,
        action_horizon=1,
        goal_relabeling_strategy=None,  # Not needed for Q-learning/SARSA
        subsample_length=250,  # Random sample if traj > 250 steps
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=0.5,
        ),
    )

    frame_transform_kwargs = dict(
        resize_size={"primary": (256, 256)},  # Key must match "image_primary"
        image_dropout_prob=0.0,
        num_parallel_calls=200,
    )

    # Apply IVA-specific overrides
    base_cfg.update(iva_overrides)
    base_cfg.update(traj_transform_kwargs=traj_transform_kwargs)
    base_cfg.update(frame_transform_kwargs=frame_transform_kwargs)

    # Handle finetune mode
    if config_string == "ensemble_sarsa_finetune":
        # Finetune: lower learning rate, fewer steps
        base_cfg.agent_kwargs.learning_rate = 1e-4
        base_cfg.agent_kwargs.warmup_steps = 500
        base_cfg.num_steps = int(2e5)

    return base_cfg
