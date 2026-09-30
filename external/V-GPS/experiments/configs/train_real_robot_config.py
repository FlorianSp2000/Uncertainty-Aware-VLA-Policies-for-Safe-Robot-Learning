# Config adapted from V-GPS (https://github.com/nakamotoo/V-GPS) experiments/configs/train_config.py.
"""Configuration for ensemble finetuning on rollout caches (thesis: SAFE WidowX / LIBERO, `real_robot_train_pool="finetune"`).

(The loader also supports a real-robot demonstration dataset from preliminary experiments that are not part of the thesis.)

Extends train_ensemble_config.py, as train_iva_config.py does.

Modes:
- ensemble_sarsa:          train from scratch on the demonstrations
- ensemble_sarsa_finetune: finetune from the Bridge/Fractal ensemble (lower LR, fewer steps)

Every experimental knob is overridden from slurm/train-real-robot-ensemble.sbatch;
nothing here is meant to be edited per run.
"""
import importlib.util
import os
from copy import deepcopy

from ml_collections.config_dict import placeholder

_spec = importlib.util.spec_from_file_location(
    "train_ensemble_config",
    os.path.join(os.path.dirname(__file__), "train_ensemble_config.py"),
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
get_ensemble_config = _module.get_config


# Per-dataset differences of the VLA-rollout presets.
VLA_ROLLOUT_PRESETS = {
    # OpenVLA acts in Bridge units (it un-normalises with the Bridge statistics), so the critic
    # keeps the Bridge action mapping; 50 steps @ 5 Hz, last 3 frames terminal (Bridge rule).
    "vla_safe_widowx": dict(
        real_robot_num_terminal=3,
        real_robot_statistics_path=(
            "/V-GPS/datasets/open_x/bridge_dataset/1.0.0/dataset_statistics_"
            "4bf6112c156ef12280f9635f798ee6f7f5ae3160c09ac2bbe2150843f87d5d13.json"),
    ),
    # pi0-FAST LIBERO actions are in LIBERO units: bounds from the finetune successes. One
    # frame per policy call; the episode ends on success, so 1 terminal frame.
    "vla_libero_pi0fast": dict(
        real_robot_num_terminal=1,
        real_robot_statistics_path=None,
    ),
}


def get_config(config_string):
    base_cfg = deepcopy(get_ensemble_config("ensemble_sarsa"))

    base_cfg.update(dict(
        real_robot_data_dir=placeholder(str),  # cache from experiments/convert_real_robot.py
        real_robot_tasks=None,                 # None = all six cup/circle combinations
        resume_path=placeholder(str),
        # Per-step cost. -1.0 makes Q a discounted count of steps to termination. 0.0 removes
        # it, which is only meaningful when negative demonstrations supply the other end of
        # the scale -- with none, every member fits Q = 0 and the spread collapses too.
        real_robot_step_reward=-1.0,
        # Keep every k-th frame of a 20 Hz demonstration, emitting all k phase offsets. The
        # rollouts we score store one observation per policy call = one per 20 control steps,
        # so k = 20 puts training and evaluation on the same timestep.
        real_robot_frame_stride=20,
        # Trailing frames scored as success. 3 at 20 Hz is a fraction of a second; after
        # striding by 20 one frame already exceeds that, so 1 keeps the same intent.
        real_robot_num_terminal=1,
        real_robot_val_fraction=0.1,
        # The action space the pretrained critic was fitted on: per-dimension p001/p999 bounds
        # mapped to [-1, 1], gripper binarised first. Same NormalizationType Bridge/Fractal
        # passes at data_config.py:145, applied by the same octo functions -- the real-robot
        # path just does not run through make_dataset_from_rlds, which is the only place that
        # calls them. Set to "none" to train on raw recorder units.
        real_robot_normalization="bounds",   # "bounds" | "normal" | "none"
        # Recompute the cached percentile bounds instead of reusing the JSON beside the cache.
        real_robot_force_recompute_statistics=False,
        # Index pool whose episodes are training data: "teleop" = the lab's demonstrations,
        # "finetune" = the successful SAFE rollouts set aside by convert_safe_rollouts.py.
        real_robot_train_pool="teleop",
        # Fixed octo statistics JSON instead of the train pool's own bounds. For a policy that
        # already acts in Bridge units (OpenVLA on the WidowX) this is the Bridge file the
        # pretrained critic was normalised with.
        real_robot_statistics_path=placeholder(str),
        # Restart the step counter when loading the Bridge/Fractal weights. Both action spaces
        # are 7-D, so the action-dim heuristic would read the checkpoint as the same run and
        # start counting at its 200000, training for zero steps.
        real_robot_reset_step=True,
        # Gripper binarisation per step instead of over the whole episode. The episode-wide
        # octo rule resolves an intermediate command by the state reached LATER, which an
        # online monitor cannot know; identical on two-valued grippers (SAFE-WidowX), not on
        # pi0-FAST's LIBERO commands. Training and evaluation both read this one field.
        real_robot_causal_gripper=False,
        # Validation episodes: None = carve real_robot_val_fraction out of the train pool;
        # a pool name = that whole pool (``holdout`` of experiments/split_vla_rollouts.py).
        real_robot_val_pool=None,
        num_val_batches=6,     # batches of batch_size/ensemble_size frames per validation pass
        plot_interval=2000,    # value-function plots; copies the ensemble to the host
        # Before training, compare one packed update (several members per GPU) against the
        # per-member update; fail if the relative difference of the update exceeds the tol.
        check_packing=True,
        check_packing_tol=1e-2,
        # No negative demonstrations: in a normal scene both cups and all three circles are
        # present, so a swapped instruction stays satisfiable and would train the critic to
        # reject feasible tasks.
        create_negative_demos=False,
        negative_demo_ratio=0.10,
    ))

    base_cfg.update(traj_transform_kwargs=dict(
        window_size=1,
        action_horizon=1,
        goal_relabeling_strategy=None,
        # Absent on purpose: subsample_length gathers with a random shuffle before the SARSA
        # next-state is attached, which scrambles (s', a') whenever it fires. Striding in the
        # loader already bounds episode length to ~11 frames.
        task_augment_strategy=None,
    ))
    base_cfg.update(frame_transform_kwargs=dict(
        resize_size={"primary": (256, 256)},  # source frames are 224x224
        image_dropout_prob=0.0,
        num_parallel_calls=200,
    ))

    if config_string in ("ensemble_sarsa_finetune",) + tuple(VLA_ROLLOUT_PRESETS):
        base_cfg.agent_kwargs.learning_rate = 1e-4
        base_cfg.agent_kwargs.warmup_steps = 500
        base_cfg.num_steps = int(5e4)

    # SAFE-protocol rollout experiment (thesis): every knob of a VLA-rollout finetune, so
    # the sbatch passes only the preset, the split directory and the run name.
    if config_string in VLA_ROLLOUT_PRESETS:
        base_cfg.update(dict(
            real_robot_train_pool="finetune",
            real_robot_val_pool="holdout",
            real_robot_frame_stride=1,      # rollout caches hold one frame per policy step
            real_robot_causal_gripper=True,
            real_robot_reset_step=True,     # step counter from 0; Bridge optimizer state kept
            num_steps=50_000,
            save_interval=5_000,
            eval_interval=500,
            log_interval=250,
            plot_interval=5_000,
            num_val_batches=16,
        ))
        base_cfg.update(VLA_ROLLOUT_PRESETS[config_string])

    return base_cfg
