#!/bin/bash
#SBATCH --job-name=real_robot_ensemble
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:L40S:8
#SBATCH --partition=L40Sday

# Finetune the Bridge/Fractal SARSA ensemble on the successful rollouts of a rollout cache
# (thesis: SAFE WidowX / LIBERO, written by slurm/convert-{safe,libero}-rollouts.sbatch).
# Both action spaces are 7-D, so the action-dim auto-detection should log 7 -> 7.
# No negative demonstrations; the per-step cost stays at -1.0 (with neither, every member fits Q = 0).
#
# Preflight (a few hundred steps, verifies shapes/restore/loss before spending GPU hours):
#   sbatch --export=ALL,DATA_DIR=...,RESUME_PATH=...,NUM_STEPS=300,SAVE_INTERVAL=200,EVAL_INTERVAL=100 \
#          slurm/train-real-robot-ensemble.sbatch

set -e
set -o allexport; source .env; set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)

echo "SLURM Job ID: $SLURM_JOB_ID"
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi

# ---------------------------------------------------------------------------
# Configuration -- every experimental knob lives here
# ---------------------------------------------------------------------------
DATA_DIR=${DATA_DIR:?set DATA_DIR to the rollout cache, e.g. datasets/safe_widowx/processed_ft100_cal72_seed0}
RESUME_PATH=${RESUME_PATH:?set RESUME_PATH=results/VGPS_ensemble/<Bridge/Fractal SayNo run>}

BATCH_SIZE=${BATCH_SIZE:-1024}
ENSEMBLE_SIZE=${ENSEMBLE_SIZE:-8}
SEED=${SEED:-44}
NUM_STEPS=${NUM_STEPS:-50000}
EVAL_INTERVAL=${EVAL_INTERVAL:-2000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}

# Per-step cost on demonstration steps. -1.0 = Q is a discounted step count.
STEP_REWARD=${STEP_REWARD:--1.0}
# Keep every k-th frame of the 20 Hz recordings, all k phase offsets emitted. k=20 puts
# training on the rollouts' own observation rate (one per policy call).
FRAME_STRIDE=${FRAME_STRIDE:-20}
NUM_TERMINAL=${NUM_TERMINAL:-1}
VAL_FRACTION=${VAL_FRACTION:-0.1}

# Action space. BOUNDS reproduces the one the Bridge/Fractal checkpoint was fitted on:
# per-dimension p001/p999 mapped to [-1, 1], gripper binarised first. "none" trains on raw
# recorder units, which is what the 2026-09 run did before this was fixed.
NORMALIZATION=${NORMALIZATION:-bounds}
# Recompute the cached percentile bounds rather than reusing the JSON beside the cache.
FORCE_RECOMPUTE_STATISTICS=${FORCE_RECOMPUTE_STATISTICS:-False}

# Which index pool is training data, and whether actions use a fixed statistics file instead
# of that pool's own bounds. SAFE OpenVLA-WidowX rollouts (thesis):
#   DATA_DIR=datasets/safe_widowx/processed_ft100_cal72_seed0 TRAIN_POOL=finetune
#   STATISTICS_PATH=datasets/open_x/bridge_dataset/1.0.0/dataset_statistics_4bf6112c156ef12280f9635f798ee6f7f5ae3160c09ac2bbe2150843f87d5d13.json
#   FRAME_STRIDE=1 NUM_TERMINAL=3 PROJECT=VLA_rollouts EXP_PREFIX=SAFE_WIDOWX
# True = restart the step counter at 0 on the Bridge/Fractal weights (finetune). False = continue
# a finetune run from its latest checkpoint (RESUME_PATH = that run), optimizer state included.
RESET_STEP=${RESET_STEP:-True}
TRAIN_POOL=${TRAIN_POOL:-teleop}
STATISTICS_PATH=${STATISTICS_PATH:-}
PROJECT=${PROJECT:-REAL_ensemble}
EXP_PREFIX=${EXP_PREFIX:-REAL}

# Arm 2 knob, off for arm 1 (see the header).
CREATE_NEGATIVE_DEMOS=${CREATE_NEGATIVE_DEMOS:-false}
NEGATIVE_DEMO_RATIO=${NEGATIVE_DEMO_RATIO:-0.10}

ALGORITHM="ensemble_sarsa_finetune"
SAVE_DIR_ABSOLUTE="/V-GPS/results/"

mkdir -p ${WORK_DIR}/external/V-GPS/results ${WORK_DIR}/external/V-GPS/wandb ${WORK_DIR}/logs

EXP_NAME="${EXP_PREFIX}_${ALGORITHM}_ens${ENSEMBLE_SIZE}_b${BATCH_SIZE}s${SEED}_stride${FRAME_STRIDE}_r${STEP_REWARD}_norm${NORMALIZATION}"
if [ "$CREATE_NEGATIVE_DEMOS" = true ]; then
    EXP_NAME="${EXP_NAME}_negdemo${NEGATIVE_DEMO_RATIO}"
fi

if [ "$TRAIN_POOL" != teleop ]; then EXP_NAME="${EXP_NAME}_pool${TRAIN_POOL}"; fi
if [ -n "$STATISTICS_PATH" ]; then EXP_NAME="${EXP_NAME}_fixedstats"; fi
echo "Experiment: ${EXP_NAME}"
echo "steps=${NUM_STEPS} stride=${FRAME_STRIDE} step_reward=${STEP_REWARD} negdemos=${CREATE_NEGATIVE_DEMOS} norm=${NORMALIZATION}"

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets:/V-GPS/datasets \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/train_real_robot_ensemble.py \
    --config /V-GPS/experiments/configs/train_real_robot_config.py:${ALGORITHM} \
    --name "${EXP_NAME}" \
    --project ${PROJECT} \
    --config.ensemble_size ${ENSEMBLE_SIZE} \
    --config.seed ${SEED} \
    --config.num_steps ${NUM_STEPS} \
    --config.eval_interval ${EVAL_INTERVAL} \
    --config.save_interval ${SAVE_INTERVAL} \
    --config.save_dir ${SAVE_DIR_ABSOLUTE} \
    --config.batch_size ${BATCH_SIZE} \
    --config.real_robot_data_dir /V-GPS/${DATA_DIR} \
    --config.real_robot_step_reward ${STEP_REWARD} \
    --config.real_robot_frame_stride ${FRAME_STRIDE} \
    --config.real_robot_num_terminal ${NUM_TERMINAL} \
    --config.real_robot_val_fraction ${VAL_FRACTION} \
    --config.real_robot_normalization ${NORMALIZATION} \
    --config.real_robot_force_recompute_statistics ${FORCE_RECOMPUTE_STATISTICS} \
    --config.real_robot_train_pool ${TRAIN_POOL} \
    --config.real_robot_reset_step=${RESET_STEP} \
    ${STATISTICS_PATH:+--config.real_robot_statistics_path /V-GPS/${STATISTICS_PATH}} \
    $([ "$CREATE_NEGATIVE_DEMOS" = true ] && echo "--config.create_negative_demos True --config.negative_demo_ratio ${NEGATIVE_DEMO_RATIO}") \
    ${RESUME_PATH:+--config.resume_path /V-GPS/${RESUME_PATH}}
