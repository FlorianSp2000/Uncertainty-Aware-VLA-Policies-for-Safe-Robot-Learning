#!/bin/bash
#SBATCH --job-name=eval_real_robot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=L40Sday

# Score the episodes of a rollout cache with a finetuned ensemble and dump per-episode Q tensors
# (thesis: slurm/eval-vla-rollouts.sbatch runs the same script per SAFE-protocol fold).
#
#   sbatch --export=ALL,DATA_DIR=<cache>,RUN_DIR=results/VLA_rollouts/<run>,STEP=50000 slurm/eval-real-robot.sbatch

set -e
set -o allexport; source .env; set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)

DATA_DIR=${DATA_DIR:?set DATA_DIR to the rollout cache}
RUN_DIR=${RUN_DIR:?set RUN_DIR to the ensemble run directory, relative to /V-GPS}
STEP=${STEP:?set STEP to the checkpoint step}
POOLS=${POOLS:-rollout}   # several: join with + (sbatch --export splits on commas)
ENSEMBLE_SIZE=${ENSEMBLE_SIZE:-8}
SEED=${SEED:-44}
# Must match the run being scored; the eval loads the same cached statistics the loader used.
NORMALIZATION=${NORMALIZATION:-bounds}
MAX_EPISODES=${MAX_EPISODES:-0}   # 0 = all; a small value is the pipeline preflight
# Must match the run being scored (see train-real-robot-ensemble.sbatch).
TRAIN_POOL=${TRAIN_POOL:-teleop}
STATISTICS_PATH=${STATISTICS_PATH:-}
FRAME_STRIDE=${FRAME_STRIDE:-20}
SWAP_INSTRUCTION=${SWAP_INSTRUCTION:-False}   # positive control: another task's instruction
CAUSAL_GRIPPER=${CAUSAL_GRIPPER:-False}       # True = per-step gripper binarisation (online-realistic)
OUT=${OUT:-results/real_robot_eval/$(basename ${RUN_DIR})_step${STEP}_${POOLS//+/-}.pkl}
mkdir -p $(dirname ${WORK_DIR}/external/V-GPS/${OUT})

mkdir -p ${WORK_DIR}/logs ${WORK_DIR}/external/V-GPS/results/real_robot_eval

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "run=${RUN_DIR} step=${STEP} pools=${POOLS} -> ${OUT}"

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets:/V-GPS/datasets \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/eval_real_robot.py \
    --config /V-GPS/experiments/configs/train_real_robot_config.py:ensemble_sarsa_finetune \
    --config.ensemble_size ${ENSEMBLE_SIZE} \
    --config.seed ${SEED} \
    --config.real_robot_data_dir /V-GPS/${DATA_DIR} \
    --config.real_robot_normalization ${NORMALIZATION} \
    --config.real_robot_train_pool ${TRAIN_POOL} \
    --config.real_robot_frame_stride ${FRAME_STRIDE} \
    ${STATISTICS_PATH:+--config.real_robot_statistics_path /V-GPS/${STATISTICS_PATH}} \
    --swap_instruction=${SWAP_INSTRUCTION} \
    --config.real_robot_causal_gripper=${CAUSAL_GRIPPER} \
    --run_dir /V-GPS/${RUN_DIR} \
    --step ${STEP} \
    --data_dir /V-GPS/${DATA_DIR} \
    --pools ${POOLS//+/,} \
    --max_episodes ${MAX_EPISODES} \
    --out /V-GPS/${OUT}
