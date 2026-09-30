#!/bin/bash
#SBATCH --job-name=iva_ensemble_sarsa
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:L40S:8
#SBATCH --mail-type=ALL
#SBATCH --partition=L40Sday

# IVA Ensemble Training Script
# Usage:
#   1. From scratch: TRAIN_MODE="scratch", RESUME_PATH=""
#   2. Finetune/resume: TRAIN_MODE="finetune", set RESUME_PATH
#      (auto-detects 7D bridge/fractal vs 8D IVA checkpoint)
#
# Experiment Variations:
#   A) Default: FP_STRATEGY="copy_prev", USE_TP_ONLY=false
#      - Train on _fp files (84% TP + sparse FP injections)
#      - FP steps: reward=-1.0, action=previous step's action
#
#   B) TP-only (OOD experiment): USE_TP_ONLY=true
#      - Train on _tp files (no FP), eval on _fp for OOD detection
#
#   C) Zero action on FP: FP_STRATEGY="zero"
#      - Train on _fp files
#      - FP steps: reward=-1.0, action=zeros
#
# Note: reward is -1.0 under both strategies; they differ only in the action.
#       With USE_TP_ONLY=true, FP_STRATEGY has NO effect -- _tp files carry an
#       action on every step, so the strategy branch never fires.

# Stop script if any command fails
set -e

# Load .env file
set -o allexport
source .env
set +o allexport

# Path to Singularity image
SINGULARITY_IMAGE=train_q.sif

# Define working directory
WORK_DIR=$(pwd)

echo "SLURM Job ID: $SLURM_JOB_ID"

echo "GPU Details with nvidia-smi:"
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi

# =============================================================================
# Configuration
# =============================================================================

# Training mode: "scratch" or "finetune"
TRAIN_MODE="finetune"  # Change to "scratch" for from-scratch training

# IVA dataset path (relative to WORK_DIR)
DATA_DIR="datasets/iva_dataset"

# Checkpoint path (auto-detects 7D bridge/fractal vs 8D IVA)
RESUME_PATH=${RESUME_PATH-}  # finetune: results/VGPS_ensemble/<Bridge/Fractal SayNo run>, relative to external/V-GPS

# Ensemble parameters
BATCH_SIZE=1024
ENSEMBLE_SIZE=8
ALGORITHM="ensemble_sarsa"  # or "ensemble_sarsa_finetune"
SEED=44

# Action on false-premise steps: "copy_prev" (default) or "zero".
# No effect when USE_TP_ONLY=true.
FP_STRATEGY="copy_prev"

# Data loading: false (load _fp) or true (load _tp, OOD experiment)
USE_TP_ONLY=true

# V2 negative demos (bridge/fractal style): random prompt swapping
CREATE_NEGATIVE_DEMOS=true
NEGATIVE_DEMO_RATIO=0.10  # 10% swapped

# Optional: specific tasks to train on (None = all 9 tasks)
# IVA_TASKS="turn_tap,close_jar"

# Note: Consider increasing LR (3e-4) for TP-only or neutral strategy
# (finetune default is 1e-4, may be too conservative for cleaner data)

# Derived paths

SAVE_DIR="results/"
SAVE_DIR_ABSOLUTE="/V-GPS/${SAVE_DIR}"

# Make sure directories exist
mkdir -p ${WORK_DIR}/external/V-GPS/${SAVE_DIR}
mkdir -p ${WORK_DIR}/external/V-GPS/wandb
mkdir -p ${WORK_DIR}/logs

# Select config based on mode

if [ "$TRAIN_MODE" = "finetune" ]; then
    echo "Mode: FINETUNE from pretrained checkpoint"
    ALGORITHM="ensemble_sarsa_finetune"
    [ -n "$RESUME_PATH" ] || { echo "finetune needs RESUME_PATH"; exit 1; }
    NUM_STEPS=200000
elif [ "$TRAIN_MODE" = "scratch" ]; then
    echo "Mode: TRAIN from scratch"
    ALGORITHM="ensemble_sarsa"
    NUM_STEPS=500000
else
    echo "Error: TRAIN_MODE must be 'scratch' or 'finetune'"
    exit 1
fi

echo "Algorithm: $ALGORITHM"
echo "Num steps: $NUM_STEPS"

# Run training

# Build experiment name with strategy info
EXP_NAME="IVA_${ALGORITHM}_ens${ENSEMBLE_SIZE}_b${BATCH_SIZE}s${SEED}"
if [ "$USE_TP_ONLY" = true ]; then
    EXP_NAME="${EXP_NAME}_tponly"
fi
if [ "$FP_STRATEGY" = "zero" ]; then
    EXP_NAME="${EXP_NAME}_fpzero"
fi
if [ "$CREATE_NEGATIVE_DEMOS" = true ]; then
    EXP_NAME="${EXP_NAME}_negdemo${NEGATIVE_DEMO_RATIO}"
fi

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/${DATA_DIR}:/V-GPS/${DATA_DIR} \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/train_iva_ensemble.py \
    --config /V-GPS/experiments/configs/train_iva_config.py:${ALGORITHM} \
    --name "${EXP_NAME}" \
    --project IVA_ensemble \
    --config.ensemble_size ${ENSEMBLE_SIZE} \
    --config.seed ${SEED} \
    --config.num_steps ${NUM_STEPS} \
    --config.save_dir ${SAVE_DIR_ABSOLUTE} \
    --config.batch_size ${BATCH_SIZE} \
    --config.iva_data_dir /V-GPS/${DATA_DIR} \
    --config.iva_fp_strategy ${FP_STRATEGY} \
    $([ "$USE_TP_ONLY" = true ] && echo "--config.iva_use_tp_only True") \
    $([ "$CREATE_NEGATIVE_DEMOS" = true ] && echo "--config.create_negative_demos True --config.negative_demo_ratio ${NEGATIVE_DEMO_RATIO}") \
    ${RESUME_PATH:+--config.resume_path /V-GPS/${RESUME_PATH}} \
    ${IVA_TASKS:+--config.iva_tasks ${IVA_TASKS}}
