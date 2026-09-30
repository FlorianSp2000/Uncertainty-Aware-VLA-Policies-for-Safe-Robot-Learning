#!/bin/bash
#SBATCH --job-name=vgps_classifier_baseline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --mail-type=ALL
#SBATCH --partition=L40Sday

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

# Define paths for data directories
DATA_DIR="datasets/open_x"

# Training configuration
ALGORITHM="binary_classifier"
BATCH_SIZE=256
SEED=42
SAVE_DIR_REL="results/vgps_classifier"

# WandB configuration
WANDB_PROJECT="octo_dual_head_finetuning" # TODO: change project name

# Negative demonstration parameters (match Octo baseline)
NEGATIVE_DEMO_RATIO=0.50
EXCLUDE_EMPTY_LANG_INSTR=true

# Dataset size (number of trajectories to use for training, -1 = use all)
NUM_TRAIN_TRAJS=-1  # Set to -1 to use full dataset

# Training intervals
TRAIN_STEPS=100000
EVAL_INTERVAL=1000
SAVE_INTERVAL=20000
LOG_INTERVAL=100

# Generate experiment name with dataset size suffix
DATASET_SUFFIX=""
if [ "${NUM_TRAIN_TRAJS}" -gt 0 ] 2>/dev/null; then
    DATASET_SUFFIX="_${NUM_TRAIN_TRAJS}trajs"
fi
EXPERIMENT_NAME="ResNet_MUSE_classifier_b${BATCH_SIZE}s${SEED}_neg${NEGATIVE_DEMO_RATIO}${DATASET_SUFFIX}"

# Make sure the save directory exists
mkdir -p ${WORK_DIR}/external/V-GPS/${SAVE_DIR_REL}
mkdir -p ${WORK_DIR}/external/V-GPS/wandb

echo "Starting binary classifier experiment: ${EXPERIMENT_NAME}"
echo "Settings:"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Training steps: ${TRAIN_STEPS}"
echo "  - Num train trajectories: ${NUM_TRAIN_TRAJS:-all}"
echo "  - Negative demo ratio: ${NEGATIVE_DEMO_RATIO}"
echo "  - WandB project: ${WANDB_PROJECT}"
echo "  - Experiment name: ${EXPERIMENT_NAME}"

# Run training
singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets/open_x:/V-GPS/datasets/open_x \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/train_vgps_baseline.py \
    --config=/V-GPS/experiments/configs/train_classifier_config.py:${ALGORITHM} \
    --config.batch_size=${BATCH_SIZE} \
    --config.num_steps=${TRAIN_STEPS} \
    --config.seed=${SEED} \
    --config.save_dir="/V-GPS/${SAVE_DIR_REL}" \
    --config.eval_interval=${EVAL_INTERVAL} \
    --config.save_interval=${SAVE_INTERVAL} \
    --config.log_interval=${LOG_INTERVAL} \
    --config.negative_demo_ratio=${NEGATIVE_DEMO_RATIO} \
    --config.exclude_empty_lang_instr=${EXCLUDE_EMPTY_LANG_INSTR} \
    --train_take_n=${NUM_TRAIN_TRAJS} \
    --oxedata_config=/V-GPS/experiments/configs/data_config.py:baseline_classifier \
    --oxedata_config.oxe_kwargs.data_dir=${DATA_DIR} \
    --oxedata_config.oxe_kwargs.data_mix=bridge_fractal \
    --name="${EXPERIMENT_NAME}" \
    --project="${WANDB_PROJECT}"

echo "Binary classifier training completed for experiment: ${EXPERIMENT_NAME}"
