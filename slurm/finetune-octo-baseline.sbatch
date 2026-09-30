#!/bin/bash
#SBATCH --job-name=octo_finetune_baseline
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

# Path to Singularity image (update this to your actual path)
SINGULARITY_IMAGE=train_q.sif

# Define working directory (adjust as needed)
WORK_DIR=$(pwd)

echo "SLURM Job ID: $SLURM_JOB_ID"

echo "GPU Details with nvidia-smi:"
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi

singularity run --nv ${SINGULARITY_IMAGE} echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Define paths for data directories as they appear inside the container
DATA_DIR="datasets/open_x"


# Core Octo finetuning configuration (now driven by finetune_config.py)
FINETUNING_MODE="full"  # Options: "full", "head_only", "head_mlp_only", "feasibility_head_only"
TASK_MODALITY="language_conditioned"  # "image_conditioned", "language_conditioned", "multimodal"; Has no influence in our script as we use data_config.py to set data loading
MODEL_SIZE="small"               
PRETRAINED_PATH="hf://rail-berkeley/octo-$MODEL_SIZE-1.5"  # Pretrained Octo model "hf://rail-berkeley/octo-base-1.5"
BATCH_SIZE=256                 # Batch size for single GPU finetuning
SEED=42                       # Random seed
SAVE_DIR_REL="results/octo_finetuning"  # Save directory (relative to /V-GPS)

# WandB configuration
WANDB_PROJECT="octo_dual_head_finetuning"

# Custom loss weights for dual-head architecture
ACTION_LOSS_WEIGHT=0.0        # Weight for action prediction loss
CLASSIFICATION_LOSS_WEIGHT=1.0 # Weight for feasibility classification loss

# V-GPS negative demonstration parameters
NEGATIVE_DEMO_RATIO=0.50      # 50% negative demonstrations
EXCLUDE_EMPTY_LANG_INSTR=true # Exclude empty language instructions
USE_DEBUG_LABELS=false        # Set to true to use 'feasible'/'infeasible' label strings (debug mode)

# Training intervals (override config defaults)
TRAIN_STEPS=100000              # Number of training steps
EVAL_INTERVAL=1000            # Run validation every 1000 steps
SAVE_INTERVAL=10000            # Save checkpoint every 2500 steps
LOG_INTERVAL=100               # Log training metrics every 100 steps

# Generate experiment name
NEG_DEMO_SUFFIX="_neg${NEGATIVE_DEMO_RATIO}"
DEBUG_SUFFIX=""
if [ "${USE_DEBUG_LABELS}" = "true" ]; then
    DEBUG_SUFFIX="_debug"
fi
# _bridge_fractal
EXPERIMENT_NAME="Octo_${MODEL_SIZE}_act${ACTION_LOSS_WEIGHT}_cls${CLASSIFICATION_LOSS_WEIGHT}_${FINETUNING_MODE}_b${BATCH_SIZE}s${SEED}${NEG_DEMO_SUFFIX}${DEBUG_SUFFIX}"

# Make sure the experiment-specific save directory exists
mkdir -p ${WORK_DIR}/external/V-GPS/results/octo_finetuning/${EXPERIMENT_NAME}
mkdir -p ${WORK_DIR}/external/V-GPS/wandb

echo "Starting Octo finetuning experiment: ${EXPERIMENT_NAME}"
echo "🔧 Using data_config.py:baseline_classifier mode for Octo-compatible settings"
echo "Octo finetuning settings:"
echo "  - Pretrained path: ${PRETRAINED_PATH}"
echo "  - Finetuning mode: ${FINETUNING_MODE}"
echo "  - Task modality: ${TASK_MODALITY}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Training steps: ${TRAIN_STEPS}"
echo "  - Negative demo ratio: ${NEGATIVE_DEMO_RATIO}"
echo "  - Debug labels mode: ${USE_DEBUG_LABELS}"
echo "  - Action loss weight: ${ACTION_LOSS_WEIGHT}"
echo "  - Classification loss weight: ${CLASSIFICATION_LOSS_WEIGHT}"
echo "  - WandB project: ${WANDB_PROJECT}"
echo "  - Experiment name: ${EXPERIMENT_NAME}"

#     --config.pretrained_step=null \

# Run Octo finetuning with config-driven approach
singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets/open_x:/V-GPS/datasets/open_x \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/finetune_octo_baseline.py \
    --config=/V-GPS/octo/scripts/configs/finetune_config.py:${FINETUNING_MODE},${TASK_MODALITY} \
    --config.pretrained_path="${PRETRAINED_PATH}" \
    --config.batch_size=${BATCH_SIZE} \
    --config.num_steps=${TRAIN_STEPS} \
    --config.seed=${SEED} \
    --config.save_dir="/V-GPS/${SAVE_DIR_REL}/${EXPERIMENT_NAME}" \
    --config.wandb.project="${WANDB_PROJECT}" \
    --config.eval_interval=${EVAL_INTERVAL} \
    --config.save_interval=${SAVE_INTERVAL} \
    --config.log_interval=${LOG_INTERVAL} \
    --oxedata_config=/V-GPS/experiments/configs/data_config.py:baseline_classifier \
    --oxedata_config.batch_size=${BATCH_SIZE} \
    --oxedata_config.oxe_kwargs.data_dir=${DATA_DIR} \
    --oxedata_config.oxe_kwargs.data_mix=bridge_fractal \
    --action_loss_weight=${ACTION_LOSS_WEIGHT} \
    --classification_loss_weight=${CLASSIFICATION_LOSS_WEIGHT} \
    --negative_demo_ratio=${NEGATIVE_DEMO_RATIO} \
    --train_take_n=0 \
    --exclude_empty_lang_instr=${EXCLUDE_EMPTY_LANG_INSTR} \
    --use_debug_labels=${USE_DEBUG_LABELS} \
    --experiment_name="${EXPERIMENT_NAME}"

echo "Octo finetuning completed for experiment: ${EXPERIMENT_NAME}"