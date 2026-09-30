#!/bin/bash
#SBATCH --job-name=vgps_ensemble_sarsa
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:L40S:8
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

# Define scratch directories for improved I/O performance
SCRATCH_DIR=/scratch/$SLURM_JOB_ID
mkdir -p ${SCRATCH_DIR}
# echo "Scratch directory: $SCRATCH_DIR"

# Copy dataset to scratch for faster access
# rsync -a ${WORK_DIR}/datasets/open_x ${SCRATCH_DIR}/

# Define paths for data and save directories as they appear inside the container
DATA_DIR="datasets/open_x"
SAVE_DIR="results/"
SAVE_DIR_ABSOLUTE="/V-GPS/${SAVE_DIR}"

# Make sure the save directory exists
mkdir -p ${WORK_DIR}/external/V-GPS/${SAVE_DIR}
mkdir -p ${WORK_DIR}/external/V-GPS/wandb

# Ensemble-specific parameters
BATCH_SIZE=1024  # Total batch size (will be divided by ensemble_size)
ENSEMBLE_SIZE=8  # One model per GPU
#BATCH_SIZE_PER_MEMBER=32  # 256 / 8 = 32 per member
ALGORITHM="ensemble_sarsa"  # Use ensemble config
SEED=44
# Resume an interrupted run: RESUME_PATH=/V-GPS/results/VGPS_ensemble/<run> sbatch ...
RESUME_PATH=${RESUME_PATH:-}

# Bind mount the datasets directory at runtime instead of copying it into the image
# For ensemble training, we use train_ensemble.py and ensemble-specific configs
# --config.eval_interval 1000 \
    # --batch_size_per_member ${BATCH_SIZE_PER_MEMBER} \
    # -B ${WORK_DIR}/external/V-GPS/${SAVE_DIR}:${SAVE_DIR_ABSOLUTE} \
    # -B ${WORK_DIR}/external/V-GPS/wandb:/V-GPS/wandb \

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets/open_x:/V-GPS/datasets/open_x \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/train_ensemble.py \
    --config /V-GPS/experiments/configs/train_ensemble_config.py:${ALGORITHM} \
    --oxedata_config /V-GPS/experiments/configs/data_config.py \
    --name "VGPS_${ALGORITHM}_bridge_fractal_ens${ENSEMBLE_SIZE}_b${BATCH_SIZE}s${SEED}" \
    --project VGPS_ensemble \
    --config.ensemble_size ${ENSEMBLE_SIZE} \
    --config.num_steps 510000 \
    --config.seed ${SEED} \
    ${RESUME_PATH:+--config.resume_path ${RESUME_PATH}} \
    --config.save_dir ${SAVE_DIR_ABSOLUTE} \
    --config.batch_size ${BATCH_SIZE} \
    --config.agent_kwargs.discount 0.98 \
    --oxedata_config.batch_size ${BATCH_SIZE} \
    --oxedata_config.oxe_kwargs.data_dir ${DATA_DIR} \
    --oxedata_config.oxe_kwargs.data_mix bridge_fractal \
    --oxedata_config.oxe_kwargs.discount 0.98