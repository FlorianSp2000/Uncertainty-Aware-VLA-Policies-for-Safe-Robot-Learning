#!/bin/bash
#SBATCH --job-name=contrastive-ensemble
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --partition=L40Sday
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:L40S:4

# Stop script if any command fails
# --mem-per-cpu=14G

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

# Define paths for data and save directories as they appear inside the container
DATA_DIR="datasets/open_x"
SAVE_DIR="results/"
SAVE_DIR_ABSOLUTE="/V-GPS/${SAVE_DIR}"

# Make sure the save directory exists
mkdir -p ${WORK_DIR}/external/V-GPS/${SAVE_DIR}

# Ensemble-specific parameters
BATCH_SIZE=512
ENSEMBLE_SIZE=4
ALGORITHM="ensemble_sarsa"
SEED=44

# Contrastive loss parameters
CONTRASTIVE_WEIGHT=1.0
CONTRASTIVE_MARGIN=0.5
CONTRASTIVE_NEGATIVE_RATIO=0.45
CONTRASTIVE_PAIRING_MODE="paired"   # random | paired | hybrid_paired

EXCLUDE_EMPTY_LANG_INSTR=true

PAIRING_SUFFIX=""
if [ "${CONTRASTIVE_PAIRING_MODE}" != "random" ]; then
    PAIRING_SUFFIX="_${CONTRASTIVE_PAIRING_MODE}"
fi
EXPERIMENT_NAME="VGPS_${ALGORITHM}_bridge_fractal_ens${ENSEMBLE_SIZE}_b${BATCH_SIZE}s${SEED}_contrastive${CONTRASTIVE_WEIGHT}_neg${CONTRASTIVE_NEGATIVE_RATIO}${PAIRING_SUFFIX}"

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets/open_x:/V-GPS/datasets/open_x \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/train_ensemble.py \
    --config /V-GPS/experiments/configs/train_ensemble_config.py:${ALGORITHM} \
    --oxedata_config /V-GPS/experiments/configs/data_config.py \
    --name "${EXPERIMENT_NAME}" \
    --project VGPS_ensemble \
    --config.ensemble_size ${ENSEMBLE_SIZE} \
    --config.num_steps 510000 \
    --config.seed ${SEED} \
    --config.save_dir ${SAVE_DIR_ABSOLUTE} \
    --config.batch_size ${BATCH_SIZE} \
    --config.agent_kwargs.discount 0.98 \
    --config.agent_kwargs.contrastive_weight=${CONTRASTIVE_WEIGHT} \
    --config.agent_kwargs.contrastive_margin=${CONTRASTIVE_MARGIN} \
    --config.contrastive_negative_ratio=${CONTRASTIVE_NEGATIVE_RATIO} \
    --config.contrastive_pairing_mode=${CONTRASTIVE_PAIRING_MODE} \
    --config.exclude_empty_lang_instr=${EXCLUDE_EMPTY_LANG_INSTR} \
    --oxedata_config.batch_size ${BATCH_SIZE} \
    --oxedata_config.oxe_kwargs.data_dir ${DATA_DIR} \
    --oxedata_config.oxe_kwargs.data_mix bridge_fractal \
    --oxedata_config.oxe_kwargs.discount 0.98
