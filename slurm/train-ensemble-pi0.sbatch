#!/bin/bash
#SBATCH --job-name=siglip_ensemble_sarsa
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:L40S:4
#SBATCH --mail-type=ALL
#SBATCH --partition=L40Sday

set -e

set -o allexport
source .env
set +o allexport

SINGULARITY_IMAGE=train_pi0.sif  # built from containers/train-pi0.def
WORK_DIR=$(pwd)

echo "SLURM Job ID: $SLURM_JOB_ID"
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi

# Paths inside container
DATA_DIR="datasets/roboreward/rlds"
SAVE_DIR="results/"
SAVE_DIR_ABSOLUTE="/V-GPS/${SAVE_DIR}"

mkdir -p ${WORK_DIR}/external/V-GPS/${SAVE_DIR}
mkdir -p ${WORK_DIR}/external/V-GPS/wandb

# SigLIP ensemble params
FREEZE_BACKBONE=${FREEZE_BACKBONE:-False}    # True = frozen PoC, False = full finetune (needs smaller batch + lower LR)
BATCH_SIZE=${BATCH_SIZE:-128}           # 256 for frozen, 128 for unfrozen (backward through 412M)
ENSEMBLE_SIZE=4          # 4 GPUs, 1 model per GPU
LEARNING_RATE=${LEARNING_RATE:-1e-4}       # 3e-4 for frozen, 1e-4 for unfrozen ViT
ALGORITHM="ensemble_sarsa_roboreward_siglip"
SEED=44
RESUME_PATH=""
INPAINTING_DATA_1="/V-GPS/experiments/data/bridge_fractal_n300_with_inpainted_cleaned.pkl"
INPAINTING_DATA_2="/V-GPS/experiments/data/bridge_fractal_train_n200_inpainted_cleaned.pkl"

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/external/openpi:/openpi \
    -B ${WORK_DIR}/datasets/roboreward/rlds:/V-GPS/datasets/roboreward/rlds \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/train_ensemble_roboreward_siglip.py \
    --config /V-GPS/experiments/configs/train_ensemble_config.py:${ALGORITHM} \
    --oxedata_config /V-GPS/experiments/configs/data_config.py:siglip \
    --name "siglip_frz${FREEZE_BACKBONE}_ens${ENSEMBLE_SIZE}_b${BATCH_SIZE}_lr${LEARNING_RATE}_s${SEED}" \
    --project jaxrl_m_siglip_ensemble \
    --config.ensemble_size ${ENSEMBLE_SIZE} \
    --config.num_steps 200000 \
    --config.seed ${SEED} \
    --config.save_dir ${SAVE_DIR_ABSOLUTE} \
    --config.batch_size ${BATCH_SIZE} \
    --config.siglip_checkpoint_path "/V-GPS/checkpoints/pi0_base/params" \
    --config.agent_kwargs.discount 0.98 \
    --config.agent_kwargs.freeze_siglip_backbone=${FREEZE_BACKBONE} \
    --config.agent_kwargs.learning_rate ${LEARNING_RATE} \
    --oxedata_config.batch_size ${BATCH_SIZE} \
    --oxedata_config.oxe_kwargs.data_dir ${DATA_DIR} \
    --oxedata_config.oxe_kwargs.data_mix roboreward \
    --oxedata_config.oxe_kwargs.discount 0.98 \
    --config.eval_inpainting=True \
    --config.eval_inpainting_data_paths "${INPAINTING_DATA_1}:${INPAINTING_DATA_2}"
