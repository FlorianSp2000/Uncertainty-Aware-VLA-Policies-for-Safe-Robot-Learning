#!/bin/bash
#SBATCH --job-name=vgps_ensemble_roboreward
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

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)

echo "SLURM Job ID: $SLURM_JOB_ID"
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi

mkdir -p ${WORK_DIR}/external/V-GPS/results
mkdir -p ${WORK_DIR}/external/V-GPS/wandb
mkdir -p logs

# Paths (as seen inside container)
DATA_DIR="datasets/roboreward/rlds"
SAVE_DIR="results/"
SAVE_DIR_ABSOLUTE="/V-GPS/${SAVE_DIR}"

BATCH_SIZE=512
ENSEMBLE_SIZE=4
ALGORITHM="ensemble_sarsa_roboreward"
SEED=44
INPAINTING_DATA_1="/V-GPS/experiments/data/bridge_fractal_n300_with_inpainted_cleaned.pkl"
INPAINTING_DATA_2="/V-GPS/experiments/data/bridge_fractal_train_n200_inpainted_cleaned.pkl"
#      \
    # --config.reward_scheme.binary_pos_threshold 5 \
    # --config.reward_scheme.binary_pos_val 1.0 \
    # --config.reward_scheme.binary_neg_val 0.0 \
    # --config.reward_scheme.td_mask_score_threshold 5 \

#     --config.reward_scheme.normalize_by_traj_len=True \

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets/roboreward/rlds:/V-GPS/datasets/roboreward/rlds \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/train_ensemble_roboreward.py \
    --config /V-GPS/experiments/configs/train_ensemble_config.py:${ALGORITHM} \
    --oxedata_config /V-GPS/experiments/configs/data_config.py \
    --name "VGPS_${ALGORITHM}_roboreward_ens${ENSEMBLE_SIZE}_b${BATCH_SIZE}s${SEED}" \
    --project VGPS_roboreward_ensemble \
    --config.ensemble_size ${ENSEMBLE_SIZE} \
    --config.num_steps 1000000 \
    --config.seed ${SEED} \
    --config.save_dir ${SAVE_DIR_ABSOLUTE} \
    --config.batch_size ${BATCH_SIZE} \
    --config.agent_kwargs.discount 0.98 \
    --oxedata_config.batch_size ${BATCH_SIZE} \
    --oxedata_config.oxe_kwargs.data_dir ${DATA_DIR} \
    --oxedata_config.oxe_kwargs.data_mix roboreward \
    --oxedata_config.oxe_kwargs.discount 0.98 \
    --config.reward_scheme.per_step_reward -1.0 \
    --config.reward_scheme.terminal_transform raw \
    --config.reward_scheme.td_mask_mode fixed \
    --config.eval_inpainting=True \
    --config.eval_inpainting_data_paths "${INPAINTING_DATA_1}:${INPAINTING_DATA_2}"
