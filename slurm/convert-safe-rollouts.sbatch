#!/bin/bash
#SBATCH --job-name=convert_safe_rollouts
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# One-off conversion, no GPU: SAFE's 532 OpenVLA-WidowX rollouts (openvla_widowx.zip, unzipped)
# -> per-episode .npz cache + index.json whose `pool` field carries the finetune / calib / test
# split. train_q_ood.sif because the pickles hold torch tensors. Run with `bash` on the login
# node: cv2 on the compute nodes may load an incompatible host libGL.
#
# Preflight: LIMIT=8 OUT_DIR=datasets/safe_widowx/preflight bash slurm/convert-safe-rollouts.sbatch

set -e

SINGULARITY_IMAGE=train_q_ood.sif
WORK_DIR=$(pwd)
RAW_DIR=${RAW_DIR:-datasets/safe_widowx/openvla_widowx}
N_FINETUNE=${N_FINETUNE:-100}
N_CALIB=${N_CALIB:-72}
SPLIT_SEED=${SPLIT_SEED:-0}
LIMIT=${LIMIT:-0}
UNSEEN_TASKS=${UNSEEN_TASKS:-}   # comma-separated tasks held out of finetune + calibration
OUT_DIR=${OUT_DIR:-datasets/safe_widowx/processed_ft${N_FINETUNE}_cal${N_CALIB}_seed${SPLIT_SEED}}
WORKERS=${WORKERS:-16}

mkdir -p ${WORK_DIR}/logs ${WORK_DIR}/${OUT_DIR}
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "raw: ${RAW_DIR} -> cache: ${OUT_DIR} (finetune ${N_FINETUNE}, calib ${N_CALIB}, seed ${SPLIT_SEED}, limit ${LIMIT})"

singularity run \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets:/V-GPS/datasets \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/convert_safe_rollouts.py \
    --raw_dir /V-GPS/${RAW_DIR} \
    --out_dir /V-GPS/${OUT_DIR} \
    --n_finetune ${N_FINETUNE} \
    --n_calib ${N_CALIB} \
    --seed ${SPLIT_SEED} \
    --limit ${LIMIT} \
    --unseen_tasks "${UNSEEN_TASKS}" \
    --workers ${WORKERS}
