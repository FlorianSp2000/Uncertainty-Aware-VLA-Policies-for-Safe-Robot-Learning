#!/bin/bash
#SBATCH --job-name=convert_libero_rollouts
#SBATCH --partition=day
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# One-off conversion, no GPU: 500 pi0-FAST LIBERO-10 rollouts (oldTOM/pi0-libero-rollouts, unzipped)
# -> per-episode .npz cache + index.json whose `pool` field carries the finetune / calib / test
# split. Run with `bash` on the login node: cv2 on the compute nodes loads an incompatible host libGL.
#
# Preflight: LIMIT=8 OUT_DIR=datasets/pi0_libero/preflight bash slurm/convert-libero-rollouts.sbatch

set -e

SINGULARITY_IMAGE=train_q_ood.sif
WORK_DIR=$(pwd)
RAW_DIR=${RAW_DIR:-datasets/pi0_libero/pi0fast-libero_10}
N_FINETUNE=${N_FINETUNE:-120}
N_CALIB=${N_CALIB:-90}
SPLIT_SEED=${SPLIT_SEED:-0}
LIMIT=${LIMIT:-0}
OUT_DIR=${OUT_DIR:-datasets/pi0_libero/processed_ft${N_FINETUNE}_cal${N_CALIB}_seed${SPLIT_SEED}}
WORKERS=${WORKERS:-16}

mkdir -p ${WORK_DIR}/logs ${WORK_DIR}/${OUT_DIR}
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "raw: ${RAW_DIR} -> cache: ${OUT_DIR} (finetune ${N_FINETUNE}, calib ${N_CALIB}, seed ${SPLIT_SEED}, limit ${LIMIT})"

singularity run \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets:/V-GPS/datasets \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/convert_libero_rollouts.py \
    --raw_dir /V-GPS/${RAW_DIR} \
    --out_dir /V-GPS/${OUT_DIR} \
    --n_finetune ${N_FINETUNE} \
    --n_calib ${N_CALIB} \
    --seed ${SPLIT_SEED} \
    --limit ${LIMIT} \
    --workers ${WORKERS}
