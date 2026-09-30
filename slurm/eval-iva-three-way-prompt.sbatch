#!/bin/bash
#SBATCH --job-name=iva_three_way_prompt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# 10 prompt conditions per frame at 1 encoder pass each: ~1h35m for 8 members, ~55m for 4.
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:L40S:1
#SBATCH --mail-type=ALL
#SBATCH --partition=L40Sday

# Three-way prompt evaluation: does the score react to a wrong instruction, or only to an
# unfamiliar one? Scores every fp-split frame under its own instruction, under IVA's
# injection, and under the train-modal instruction of each of the 8 other tasks. See
# experiments/eval_iva_three_way_prompt.py for why the third condition is the one that decides it.
#
# Env overrides: P3_MODEL_DIR, P3_STEP, P3_TAG, P3_TASKS, P3_MAX_TRAJS, P3_INSTRUCTION_SPLIT.

set -e

set -o allexport
source .env
set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)

echo "SLURM Job ID: $SLURM_JOB_ID"

MODEL_DIR="${P3_MODEL_DIR:?set P3_MODEL_DIR=results/IVA_ensemble/<run> (relative to external/V-GPS)}"
STEP="${P3_STEP-50000}"
# Short, readable checkpoint tag for the artifact directory. Defaulting it off MODEL_DIR
# keeps a forgotten override from silently writing into another checkpoint's directory.
MODEL_TAG="${P3_TAG-$(basename ${MODEL_DIR} | sed -E 's/^IVA_ensemble_sarsa_finetune_//; s/^IVA_stepreward_//; s/_b[0-9]+s[0-9]+//')}"

DATA_DIR="datasets/iva_dataset"
SEED=44
ALGORITHM="ensemble_sarsa_finetune"

DATA_SPLIT="eval"
SPLIT="fp"                                       # the injection condition exists only here
TASKS="${P3_TASKS-}"                             # empty = all 9
MAX_TRAJS="${P3_MAX_TRAJS-0}"                    # 0 = all
INSTRUCTION_SPLIT="${P3_INSTRUCTION_SPLIT-train}"

OUT_DIR="results/iva_cp/${MODEL_TAG}"
mkdir -p ${WORK_DIR}/external/V-GPS/${OUT_DIR}
mkdir -p ${WORK_DIR}/logs

TASK_TAG=${TASKS//,/-}
TASK_TAG=${TASK_TAG:-all}
OUT_PKL="/V-GPS/${OUT_DIR}/threewayprompt_step${STEP}_${DATA_SPLIT}-${SPLIT}_${TASK_TAG}.pkl"
echo "=== three-way-prompt ${MODEL_TAG} step=${STEP} -> ${OUT_PKL} ==="

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/${DATA_DIR}:/V-GPS/${DATA_DIR} \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/eval_iva_three_way_prompt.py \
    --config /V-GPS/experiments/configs/train_iva_config.py:${ALGORITHM} \
    --config.seed ${SEED} \
    --model_dir /V-GPS/${MODEL_DIR} \
    --step ${STEP} \
    --split ${SPLIT} \
    --data_split ${DATA_SPLIT} \
    --instruction_split ${INSTRUCTION_SPLIT} \
    --iva_data_dir /V-GPS/${DATA_DIR} \
    --out ${OUT_PKL} \
    --max_trajs ${MAX_TRAJS} \
    ${TASKS:+--tasks ${TASKS}}

echo "Done. Pkl in external/V-GPS/${OUT_DIR}/"
