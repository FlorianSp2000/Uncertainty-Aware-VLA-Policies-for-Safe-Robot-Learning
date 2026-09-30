#!/bin/bash
#SBATCH --job-name=iva_cp_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# ~7 min per task per split (25 trajs x 7 members), 2 splits -> ~2h15m for all 9
# tasks (turn_tap alone: ~17 min). Headroom for the
# longer tasks (put_money_in_safe ~181 frames/ep vs turn_tap ~159).
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:L40S:1
#SBATCH --mail-type=ALL
#SBATCH --partition=L40Sday

# Inference of pre-trained Q SARSA ensemble on IVA dataset. Runs the 8-member SARSA ensemble on 1 GPU (members
# evaluated sequentially) and dumps per-frame disagreement pkls for both splits:
#   tp = feasible-only trajectories -> CP calibration
#   fp = native false-premise injections -> detection

# Validation run: TASKS="close_jar"; full run: TASKS="${IVA_TASKS-}" (all 9).

set -e

set -o allexport
source .env
set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)

echo "SLURM Job ID: $SLURM_JOB_ID"

# =============================================================================
# Configuration
# =============================================================================

# Run 3 (tponly + negdemo 0.10), 50k checkpoint — the CP base model
MODEL_DIR="${IVA_MODEL_DIR:?set IVA_MODEL_DIR=results/IVA_ensemble/<run> (relative to external/V-GPS)}"
STEP="${IVA_STEP-50000}"

# Splits to score. The disagreement-floor comparison only needs "tp" (the floor is
# measured on feasible frames); "fp" adds the paired language effect and doubles the
# walltime. Ensemble discount does not enter a forward pass, so a gamma=0.90 arm is
# scored correctly by the default config template.
SPLITS="${IVA_SPLITS-tp fp}"

DATA_DIR="datasets/iva_dataset"
SEED=44                      # must match the training run (dataset ordering + eval seeds)
ALGORITHM="ensemble_sarsa_finetune"

# Comma-separated task subset; empty = all 9 tasks
TASKS="${IVA_TASKS-}"

# Which IVA split to score. "eval" is the working configuration.
# "train" was tried as a disjoint, leak-free calibration source and FAILED: the
# ensemble was finetuned on Train, so its scores are not exchangeable with eval
# To reproduce that negative result: DATA_SPLIT=train MAX_TRAJS=50 KEEP_IMAGES=false, walltime 08:00:00
DATA_SPLIT="${IVA_DATA_SPLIT-eval}"

MAX_TRAJS="${IVA_MAX_TRAJS-0}"                  # 0 = all trajectories per task
KEEP_IMAGES="${IVA_KEEP_IMAGES-true}"             # PNG-encode every frame -> self-contained pkl
IMAGE_SIZE=128               # stored frame size (model always sees 256); 128 = IVA native, ~3x smaller
ALPHA=0.15                   # CP significance level, recorded in the dump
LOG_PARAM_TREE=false         # dump every param array (~150 lines/split); the [wiring] config log is always on

# Artifact naming: the dir names the scored
# checkpoint, the filename says run kind, step, data split, detection split, tasks.
MODEL_TAG=$(basename ${MODEL_DIR} | sed -E 's/^IVA_ensemble_sarsa_finetune_//; s/_b[0-9]+s[0-9]+//')
OUT_DIR="results/iva_cp/${MODEL_TAG}"

mkdir -p ${WORK_DIR}/external/V-GPS/${OUT_DIR}
mkdir -p ${WORK_DIR}/logs

# =============================================================================
# Run inference for both splits
# =============================================================================

# task list -> filename tag: "close_jar,turn_tap" -> "close_jar-turn_tap";
# empty TASKS means "all tasks", tagged "all"
TASK_TAG=${TASKS//,/-}
TASK_TAG=${TASK_TAG:-all}

for SPLIT in ${SPLITS}; do
    OUT_PKL="/V-GPS/${OUT_DIR}/evalcp_step${STEP}_${DATA_SPLIT}-${SPLIT}_${TASK_TAG}.pkl"
    echo "=== split=${SPLIT} -> ${OUT_PKL} ==="

    singularity run --nv \
        -B ${WORK_DIR}/external/V-GPS:/V-GPS \
        -B ${WORK_DIR}/${DATA_DIR}:/V-GPS/${DATA_DIR} \
        ${SINGULARITY_IMAGE} \
        python /V-GPS/experiments/eval_iva_cp.py \
        --config /V-GPS/experiments/configs/train_iva_config.py:${ALGORITHM} \
        --config.seed ${SEED} \
        --model_dir /V-GPS/${MODEL_DIR} \
        --step ${STEP} \
        --split ${SPLIT} \
        --data_split ${DATA_SPLIT} \
        --iva_data_dir /V-GPS/${DATA_DIR} \
        --out ${OUT_PKL} \
        --max_trajs ${MAX_TRAJS} \
        --keep_images=${KEEP_IMAGES} \
        --image_size ${IMAGE_SIZE} \
        --log_param_tree=${LOG_PARAM_TREE} \
        --alpha ${ALPHA} \
        ${TASKS:+--tasks ${TASKS}}
done

echo "Done. Pkls in external/V-GPS/${OUT_DIR}/"
