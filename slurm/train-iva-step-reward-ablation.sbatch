#!/bin/bash
#SBATCH --job-name=iva_step_reward
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# 200k steps at the measured ~13.9k steps/h is ~14.5h. Override per submission with `sbatch --time=...`.
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
# 144G (12x12G) is NOT enough: the IVA loader holds every frame of the split in RAM as PNG
# bytes, and the job either hangs or is OOM-killed around 144G.
#SBATCH --mem-per-cpu=28G
#SBATCH --gres=gpu:L40S:4
#SBATCH --partition=L40Sday
#
# IVA training with and without step reward (thesis: iva_frame_auroc_reward_design).
#
# Everything is held at the settings of slurm/train-iva-ensemble.sbatch except the two knobs below. Per-member batch is 128 in every arm.
#
#   STEP_REWARD=-1.0  Q = discounted steps-to-termination (the step-reward horizon). Reference design.
#   STEP_REWARD=0.0   step reward removed: a feasible episode has Q=0 at every frame, so only
#                     the negative-demo branch (reward -1, td_mask 1) moves Q. Q becomes
#                     -50 * P(infeasible) instead of a time-to-goal estimate.
#   DISCOUNT          size of the step-reward horizon's window, 1/(1-gamma) steps.
#
# Submit with --export=ALL,ARM=<name>,STEP_REWARD=...,DISCOUNT=...,SEED=...,RESUME_PATH=results/VGPS_ensemble/<Bridge/Fractal SayNo run>
set -e

set -o allexport
source .env
set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)

ARM=${ARM:?ARM must be set}
STEP_REWARD=${STEP_REWARD:--1.0}
DISCOUNT=${DISCOUNT:-0.98}
SEED=${SEED:-44}
ENSEMBLE_SIZE=${ENSEMBLE_SIZE:-4}
BATCH_SIZE=${BATCH_SIZE:-512}          # 128 per member, same as the reference run
NUM_STEPS=${NUM_STEPS:-200000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}  # 50000 is too coarse for 200k-step runs
EVAL_INTERVAL=${EVAL_INTERVAL:-10000}

DATA_DIR="datasets/iva_dataset"
RESUME_PATH=${RESUME_PATH:?set RESUME_PATH=results/VGPS_ensemble/<Bridge/Fractal SayNo run>}
SAVE_DIR_ABSOLUTE="/V-GPS/results/"

echo "SLURM Job ID: $SLURM_JOB_ID"

# Memory trace. Jobs 2245443/4/5 all died at MaxRSS 143.9G of a 144G limit at step ~15.5k, and
# the reference run 1963991 was OOM-killed the same way at step 49999. Whether 336G is headroom
# or merely a later wall decides whether a converged checkpoint is reachable at all, so the
# growth curve is recorded rather than assumed.
( while true; do
    echo "[mem] $(date +%H:%M:%S) $(sstat -j ${SLURM_JOB_ID}.batch --noheader --format=MaxRSS,AveRSS 2>/dev/null | tr -s ' ')"
    sleep 120
  done ) &
MEM_MONITOR_PID=$!
trap "kill ${MEM_MONITOR_PID} 2>/dev/null" EXIT
echo "ARM=$ARM STEP_REWARD=$STEP_REWARD DISCOUNT=$DISCOUNT SEED=$SEED ENS=$ENSEMBLE_SIZE BATCH=$BATCH_SIZE"
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi

mkdir -p ${WORK_DIR}/external/V-GPS/results
mkdir -p ${WORK_DIR}/external/V-GPS/wandb
mkdir -p ${WORK_DIR}/logs

EXP_NAME="IVA_stepreward_${ARM}_ens${ENSEMBLE_SIZE}_b${BATCH_SIZE}s${SEED}_sr${STEP_REWARD}_g${DISCOUNT}_tponly_negdemo0.10"

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/${DATA_DIR}:/V-GPS/${DATA_DIR} \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/train_iva_ensemble.py \
    --config /V-GPS/experiments/configs/train_iva_config.py:ensemble_sarsa_finetune \
    --name "${EXP_NAME}" \
    --project IVA_ensemble \
    --config.ensemble_size ${ENSEMBLE_SIZE} \
    --config.seed ${SEED} \
    --config.num_steps ${NUM_STEPS} \
    --config.save_interval ${SAVE_INTERVAL} \
    --config.eval_interval ${EVAL_INTERVAL} \
    --config.save_dir ${SAVE_DIR_ABSOLUTE} \
    --config.batch_size ${BATCH_SIZE} \
    --config.iva_data_dir /V-GPS/${DATA_DIR} \
    --config.iva_fp_strategy copy_prev \
    --config.iva_use_tp_only True \
    --config.iva_step_reward ${STEP_REWARD} \
    --config.agent_kwargs.discount ${DISCOUNT} \
    --config.create_negative_demos True \
    --config.negative_demo_ratio 0.10 \
    --config.resume_path /V-GPS/${RESUME_PATH}
