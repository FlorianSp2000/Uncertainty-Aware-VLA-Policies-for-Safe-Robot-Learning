#!/bin/bash
#SBATCH --job-name=reward_ablation
#SBATCH --output=logs/reward_ablation_%A_%a.out
#SBATCH --error=logs/reward_ablation_%A_%a.err
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:L40S:4
#SBATCH --mail-type=ALL
#SBATCH --partition=L40Sday
#SBATCH --array=0-3%4

# Reward Ablation Experiment for Q-Ensemble
# Array job conditions (all 4 run in parallel):
#   0: baseline (no negative demos)
#   1: negative demos, final_reward=-1 (standard)
#   2: negative demos, final_reward=-10 (moderate penalty)
#   3: negative demos, final_reward=-100 (strong penalty)

set -e

# Load .env file
set -o allexport
source .env
set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"

# GPU info
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi

# Directories
DATA_DIR="datasets/open_x"
SAVE_DIR="results/ablations/"
SAVE_DIR_ABSOLUTE="/V-GPS/${SAVE_DIR}"

mkdir -p ${WORK_DIR}/external/V-GPS/${SAVE_DIR}
mkdir -p logs

# Ablation conditions
INCLUDE_NEG=(false true true true)
NEG_FINAL_REWARD=(1 1 10 100)

INCLUDE_NEG_DEMO=${INCLUDE_NEG[$SLURM_ARRAY_TASK_ID]}
FINAL_REWARD=${NEG_FINAL_REWARD[$SLURM_ARRAY_TASK_ID]}

# Experiment name based on condition
if [ "$INCLUDE_NEG_DEMO" = "true" ]; then
    EXP_NAME="reward_ablation_neg${FINAL_REWARD}"
else
    EXP_NAME="reward_ablation_baseline"
fi

echo "Experiment: $EXP_NAME"
echo "Include negative demos: $INCLUDE_NEG_DEMO"
echo "Negative final reward: -$FINAL_REWARD"

# Training parameters
BATCH_SIZE=512
ENSEMBLE_SIZE=4
NUM_STEPS=200000
NUM_TRAIN_SAMPLES=0
LOG_INTERVAL=1000
EVAL_INTERVAL=20000
SEED=42

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets/open_x:/V-GPS/datasets/open_x \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/ablations/train_reward_ablation.py \
    --config /V-GPS/experiments/configs/train_ensemble_config.py:ensemble_sarsa \
    --oxedata_config /V-GPS/experiments/configs/data_config.py \
    --name "${EXP_NAME}" \
    --project "q-ensemble-ablations" \
    --config.ensemble_size ${ENSEMBLE_SIZE} \
    --config.batch_size ${BATCH_SIZE} \
    --config.num_steps ${NUM_STEPS} \
    --config.num_train_samples ${NUM_TRAIN_SAMPLES} \
    --config.log_interval ${LOG_INTERVAL} \
    --config.eval_interval ${EVAL_INTERVAL} \
    --config.seed ${SEED} \
    --config.create_negative_demos=${INCLUDE_NEG_DEMO} \
    --config.negative_demo_ratio 0.05 \
    --config.negative_final_reward -${FINAL_REWARD} \
    --config.exclude_empty_lang_instr true \
    --config.save_dir ${SAVE_DIR_ABSOLUTE} \
    --config.agent_kwargs.discount 0.98 \
    --oxedata_config.batch_size ${BATCH_SIZE} \
    --oxedata_config.oxe_kwargs.data_dir ${DATA_DIR} \
    --oxedata_config.oxe_kwargs.discount 0.98

echo "Training complete for $EXP_NAME"
