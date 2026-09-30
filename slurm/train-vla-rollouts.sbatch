#!/bin/bash
#SBATCH --job-name=train_vla_rollouts
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:L40S:4
#SBATCH --partition=L40Sday

# Finetune the Bridge/Fractal SARSA ensemble (8 members, 2 per GPU) on the successful rollouts
# of one SAFE-protocol fold (thesis VLA-rollout experiment). Every training
# knob is in the config preset (train_real_robot_config.py: vla_safe_widowx /
# vla_libero_pi0fast); this script only picks preset, fold and run name.
#   BASE_RUN=results/VGPS_ensemble/<Bridge/Fractal SayNo run> PRESET=vla_safe_widowx FOLD=0 sbatch slurm/train-vla-rollouts.sbatch
#   PRESET=vla_libero_pi0fast FOLD=0 SPLIT=foresight_cv sbatch ...   (Foresight's 3-fold CV)
# Smoke test: NUM_STEPS=300 SAVE_INTERVAL=300 EVAL_INTERVAL=100 PLOT_INTERVAL=100 (+ --time).

set -e
set -o allexport; source .env; set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)

PRESET=${PRESET:?vla_safe_widowx or vla_libero_pi0fast}
FOLD=${FOLD:?0, 1 or 2}
SPLIT=${SPLIT:-safe_protocol}   # safe_protocol | foresight_cv
case ${PRESET} in
  vla_safe_widowx)    DATA_DIR=datasets/safe_widowx/${SPLIT}_fold${FOLD}; TAG=safe ;;
  vla_libero_pi0fast) DATA_DIR=datasets/pi0_libero/${SPLIT}_fold${FOLD}; TAG=libero ;;
  *) echo "unknown PRESET ${PRESET}"; exit 1 ;;
esac
RESUME_PATH=${BASE_RUN:?set BASE_RUN to the Bridge/Fractal SayNo run, results/VGPS_ensemble/<run>}
PROJECT=VLA_rollouts
if [ "${SPLIT}" = foresight_cv ]; then TAG=${TAG}_cv; fi
NAME=${NAME:-${TAG}_fold${FOLD}}
# Smoke-test overrides only; unset in real runs, the preset decides.
OVERRIDES=""
for V in NUM_STEPS:num_steps SAVE_INTERVAL:save_interval EVAL_INTERVAL:eval_interval PLOT_INTERVAL:plot_interval; do
  ENV=${V%%:*}; KEY=${V##*:}
  if [ -n "${!ENV}" ]; then OVERRIDES="${OVERRIDES} --config.${KEY} ${!ENV}"; fi
done

mkdir -p ${WORK_DIR}/external/V-GPS/results ${WORK_DIR}/logs
echo "SLURM Job ID: $SLURM_JOB_ID  preset=${PRESET} data=${DATA_DIR} name=${NAME} overrides=[${OVERRIDES}]"
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi
cat ${WORK_DIR}/${DATA_DIR}/split.json | head -12

singularity run --nv \
    -B ${WORK_DIR}/external/V-GPS:/V-GPS \
    -B ${WORK_DIR}/datasets:/V-GPS/datasets \
    ${SINGULARITY_IMAGE} \
    python /V-GPS/experiments/train_real_robot_ensemble.py \
    --config /V-GPS/experiments/configs/train_real_robot_config.py:${PRESET} \
    --name "${NAME}" \
    --project ${PROJECT} \
    --config.ensemble_size 8 \
    --config.batch_size 1024 \
    --config.seed 44 \
    --config.save_dir /V-GPS/results/ \
    --config.real_robot_data_dir /V-GPS/${DATA_DIR} \
    --config.resume_path /V-GPS/${RESUME_PATH} \
    ${OVERRIDES}
