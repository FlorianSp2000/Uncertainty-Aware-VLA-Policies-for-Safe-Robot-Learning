#!/bin/bash
#SBATCH --job-name=eval_vla_rollouts
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=L40Sday

# Score one SAFE-protocol fold with one checkpoint of its finetune run (thesis VLA-rollout
# experiment): per-frame Q of all 8 members on holdout (separate calibration),
# eval_seen (SAFE calibration + seen test) and unseen. Same preset as training, so action
# statistics and gripper treatment cannot differ.
#   BASE_RUN=results/VGPS_ensemble/<Bridge/Fractal SayNo run> PRESET=vla_safe_widowx FOLD=0 RUN=safe_fold0_<timestamp> STEPS="0 10000 50000" sbatch slurm/eval-vla-rollouts.sbatch
# Step 0 scores the Bridge/Fractal base ensemble (zero-shot) at its step 200000.

set -e
set -o allexport; source .env; set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)
PRESET=${PRESET:?vla_safe_widowx or vla_libero_pi0fast}
FOLD=${FOLD:?0, 1 or 2}
STEPS=${STEPS:?space-separated checkpoint steps, 0 = base ensemble (zero-shot)}
SPLIT=${SPLIT:-safe_protocol}   # safe_protocol | foresight_cv
case ${PRESET} in
  vla_safe_widowx)    DATA_DIR=datasets/safe_widowx/${SPLIT}_fold${FOLD}; TAG=safe ;;
  vla_libero_pi0fast) DATA_DIR=datasets/pi0_libero/${SPLIT}_fold${FOLD}; TAG=libero ;;
  *) echo "unknown PRESET ${PRESET}"; exit 1 ;;
esac
if [ "${SPLIT}" = foresight_cv ]; then
  POOLS=holdout,cv_calib,cv_test; TAG=${TAG}_cv; POOLTAG=holdout-cvcalib-cvtest
else
  POOLS=holdout,eval_seen,unseen; POOLTAG=holdout-evalseen-unseen
fi
mkdir -p ${WORK_DIR}/logs
echo "SLURM Job ID: $SLURM_JOB_ID  preset=${PRESET} fold=${FOLD} run=${RUN} steps=[${STEPS}]"

for STEP in ${STEPS}; do
  if [ "${STEP}" = 0 ]; then
    RUN_DIR=${BASE_RUN:?set BASE_RUN to the Bridge/Fractal SayNo run, results/VGPS_ensemble/<run>}
    CKPT=200000; LABEL=zeroshot
  else
    RUN_DIR=results/VLA_rollouts/${RUN:?set RUN to the finetune run directory name}
    CKPT=${STEP}; LABEL=step${STEP}
  fi
  OUT=results/vla_rollouts_v2/${TAG}/fold${FOLD}/eval_${LABEL}_${POOLTAG}.pkl
  mkdir -p $(dirname ${WORK_DIR}/external/V-GPS/${OUT})
  echo "== ${RUN_DIR} @ ${CKPT} -> ${OUT}"
  singularity run --nv \
      -B ${WORK_DIR}/external/V-GPS:/V-GPS \
      -B ${WORK_DIR}/datasets:/V-GPS/datasets \
      ${SINGULARITY_IMAGE} \
      python /V-GPS/experiments/eval_real_robot.py \
      --config /V-GPS/experiments/configs/train_real_robot_config.py:${PRESET} \
      --config.ensemble_size 8 \
      --config.seed 44 \
      --config.real_robot_data_dir /V-GPS/${DATA_DIR} \
      --run_dir /V-GPS/${RUN_DIR} \
      --step ${CKPT} \
      --data_dir /V-GPS/${DATA_DIR} \
      --pools ${POOLS} \
      --out /V-GPS/${OUT}
done
