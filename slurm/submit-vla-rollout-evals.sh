#!/bin/bash
# Submit eval-vla-rollouts.sbatch for every saved checkpoint of the VLA-rollout finetune runs that
# has neither a dump nor a queued/running eval job yet. Idempotent; run from the repo root.
#   bash slurm/submit-vla-rollout-evals.sh [extra sbatch args, e.g. --partition=day --gres=gpu:A4000:1]
set -e
RESULTS=external/V-GPS/results
for RUN_PATH in ${RESULTS}/VLA_rollouts/{safe,libero,libero_cv}_fold[0-2]_*; do
  [ -d "${RUN_PATH}/ensemble_member_7" ] || continue
  RUN=$(basename ${RUN_PATH}); TAG=${RUN%%_fold*}; FOLD=${RUN#*_fold}; FOLD=${FOLD%%_*}
  case ${TAG} in
    safe) PRESET=vla_safe_widowx; SPLIT=safe_protocol; POOLTAG=holdout-evalseen-unseen ;;
    libero) PRESET=vla_libero_pi0fast; SPLIT=safe_protocol; POOLTAG=holdout-evalseen-unseen ;;
    libero_cv) PRESET=vla_libero_pi0fast; SPLIT=foresight_cv; POOLTAG=holdout-cvcalib-cvtest ;;
  esac
  for CKPT in ${RUN_PATH}/ensemble_member_7/checkpoint_*; do
    STEP=${CKPT##*_}
    DUMP=${RESULTS}/vla_rollouts_v2/${TAG}/fold${FOLD}/eval_step${STEP}_${POOLTAG}.pkl
    NAME=eval_vla_${TAG}_f${FOLD}_s${STEP}
    if [ -f "${DUMP}" ] || squeue --me -h -o %j | grep -qx "${NAME}"; then continue; fi
    sbatch --parsable --job-name=${NAME} "$@" \
      --export=ALL,BASE_RUN=${BASE_RUN:?set BASE_RUN (see eval-vla-rollouts.sbatch)},PRESET=${PRESET},SPLIT=${SPLIT},FOLD=${FOLD},RUN=${RUN},STEPS=${STEP} slurm/eval-vla-rollouts.sbatch
  done
done
