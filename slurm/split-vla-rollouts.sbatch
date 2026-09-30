#!/bin/bash
#SBATCH --job-name=split_vla_rollouts
#SBATCH --partition=day
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# SAFE seen/unseen-task splits of the VLA-rollout caches (thesis VLA-rollout experiment),
# one cache per fold; the .npz files are symlinks into the full cache, no
# reconversion. Cheap: run with `bash` on the login node.
#   DATASET=safe_widowx   bash slurm/split-vla-rollouts.sbatch
#   DATASET=libero_pi0fast bash slurm/split-vla-rollouts.sbatch
#   DATASET=libero_pi0fast_cv bash slurm/split-vla-rollouts.sbatch   (Foresight's 3-fold CV)

set -e

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)
DATASET=${DATASET:?set DATASET to safe_widowx, libero_pi0fast or libero_pi0fast_cv}
FOLDS=${FOLDS:-"0 1 2"}
DRAW_SEED=0      # permutation of the tasks; fold k holds out its k-th block
SPLIT_SEED=0     # episode assignment within the seen tasks
HOLDOUT_FRAC=0.25
PROTOCOL=safe

case ${DATASET} in
  # SAFE App. B.5 / Table 5: WidowX 2 of 8 tasks unseen, D_train : D_eval-seen = 2 : 1
  safe_widowx)    SOURCE_DIR=datasets/safe_widowx/processed_ft100_cal72_seed0; N_UNSEEN=2; SEEN_EVAL_FRAC=0.33 ;;
  # SAFE App. B.5: LIBERO-10 3 of 10 tasks unseen, D_train : D_eval-seen = 60 : 40
  libero_pi0fast) SOURCE_DIR=datasets/pi0_libero/processed_ft120_cal90_seed0; N_UNSEEN=3; SEEN_EVAL_FRAC=0.4 ;;
  # Foresight App. 8: 3-fold CV over rollouts, all tasks seen, non-test folds 6:1:1
  libero_pi0fast_cv) SOURCE_DIR=datasets/pi0_libero/processed_ft120_cal90_seed0; PROTOCOL=foresight_cv ;;
  *) echo "unknown DATASET ${DATASET}"; exit 1 ;;
esac

if [ "${PROTOCOL}" = safe ]; then
  PROTOCOL_ARGS="--draw_seed ${DRAW_SEED} --n_unseen ${N_UNSEEN} --seen_eval_frac ${SEEN_EVAL_FRAC} --holdout_frac ${HOLDOUT_FRAC}"
  PREFIX=safe_protocol
else
  PROTOCOL_ARGS=""
  PREFIX=foresight_cv
fi

for FOLD in ${FOLDS}; do
  OUT_DIR=$(dirname ${SOURCE_DIR})/${PREFIX}_fold${FOLD}
  echo "== ${DATASET} fold ${FOLD} -> ${OUT_DIR}"
  singularity run \
      -B ${WORK_DIR}/external/V-GPS:/V-GPS \
      -B ${WORK_DIR}/datasets:/V-GPS/datasets \
      ${SINGULARITY_IMAGE} \
      python /V-GPS/experiments/split_vla_rollouts.py \
      --source_dir /V-GPS/${SOURCE_DIR} \
      --out_dir /V-GPS/${OUT_DIR} \
      --protocol ${PROTOCOL} --fold ${FOLD} --split_seed ${SPLIT_SEED} ${PROTOCOL_ARGS}
done
