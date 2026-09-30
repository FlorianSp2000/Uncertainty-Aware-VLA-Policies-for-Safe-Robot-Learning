#!/bin/bash
#SBATCH --job-name=perturb_sens
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=L40Sday
#SBATCH --gres=gpu:L40S:1

# Bridge/Fractal perturbation dumps behind the thesis tables bf_*: score Bridge/Fractal models under
# paired language / image / action perturbations of the same trajectories.
# One HDF5 per (model, pkl) in external/V-GPS/results/perturbation_sensitivity/job_<id>/.
# Nothing is overwritten: outputs carry the job id.
#
# MODELS: "type:path:ensemble_size:batch_size:step:wandb_run" entries, space separated
#   type = ensemble | classifier | octo; wandb_run only for the classifier (its config lives there), else "-".
# SWAPS_FROM_DIR: job dir of an earlier ensemble run whose swapped strings every model reuses
#   (file matched by *_<pkl tag>.hdf5, exactly one); empty = draw swaps in this run.
# Override per submission: sbatch --export=ALL,MODELS="...",... slurm/eval-perturbation-sensitivity.sbatch

set -e
set -o allexport; source .env; set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)
OUT_DIR=/V-GPS/results/perturbation_sensitivity/job_${SLURM_JOB_ID}
mkdir -p ${WORK_DIR}/external/V-GPS/results/perturbation_sensitivity/job_${SLURM_JOB_ID} ${WORK_DIR}/logs

# e.g. MODELS="ensemble:/V-GPS/results/VGPS_ensemble/<run>:8:1024:latest:- ensemble:/V-GPS/results/VGPS_ensemble/<run2>:8:1024:latest:-"
MODELS=${MODELS:?set MODELS, see the header}
PKLS=${PKLS:-"/V-GPS/experiments/data/bridge_fractal_val_n200.pkl /V-GPS/experiments/data/bridge_fractal_n300_with_inpainted_cleaned.pkl"}
N_TRAJ=${N_TRAJ:-0}
N_VERBOSE=${N_VERBOSE:-0}
NEAR_POOL_FRAMES=${NEAR_POOL_FRAMES:-500000}
DUMP_FEATURES=${DUMP_FEATURES:-False}        # True: feat/<cond> encoder features (ensemble only)
CONDITIONS=${CONDITIONS:-}                   # empty = the script's default list (ensemble)
CLASSIFIER_CONDITIONS="orig,lang_swap_train,lang_swap_pkl,lang_swap_near,inpaint,jpeg_orig,composite_local_r8,composite_local_r16,composite_local_r32,composite_global_r8,composite_global_r16,composite_global_r32"
SWAPS_FROM_DIR=${SWAPS_FROM_DIR:-}
QUALITATIVE=${QUALITATIVE:-False}            # True: contact sheet + near-swap pairs per pkl

echo "SLURM Job ID: $SLURM_JOB_ID"
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi

for entry in ${MODELS}; do
  IFS=: read -r TYPE CKPT ENS BS STEP WANDB_RUN <<< "${entry}"
  TAG=$(basename ${CKPT})
  [ "${TYPE}" = "classifier" ] && TAG=$(basename $(dirname $(dirname ${CKPT})))
  STEP_FLAG=""
  if [ "${STEP}" != "latest" ]; then STEP_FLAG="--step ${STEP}"; fi
  for PKL in ${PKLS}; do
    PKL_TAG=$(basename ${PKL} .pkl)
    OUT=${OUT_DIR}/${TAG}_step${STEP}_${PKL_TAG}.hdf5
    SWAPS_FLAG=""
    if [ -n "${SWAPS_FROM_DIR}" ]; then
      REF=$(ls ${WORK_DIR}/external/V-GPS/results/perturbation_sensitivity/${SWAPS_FROM_DIR}/*_${PKL_TAG}.hdf5 | grep -v "_classifier\|Octo" | head -2)
      [ $(echo "${REF}" | wc -l) -eq 1 ] || { echo "need exactly one reference dump for ${PKL_TAG}: ${REF}"; exit 1; }
      SWAPS_FLAG="--swaps_from /V-GPS/results/perturbation_sensitivity/${SWAPS_FROM_DIR}/$(basename ${REF})"
    fi
    COND=${CONDITIONS}
    # classifiers take no action input: score only the language and image conditions
    [ -z "${COND}" ] && [ "${TYPE}" != "ensemble" ] && COND=${CLASSIFIER_CONDITIONS}
    QUAL_FLAG=""
    [ "${QUALITATIVE}" = "True" ] && QUAL_FLAG="--qualitative_dir ${OUT_DIR}/qualitative_${PKL_TAG}"
    echo "=== ${TYPE} ${TAG} (${ENS} members, step ${STEP}) on ${PKL_TAG} ${SWAPS_FLAG}"
    singularity run --nv \
      -B ${WORK_DIR}/external/V-GPS:/V-GPS \
      -B ${WORK_DIR}/datasets/open_x:/V-GPS/datasets/open_x \
      ${SINGULARITY_IMAGE} \
      python /V-GPS/experiments/eval_perturbation_sensitivity.py \
        --model_type ${TYPE} --resume_path ${CKPT} ${STEP_FLAG} --ensemble_size ${ENS} --batch_size ${BS} \
        $([ "${WANDB_RUN}" != "-" ] && echo "--classifier_wandb ${WANDB_RUN}") \
        --pkl ${PKL} --out ${OUT} --n_traj ${N_TRAJ} --n_verbose ${N_VERBOSE} --seed 0 \
        --near_pool_frames ${NEAR_POOL_FRAMES} --dump_features=${DUMP_FEATURES} \
        ${COND:+--conditions ${COND}} ${SWAPS_FLAG} ${QUAL_FLAG}
  done
done

echo "Done. Artifacts in external/V-GPS/results/perturbation_sensitivity/job_${SLURM_JOB_ID}/"
