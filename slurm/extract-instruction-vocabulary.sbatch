#!/bin/bash
#SBATCH --job-name=instruction_vocab
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=day

# Data prep, no GPU: one language instruction per Bridge/Fractal trajectory, training and
# validation split, loaded as in training (experiments/extract_instruction_vocabulary.py).
# Feeds the seen/unseen-instruction split of the object-removal evaluation
# (src/results/instruction-overlap-bridge-fractal/).
#
# Smoke test: sbatch --export=ALL,MAX_TRAJ=500 --time=00:30:00 slurm/extract-instruction-vocabulary.sbatch

set -e
set -o allexport; source .env; set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)
MAX_TRAJ=${MAX_TRAJ:-0}
SPLITS=${SPLITS:-train,val}
TAG=$([ "${MAX_TRAJ}" = "0" ] && echo "full" || echo "smoke${MAX_TRAJ}")
OUT_DIR=/V-GPS/results/instruction_vocabulary
mkdir -p ${WORK_DIR}/external/V-GPS/results/instruction_vocabulary ${WORK_DIR}/logs

echo "SLURM Job ID: $SLURM_JOB_ID  MAX_TRAJ=$MAX_TRAJ  SPLITS=$SPLITS"

singularity run \
  -B ${WORK_DIR}/external/V-GPS:/V-GPS \
  -B ${WORK_DIR}/datasets/open_x:/V-GPS/datasets/open_x \
  ${SINGULARITY_IMAGE} \
  python /V-GPS/experiments/extract_instruction_vocabulary.py \
    --out_path ${OUT_DIR}/instructions_bridge_fractal_${TAG}.hdf5 \
    --splits ${SPLITS} --max_trajectories ${MAX_TRAJ} \
    --oxedata_config /V-GPS/experiments/configs/data_config.py \
    --oxedata_config.oxe_kwargs.data_dir datasets/open_x \
    --oxedata_config.oxe_kwargs.data_mix bridge_fractal \
    --oxedata_config.oxe_kwargs.discount 0.98

echo "Done: external/V-GPS/results/instruction_vocabulary/instructions_bridge_fractal_${TAG}.hdf5"
