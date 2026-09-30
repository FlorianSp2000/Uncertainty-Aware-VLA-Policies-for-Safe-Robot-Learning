#!/bin/bash
#SBATCH --job-name=pi0_download
#SBATCH --cpus-per-task=4
#SBATCH --partition=day
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --error=logs/pi0_download.%J.err
#SBATCH --output=logs/pi0_download.%J.out
#SBATCH --mail-type=ALL

set -e
mkdir -p logs

echo "SLURM Job ID: $SLURM_JOB_ID"

WORK_DIR=$(pwd)  # submit from the repository root
TARGET_DIR=${WORK_DIR}/external/V-GPS/checkpoints/pi0_base
mkdir -p $TARGET_DIR

echo "Downloading pi0_base checkpoint from gs://openpi-assets/checkpoints/pi0_base/ ..."
singularity run ${WORK_DIR}/oxe_download.sif \
    gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_base/ ${WORK_DIR}/external/V-GPS/checkpoints/

echo "Done. Contents:"
find $TARGET_DIR -type f | sort
du -sh $TARGET_DIR
