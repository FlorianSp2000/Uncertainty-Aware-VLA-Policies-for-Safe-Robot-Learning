#!/bin/bash
#SBATCH --job-name=roboreward_download
#SBATCH --cpus-per-task=8
#SBATCH --partition=day
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2:00:00
#SBATCH --error=logs/roboreward_download.%J.err
#SBATCH --output=logs/roboreward_download.%J.out
#SBATCH --mail-type=ALL

# Exit immediately if any command fails
set -e

# Create log directory if it doesn't exist
mkdir -p logs

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Define paths
WORK_DIR=$(pwd)  # submit from the repository root
DOWNLOAD_DIR=${WORK_DIR}/datasets/roboreward
SINGULARITY_IMAGE=${WORK_DIR}/roboreward.sif

# Create dataset directory
mkdir -p $DOWNLOAD_DIR

echo "===== Downloading RoboReward Dataset ====="
echo "Target directory: $DOWNLOAD_DIR"
echo "Using container: $SINGULARITY_IMAGE"

# Download RoboReward dataset
echo "Downloading RoboReward (train + validation + test)..."
singularity run -B ${WORK_DIR}:${WORK_DIR} $SINGULARITY_IMAGE python3 ${WORK_DIR}/src/download_roboreward.py --cache_dir $DOWNLOAD_DIR

echo ""
echo "===== Download Summary ====="
echo "Dataset location: $DOWNLOAD_DIR"
echo "End time: $(date)"

# Show disk usage
echo ""
echo "Disk usage:"
du -sh $DOWNLOAD_DIR
echo ""
echo "Directory contents:"
find $DOWNLOAD_DIR -type f -name "*.arrow" -o -name "*.parquet" | head -20

echo "===== RoboReward download completed successfully ====="
