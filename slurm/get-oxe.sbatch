#!/bin/bash
#SBATCH --job-name=oxe_download
#SBATCH --cpus-per-task=20
#SBATCH --partition=day
#SBATCH --mem-per-cpu=6G
#SBATCH --time=23:59:59
#SBATCH --error=logs/oxe_download.%J.err
#SBATCH --output=logs/oxe_download.%J.out
#SBATCH --mail-type=ALL

# Exit immediately if any command fails
set -e

# Create log directory if it doesn't exist
mkdir -p logs

echo "SLURM Job ID: $SLURM_JOB_ID"

# Define paths
WORK_DIR=$(pwd)  # submit from the repository root
DOWNLOAD_DIR=${WORK_DIR}/datasets/open_x
SCRATCH_DIR=/scratch/$SLURM_JOB_ID
RLDS_MOD_DIR=${WORK_DIR}/external/rlds_dataset_mod
SINGULARITY_IMAGE=${WORK_DIR}/oxe_download.sif  # built from containers/get-oxe.def

# Create necessary directories
mkdir -p $DOWNLOAD_DIR
mkdir -p $SCRATCH_DIR

# Set parameters for dataset conversion
N_WORKERS=20                  # Adjust based on available RAM
MAX_EPISODES_IN_MEMORY=200    # Adjust based on available RAM

# Increase limit on number of open files
ulimit -n 20000

# Define datasets to download and process
# Fractal from the OXE bucket, resized to 256x256 by rlds_dataset_mod. Bridge is NOT taken from OXE:
# the loader (octo/data/oxe) expects BridgeData V2 as `bridge_dataset` 1.0.0 (already 256x256),
# downloaded from the Berkeley release at the end of this script.
DATASETS=(
    "fractal20220817_data 0.1.0 resize_and_jpeg_encode"
)

# Process each dataset
for DATASET_INFO in "${DATASETS[@]}"; do
    # Parse dataset info
    read -r DATASET VERSION TRANSFORM <<< "$DATASET_INFO"
    
    echo "===== Processing dataset: $DATASET ====="
    
    # Create directory for the dataset
    mkdir -p ${DOWNLOAD_DIR}/${DATASET}
    
    echo "Downloading ${DATASET}..."
    singularity run --bind $RLDS_MOD_DIR:/rlds_dataset_mod ${SINGULARITY_IMAGE} gsutil -m cp -r gs://gresearch/robotics/${DATASET}/${VERSION} ${DOWNLOAD_DIR}/${DATASET}
    
    echo "Converting ${DATASET}..."
    singularity run --bind $RLDS_MOD_DIR:/rlds_dataset_mod ${SINGULARITY_IMAGE} python3 /rlds_dataset_mod/modify_rlds_dataset.py \
        --dataset=$DATASET \
        --data_dir=$DOWNLOAD_DIR \
        --target_dir=$SCRATCH_DIR \
        --mods=$TRANSFORM \
        --n_workers=$N_WORKERS \
        --max_episodes_in_memory=$MAX_EPISODES_IN_MEMORY
    
    # Move processed data to final destination
    echo "Moving processed ${DATASET} to final destination..."
    rm -rf ${DOWNLOAD_DIR}/${DATASET}
    mv ${SCRATCH_DIR}/${DATASET} ${DOWNLOAD_DIR}/
    
    echo "===== Completed processing: $DATASET ====="
done

echo "===== Downloading BridgeData V2 (bridge_dataset 1.0.0) ====="
wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${DOWNLOAD_DIR}     https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

echo "===== All datasets processed successfully ====="

# Print directory structure for verification
echo "Final directory structure:"
find $DOWNLOAD_DIR -type d | sort