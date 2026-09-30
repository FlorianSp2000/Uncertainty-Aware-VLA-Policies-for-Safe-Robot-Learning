#!/bin/bash
#SBATCH --job-name=critic_latency
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=L40Sday
#SBATCH --gres=gpu:L40S:1

# Critic latency (thesis table bf_latency): 1 vs all
# members, at B candidate actions per call. One JSON in results/critic_latency/job_<id>/.

set -e
set -o allexport; source .env; set +o allexport

SINGULARITY_IMAGE=train_q.sif
WORK_DIR=$(pwd)
CKPT=${CKPT:?set CKPT=/V-GPS/results/VGPS_ensemble/<run> (SayNo checkpoint dir)}
ENS=${ENS:-8}
BS=${BS:-1024}
mkdir -p ${WORK_DIR}/external/V-GPS/results/critic_latency/job_${SLURM_JOB_ID}

echo "SLURM Job ID: $SLURM_JOB_ID"
singularity run --nv ${SINGULARITY_IMAGE} nvidia-smi
singularity run --nv \
  -B ${WORK_DIR}/external/V-GPS:/V-GPS \
  -B ${WORK_DIR}/datasets/open_x:/V-GPS/datasets/open_x \
  ${SINGULARITY_IMAGE} \
  python /V-GPS/experiments/benchmark_critic_latency.py \
    --resume_path ${CKPT} --ensemble_size ${ENS} --batch_size ${BS} \
    --out /V-GPS/results/critic_latency/job_${SLURM_JOB_ID}/latency_$(basename ${CKPT}).json
