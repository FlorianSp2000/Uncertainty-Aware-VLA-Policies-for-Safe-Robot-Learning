#!/bin/bash
#SBATCH --job-name=roboreward_to_rlds
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=day
#SBATCH --time=24:00:00
#SBATCH --output=logs/roboreward_rlds.%J.out
#SBATCH --error=logs/roboreward_rlds.%J.err

set -e
mkdir -p logs

# needed for large datasets (many temp tfrecord shards), same as get-oxe.sbatch
ulimit -n 20000

# ── Paths ──────────────────────────────────────────────────────────────────
WORK_DIR=$(pwd)  # submit from the repository root
RLDS_MOD_DIR="${WORK_DIR}/external/rlds_dataset_mod"
# HF snapshot of teetone/RoboReward written by slurm/get-roboreward.sbatch (revision used in the thesis)
SNAPSHOT_DIR=${SNAPSHOT_DIR:-"${WORK_DIR}/datasets/roboreward/datasets--teetone--RoboReward/snapshots/469b9af76e8539e8d2ac553b081307738fd92ca5"}
TARGET_DIR="${WORK_DIR}/datasets/roboreward/rlds"
SINGULARITY_IMAGE="${WORK_DIR}/roboreward.sif"  # built from containers/roboreward.def
BUILDER_SCRIPT="/rlds_dataset_mod/roboreward_builder/roboreward_dataset_builder.py"

# ── All 29 source datasets (exact subdirectory names in snapshot/train/) ──
DATASETS=(
    "austin_buds_dataset_converted_externally_to_rlds"
    "austin_sirius_dataset_converted_externally_to_rlds"
    "berkeley_autolab_ur5"
    "berkeley_fanuc_manipulation"
    "berkeley_mvp_converted_externally_to_rlds"
    "berkeley_rpt_converted_externally_to_rlds"
    "bridge"
    "cmu_play_fusion"
    "dlr_edan_shared_control_converted_externally_to_rlds"
    "droid"
    "fractal20220817_data"
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds"
    "jaco_play"
    "kaist_nonprehensile_converted_externally_to_rlds"
    "nyu_door_opening_surprising_effectiveness"
    "nyu_rot_dataset_converted_externally_to_rlds"
    "robo_arena"
    "roboturk"
    "stanford_hydra_dataset_converted_externally_to_rlds"
    "taco_play"
    "tokyo_u_lsmo_converted_externally_to_rlds"
    "toto"
    "ucsd_kitchen_dataset_converted_externally_to_rlds"
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds"
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds"
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds"
    "utokyo_xarm_bimanual_converted_externally_to_rlds"
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds"
    "viola"
)

# ── Mode: set N_EPISODES=10 for test run, -1 for full conversion ──────────
N_EPISODES=-1

# ── Parallelism (MAX_IN_MEMORY must be divisible by N_WORKERS) ────────────
N_WORKERS=10
MAX_IN_MEMORY=100

# ── Pre-flight: verify all expected dataset dirs exist on disk ────────────
echo "Verifying dataset directories..."
MISSING=()
for SOURCE in "${DATASETS[@]}"; do
    if [ ! -d "${SNAPSHOT_DIR}/train/${SOURCE}" ]; then
        MISSING+=("${SOURCE}")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: missing expected dataset directories in ${SNAPSHOT_DIR}/train/:"
    for m in "${MISSING[@]}"; do echo "  MISSING: $m"; done
    exit 1
fi
echo "  All ${#DATASETS[@]} datasets found on disk."

echo ""
echo "Job ID:      $SLURM_JOB_ID"
echo "Snapshot:    $SNAPSHOT_DIR"
echo "Target:      $TARGET_DIR"
echo "N episodes:  $N_EPISODES  (set to -1 for full conversion)"
echo "N workers:   $N_WORKERS"
echo "N datasets:  ${#DATASETS[@]}"
echo "Start:       $(date)"
echo ""

FAILED=()

for SOURCE in "${DATASETS[@]}"; do
    DATASET_NAME="roboreward_${SOURCE}"
    echo "──────────────────────────────────────────────────────────────"
    echo "Processing: ${SOURCE}  →  ${DATASET_NAME}"
    echo "Time: $(date)"

    singularity run \
        -B "${WORK_DIR}:${WORK_DIR}" \
        -B "${RLDS_MOD_DIR}:/rlds_dataset_mod" \
        "${SINGULARITY_IMAGE}" \
        bash -c "
            export ROBOREWARD_SNAPSHOT_DIR='${SNAPSHOT_DIR}'
            export ROBOREWARD_SOURCE='${SOURCE}'
            export ROBOREWARD_N_WORKERS='${N_WORKERS}'
            export ROBOREWARD_MAX_IN_MEMORY='${MAX_IN_MEMORY}'
            export ROBOREWARD_N_EPISODES='${N_EPISODES}'
            python3 ${BUILDER_SCRIPT} ${TARGET_DIR}
        " && echo "  OK: ${DATASET_NAME}" \
          || { echo "  FAILED: ${DATASET_NAME}"; FAILED+=("${SOURCE}"); }
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "Done: $(date)"
echo ""
echo "Output datasets:"
ls -d "${TARGET_DIR}"/roboreward_* 2>/dev/null | while read d; do
    NAME=$(basename "$d")
    SHARDS=$(ls "$d"/1.0.0/*.tfrecord* 2>/dev/null | wc -l)
    echo "  ${NAME}  (${SHARDS} shards)"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: ${#FAILED[@]} dataset(s) failed:"
    for f in "${FAILED[@]}"; do echo "  $f"; done
    exit 1
fi
