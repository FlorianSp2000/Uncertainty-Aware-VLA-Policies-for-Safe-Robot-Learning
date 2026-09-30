#!/bin/bash
# Convert all 29 RoboReward sub-datasets from HuggingFace snapshot format to RLDS/TFDS.
#
# Prerequisites:
#   pip install opencv-python tensorflow tensorflow_datasets
#
# The RoboReward HuggingFace snapshot must be downloaded first:
#   huggingface-cli download teetone/RoboReward --repo-type dataset --local-dir <dir>
#   (or via `datasets` library with streaming=False)
#
# Usage:
#   SNAPSHOT_DIR=/path/to/snapshot TARGET_DIR=/path/to/output bash convert_roboreward_all.sh
#
# Key env vars (all have defaults below):
#   SNAPSHOT_DIR      — HF snapshot dir containing train/val/test subdirs + metadata.jsonl
#   TARGET_DIR        — output directory for RLDS tfrecords
#   N_EPISODES        — episodes per dataset (-1 = all, 10 = quick test)
#   N_WORKERS         — parallel workers (default 10)
#   MAX_IN_MEMORY     — episodes buffered before writing; must be divisible by N_WORKERS

set -e
ulimit -n 20000  # needed for large datasets (many temp tfrecord shards)

# ── Paths (override via env) ────────────────────────────────────────────────
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$HOME/.cache/huggingface/hub/datasets--teetone--RoboReward/snapshots/469b9af76e8539e8d2ac553b081307738fd92ca5}"
TARGET_DIR="${TARGET_DIR:-$HOME/datasets/roboreward/rlds}"
BUILDER_SCRIPT="$(dirname "$(realpath "$0")")/roboreward_builder/roboreward_dataset_builder.py"

# ── Parallelism ─────────────────────────────────────────────────────────────
N_WORKERS="${N_WORKERS:-10}"
MAX_IN_MEMORY="${MAX_IN_MEMORY:-100}"

# ── Mode: -1 = full conversion, 10 = quick test ─────────────────────────────
N_EPISODES="${N_EPISODES:--1}"

# ── All 29 source datasets (exact subdirectory names in snapshot/train/) ────
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

mkdir -p "${TARGET_DIR}"

# ── Pre-flight: verify all expected dataset dirs exist ──────────────────────
echo "Verifying dataset directories in ${SNAPSHOT_DIR}/train/ ..."
MISSING=()
for SOURCE in "${DATASETS[@]}"; do
    [ ! -d "${SNAPSHOT_DIR}/train/${SOURCE}" ] && MISSING+=("${SOURCE}")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: missing directories:"
    for m in "${MISSING[@]}"; do echo "  MISSING: $m"; done
    exit 1
fi
echo "  All ${#DATASETS[@]} datasets found."

echo ""
echo "Snapshot:   ${SNAPSHOT_DIR}"
echo "Target:     ${TARGET_DIR}"
echo "N episodes: ${N_EPISODES}  (-1 = all)"
echo "N workers:  ${N_WORKERS}"
echo "Start:      $(date)"
echo ""

FAILED=()

for SOURCE in "${DATASETS[@]}"; do
    echo "──────────────────────────────────────────────────────────────"
    echo "Processing: ${SOURCE}  →  roboreward_${SOURCE}"
    echo "Time: $(date)"

    ROBOREWARD_SNAPSHOT_DIR="${SNAPSHOT_DIR}" \
    ROBOREWARD_SOURCE="${SOURCE}" \
    ROBOREWARD_N_WORKERS="${N_WORKERS}" \
    ROBOREWARD_MAX_IN_MEMORY="${MAX_IN_MEMORY}" \
    ROBOREWARD_N_EPISODES="${N_EPISODES}" \
        python3 "${BUILDER_SCRIPT}" "${TARGET_DIR}" \
        && echo "  OK: roboreward_${SOURCE}" \
        || { echo "  FAILED: ${SOURCE}"; FAILED+=("${SOURCE}"); }
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
