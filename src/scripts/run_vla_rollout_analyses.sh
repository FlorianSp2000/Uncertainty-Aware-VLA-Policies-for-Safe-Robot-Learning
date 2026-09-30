#!/bin/bash
# All VLA-rollout analyses of the thesis (SAFE OpenVLA-WidowX + pi0-FAST LIBERO-10), then the four vla_* tables.
# Inputs: eval dumps in $R/{safe_widowx,libero_pi0fast}/dumps/ (slurm/eval-real-robot.sbatch), raw actions
# ($R/*/dumps/actions_raw.npz, experiments/export_cache_actions.py), SAFE's per-rollout CSVs
# (src/data/safe_widowx/openvla_widowx, token-entropy baseline) and the split index of seed 0.
# Run from the repo root:  bash src/scripts/run_vla_rollout_analyses.sh   (OUT=<dir> writes elsewhere)
set -e
PY=${PY:-uv run python}
R=src/results/vla-rollout-failure-detection
OUT=${OUT:-$R}
S=$R/safe_widowx/dumps
L=$R/libero_pi0fast/dumps
A="$PY src/scripts/analyse_vla_rollout_failure_detection.py --alpha 0.1 --split_frac 0.5 --n_split_seeds 10 --n_boot 2000 --discount 0.98"
SAFE="--policy_csv_dir src/data/safe_widowx/openvla_widowx --actions_npz $S/actions_raw.npz --num_terminal 3"
LIB="--actions_npz $L/actions_raw.npz --num_terminal 1"

ALL_TASKS="lift_aaa_battery lift_blue_cup lift_eggplant lift_red_bottle put_blue_cup_on_plate put_the_carrot_on_plate put_the_red_block_into_the_pot put_the_red_bottle_into_pot"
complement() {  # tasks not in the comma-separated list $1, comma-separated
  local held=",$1," out=""
  for t in $ALL_TASKS; do [[ $held == *",$t,"* ]] || out="$out,$t"; done
  echo "${out#,}"
}

# --- WidowX, all tasks seen: split seeds 0 and 1
FT=""; for s in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000; do FT="$FT ft$s=$S/ft100_step${s}_calib-test.pkl"; done
$A $SAFE --dataset "SAFE OpenVLA-WidowX" --out_dir $OUT/safe_widowx/split0 \
  --dump zeroshot=$S/zeroshot_bridgefractal_step200000_calib-test.pkl $FT \
  --swap zeroshot=$S/zeroshot_bridgefractal_step200000_test_swapped.pkl ft5000=$S/ft100_step5000_test_swapped.pkl
$A $SAFE --dataset "SAFE OpenVLA-WidowX, split seed 1" --out_dir $OUT/safe_widowx/split1 \
  --dump zeroshot=$S/split1_zeroshot_step200000_calib-test.pkl ft2500=$S/split1_ft100_step2500_calib-test.pkl \
         ft5000=$S/split1_ft100_step5000_calib-test.pkl \
  --swap ft5000=$S/split1_ft100_step5000_test_swapped.pkl

# --- WidowX, two tasks held out of finetuning + calibration: draws A, B, C, test set split into seen / unseen tasks
heldout() {  # name prefix, dataset label (%s = seen/unseen), held-out tasks, dump prefix, extra dumps
  local out=$1 label=$2 held=$3 prefix=$4; shift 4
  local d="zeroshot=$S/${prefix}_zeroshot_step200000_calib-test.pkl $*"
  $A $SAFE --dataset "${label//%s/seen}" --out_dir $OUT/safe_widowx/${out}_seen --test_tasks "$(complement $held)" --dump $d
  $A $SAFE --dataset "${label//%s/unseen}" --out_dir $OUT/safe_widowx/${out}_unseen --test_tasks "$held" --dump $d
}
heldout heldout_tasks "SAFE OpenVLA-WidowX, %s tasks (2 tasks held out of finetune+calibration)" \
  put_the_carrot_on_plate,put_the_red_bottle_into_pot unseen \
  ft2500=$S/unseen_ft75_step2500_calib-test.pkl ft5000=$S/unseen_ft75_step5000_calib-test.pkl
heldout heldout_tasks_B "SAFE WidowX draw B, %s tasks" lift_red_bottle,put_blue_cup_on_plate unseen-redbottle-bluecupplate \
  ft5000=$S/unseen-redbottle-bluecupplate_ft75_step5000_calib-test.pkl
heldout heldout_tasks_C "SAFE WidowX draw C, %s tasks" lift_eggplant,put_the_carrot_on_plate unseen-eggplant-carrot \
  ft5000=$S/unseen-eggplant-carrot_ft75_step5000_calib-test.pkl

# --- LIBERO-10: scored on the common prefix (min_calib), per task (min_calib_task) and in full (none)
FT=""; for s in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000; do FT="$FT ft$s=$L/causal_libft120_step${s}_calib-test.pkl"; done
for s in 7500 10000 12500 15000; do FT="$FT ft$s=$L/causal_libcont_step${s}_calib-test.pkl"; done
for h in min_calib min_calib_task none; do
  $A $LIB --dataset "pi0-FAST LIBERO-10" --out_dir $OUT/libero_pi0fast/prefix_$h --horizon $h \
    --dump zeroshot=$L/causal_zeroshot_step200000_calib-test.pkl $FT \
    --swap zeroshot=$L/causal_zeroshot_step200000_test_swapped.pkl ft5000=$L/causal_libft120_step5000_test_swapped.pkl \
           ft15000=$L/causal_libcont_step15000_test_swapped.pkl
done

# --- thesis tables vla_safe_tasks, vla_safe_detection, vla_safe_per_test_set, vla_libero_prefix_auroc
$PY src/scripts/vla_rollout_thesis_tables.py --safe_dir $OUT/safe_widowx \
  --libero_results $OUT/libero_pi0fast/prefix_min_calib/results.json \
  --safe_index src/data/safe_widowx/index_processed_ft100_cal72_seed0.json --out_dir $OUT/thesis_tables
