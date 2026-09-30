# Reproducing the thesis tables and figures

Pipeline: **checkpoints** ([TRAINING.md](TRAINING.md)) → **evaluation outputs** (§1, cluster GPU) → **tables + figures** (§2–4, laptop CPU) → **tests** (§5).

- Host commands: from the repo root, after `uv sync --extra analysis --extra plot` (+ `--extra clip` for §2a)
- `<...>` = your own checkpoint dirs / job ids
- Host file names below are the ones the analysis commands, figure registry and tests expect

## 1. Evaluation (cluster)

### 1a. Bridge/Fractal perturbation outputs

- Script `external/V-GPS/experiments/eval_perturbation_sensitivity.py`, job `slurm/eval-perturbation-sensitivity.sbatch`
- One HDF5 per (model, trajectory set): per-frame, per-member Q under
  - original instruction; relabeled instruction (random / nearest-neighbour / training instruction)
  - object removal + artefact controls (JPEG re-encoding; local / global composites, radius 8/16/32)
  - 7 action perturbations
- `MODELS="type:checkpoint:ensemble_size:batch_size:step:wandb_run"`, space separated; `type` ∈ `ensemble | classifier | octo` (sbatch header)
- First job draws the relabelings; every later job reuses them (`SWAPS_FROM_DIR=job_<first id>`) → identical inputs for all models

```bash
E=/V-GPS/results/VGPS_ensemble
# first: the two SayNo models (+ encoder features, qualitative sheets)
sbatch --export=ALL,MODELS="ensemble:$E/<sayno_no_relabeling>:8:1024:latest:- ensemble:$E/<sayno_relabeling>:8:1024:latest:-",DUMP_FEATURES=True,QUALITATIVE=True slurm/eval-perturbation-sensitivity.sbatch
# classifiers, Octo, reward-ablation ensembles: same relabelings
sbatch --export=ALL,MODELS="<entries>",SWAPS_FROM_DIR=job_<first id> slurm/eval-perturbation-sensitivity.sbatch
# training-split sets: encoder-novelty reference, seen/unseen object removal
sbatch --export=ALL,MODELS="<SayNo entries>",PKLS=/V-GPS/experiments/data/bridge_fractal_train_n200.pkl,DUMP_FEATURES=True slurm/eval-perturbation-sensitivity.sbatch
sbatch --export=ALL,MODELS="<SayNo entries>",PKLS=/V-GPS/experiments/data/bridge_fractal_train_n200_inpainted_cleaned.pkl slurm/eval-perturbation-sensitivity.sbatch
# critic latency; instruction of every training/validation trajectory (seen/unseen split)
sbatch --export=ALL,CKPT=$E/<sayno_no_relabeling> slurm/benchmark-critic-latency.sbatch
sbatch slurm/extract-instruction-vocabulary.sbatch
```

Copy to the host (`<set>` = `val_n200` | `n300_inpainted`):

| Host file | Content |
|---|---|
| `src/results/perturbation-sensitivity-bridge-fractal/controls/nonegdemo_<set>.hdf5` | SayNo |
| `.../controls/negdemo0.10_<set>.hdf5` | SayNo-CR (counterfactual relabeling, ρ = 10 %) |
| `.../controls/cls_resnet_muse_{1,2,3}_<set>.hdf5`, `.../controls/cls_octo_<set>.hdf5` | classifiers |
| `.../controls/abl_{no_relabeling,penalty1,penalty1_repeat,penalty10,penalty100}_<set>.hdf5` | reward ablation |
| `.../controls/{nonegdemo,negdemo0.10}_train_n200_feat.hdf5` | SayNo on `bridge_fractal_train_n200.pkl` |
| `.../controls/critic_latency_nonegdemo.json` | latency job output |
| `src/results/instruction-overlap-bridge-fractal/instructions_bridge_fractal_full.hdf5` | instruction job output |
| `src/results/instruction-overlap-bridge-fractal/dumps/{nonegdemo,negdemo0.10}_train_n200_inpainted.hdf5` | SayNo on `bridge_fractal_train_n200_inpainted_cleaned.pkl` |

### 1b. IVA three-way-prompt outputs

- Script `external/V-GPS/experiments/eval_iva_three_way_prompt.py`, job `slurm/eval-iva-three-way-prompt.sbatch`
- Scores every frame of the IVA evaluation `fp` split under: own instruction / IVA's false-premise injection / train-modal instruction of each of the 8 other tasks
- `P3_MODEL_DIR` relative to `external/V-GPS`

```bash
R=results/IVA_ensemble
sbatch --export=ALL,P3_MODEL_DIR=$R/<without_step_reward>,P3_STEP=200000,P3_TAG=without_step_reward_200k slurm/eval-iva-three-way-prompt.sbatch
sbatch --export=ALL,P3_MODEL_DIR=$R/<without_step_reward>,P3_STEP=50000,P3_TAG=without_step_reward_50k slurm/eval-iva-three-way-prompt.sbatch
sbatch --export=ALL,P3_MODEL_DIR=$R/<without_step_reward_seed45>,P3_STEP=50000,P3_TAG=without_step_reward_seed45_50k slurm/eval-iva-three-way-prompt.sbatch
sbatch --export=ALL,P3_MODEL_DIR=$R/<with_step_reward>,P3_STEP=200000,P3_TAG=with_step_reward_200k slurm/eval-iva-three-way-prompt.sbatch
sbatch --export=ALL,P3_MODEL_DIR=$R/<reference>,P3_STEP=50000,P3_TAG=reference_ens8_50k slurm/eval-iva-three-way-prompt.sbatch
sbatch --export=ALL,P3_MODEL_DIR=$R/<reference_no_relabeling>,P3_STEP=50000,P3_TAG=no_relabeling_ens8_50k slurm/eval-iva-three-way-prompt.sbatch
# method-overview figure: frames + scores of the reference model (images kept)
sbatch --export=ALL,IVA_MODEL_DIR=$R/<reference>,IVA_SPLITS=fp slurm/eval-iva-cp.sbatch
```

- Host: `src/data/iva_cp/three_way_prompt/<tag>.pkl` (from `external/V-GPS/results/iva_cp/<tag>/threewayprompt_step<step>_eval-fp_all.pkl`)
- Overview: `src/data/iva_cp/overview_fp_images.pkl` (from `evalcp_step50000_eval-fp_all.pkl`)

### 1c. VLA-rollout outputs (SAFE WidowX, LIBERO-10)

- Script `external/V-GPS/experiments/eval_real_robot.py`, job `slurm/eval-vla-rollouts.sbatch` (same preset as training): per frame, per member Q of (frame, instruction, executed action) for every rollout of pools `holdout`, `eval_seen`, `unseen` of one fold
- Checkpoints: every finetune checkpoint (5000 … 50000) + step 0 = the zero-shot Bridge/Fractal checkpoint (step 200000)
- `slurm/submit-vla-rollout-evals.sh`: submits one eval per saved checkpoint of every `{safe,libero}_fold<k>_*` run that has no dump yet (idempotent)

```bash
export BASE_RUN=results/VGPS_ensemble/<Bridge/Fractal SayNo run>
sbatch --export=ALL,PRESET=vla_safe_widowx,FOLD=0,RUN=safe_fold0_<timestamp>,STEPS="0 5000 50000" slurm/eval-vla-rollouts.sbatch
bash slurm/submit-vla-rollout-evals.sh          # or: all checkpoints of all runs
# raw actions per episode (action-extremeness baseline), from the full caches
python external/V-GPS/experiments/export_cache_actions.py --data_dir datasets/safe_widowx/processed_ft100_cal72_seed0 --out actions_raw.npz
python external/V-GPS/experiments/export_cache_actions.py --data_dir datasets/pi0_libero/processed_ft120_cal90_seed0 --out actions_raw.npz
```

- Cluster output: `external/V-GPS/results/vla_rollouts_v2/{safe,libero}/fold<k>/eval_{zeroshot,step<N>}_holdout-evalseen-unseen.pkl`
- Host: `src/results/vla-rollout-failure-detection/safe_protocol/{safe_widowx,libero_pi0fast}/fold<k>/dumps/` (same file names); `actions_raw.npz` → `src/results/vla-rollout-failure-detection/{safe_widowx,libero_pi0fast}/dumps/`

## 2. Bridge/Fractal tables (host)

### 2a. CLIP baseline scores

```bash
C=src/results/perturbation-sensitivity-bridge-fractal/controls
B=src/results/embedding-baselines-bridge-fractal
uv run python src/scripts/baselines/embedding_baselines.py --ref_pkl src/data/bridge_fractal_train_n200.pkl \
  --swap_pkl src/data/bridge_fractal_val_n200.pkl --inpaint_pkl src/data/bridge_fractal_n300_with_inpainted_cleaned.pkl \
  --swap_hdf5 $C/nonegdemo_val_n200.hdf5 --inpaint_hdf5 $C/nonegdemo_n300_inpainted.hdf5 \
  --clip_model openai/clip-vit-base-patch32 --clip_revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --k 5 --n_boot 2000 --n_examples 6 --cache $B/clip_embeddings_cache.npz --out_dir $B
```

### 2b. All `bf_*` tables + 3 AUROC figures (~15 min CPU)

- Tables: `bf_time_normalised_value_score`, `bf_ensemble_size_auroc`, `bf_object_removal_auroc_all_detectors`, `bf_object_removal_controls`, `bf_encoder_novelty_vs_disagreement`, `bf_latency`, `bf_classifier_variants_auroc`, `bf_relabeling_frame_auroc`, `bf_random_vs_nearest_relabeling_auroc`, `bf_action_perturbation_auroc`, `bf_reward_penalty_ablation_auroc`
- Figures: `auroc_initial_vs_final_frame`, `auroc_random_vs_nearest_relabeling`, `auroc_object_removal_vs_controls`; + `perturbation_thesis.json`

```bash
C=src/results/perturbation-sensitivity-bridge-fractal/controls
B=src/results/embedding-baselines-bridge-fractal
A='SayNo'; R='SayNo-CR'   # SayNo-CR = SayNo trained with counterfactual relabeling (rho = 10 %)
C1='ResNet--MUSE 1 (per trajectory)'; C2='ResNet--MUSE 2 (per trajectory)'; C3='ResNet--MUSE 3 (per sample)'; C4='Octo (per sample)'
uv run python src/scripts/analyse_perturbation_thesis.py \
 --val "$A=$C/nonegdemo_val_n200.hdf5" --inpainting "$A=$C/nonegdemo_n300_inpainted.hdf5" \
 --val "$R=$C/negdemo0.10_val_n200.hdf5" --inpainting "$R=$C/negdemo0.10_n300_inpainted.hdf5" \
 --val "$C1=$C/cls_resnet_muse_1_val_n200.hdf5" --inpainting "$C1=$C/cls_resnet_muse_1_n300_inpainted.hdf5" \
 --val "$C2=$C/cls_resnet_muse_2_val_n200.hdf5" --inpainting "$C2=$C/cls_resnet_muse_2_n300_inpainted.hdf5" \
 --val "$C3=$C/cls_resnet_muse_3_val_n200.hdf5" --inpainting "$C3=$C/cls_resnet_muse_3_n300_inpainted.hdf5" \
 --val "$C4=$C/cls_octo_val_n200.hdf5" --inpainting "$C4=$C/cls_octo_n300_inpainted.hdf5" \
 --val "No relabeling=$C/abl_no_relabeling_val_n200.hdf5" --inpainting "No relabeling=$C/abl_no_relabeling_n300_inpainted.hdf5" \
 --val 'Penalty $-1$'"=$C/abl_penalty1_val_n200.hdf5" --inpainting 'Penalty $-1$'"=$C/abl_penalty1_n300_inpainted.hdf5" \
 --val 'Penalty $-1$ (repeat run)'"=$C/abl_penalty1_repeat_val_n200.hdf5" --inpainting 'Penalty $-1$ (repeat run)'"=$C/abl_penalty1_repeat_n300_inpainted.hdf5" \
 --val 'Penalty $-10$'"=$C/abl_penalty10_val_n200.hdf5" --inpainting 'Penalty $-10$'"=$C/abl_penalty10_n300_inpainted.hdf5" \
 --val 'Penalty $-100$'"=$C/abl_penalty100_val_n200.hdf5" --inpainting 'Penalty $-100$'"=$C/abl_penalty100_n300_inpainted.hdf5" \
 --train_ref "$A=$C/nonegdemo_train_n200_feat.hdf5" --train_ref "$R=$C/negdemo0.10_train_n200_feat.hdf5" \
 --clip_scores "val200=$B/clip_scores_val_n200.hdf5" --clip_scores "inpainting258=$B/clip_scores_n300_inpainted.hdf5" --clip_label "CLIP ViT-B/32" \
 --main_ensembles "$A" "$R" --classifiers "$C1" "$C2" "$C3" "$C4" \
 --classifier_note "$C1=Bridge, Fractal" --classifier_note "$C2=Bridge, Fractal" \
 --classifier_note "$C3=Bridge, Fractal" --classifier_note "$C4=Bridge only" \
 --color "$A=sayno_no_relabeling" --color "$R=sayno_relabeling_rho10" --color "$C1=resnet_muse_classifier" \
 --color "$C2=resnet_muse_classifier" --color "$C3=resnet_muse_classifier" --color "$C4=octo_classifier" \
 --ablations "No relabeling" 'Penalty $-1$' 'Penalty $-1$ (repeat run)' 'Penalty $-10$' 'Penalty $-100$' \
 --reference_ensemble "$A" --latency $C/critic_latency_nonegdemo.json \
 --out_dir src/results/perturbation-sensitivity-bridge-fractal/analysis_thesis \
 --k 5 --n_boot 2000 --seed 0 --n_folds 2 --alphas 0.05 0.10 --table_alpha 0.1 --n_splits 200 --op_condition inpaint \
 --member_sizes 1 2 4 8 --n_subsets 50 --frame_readouts spread neg_mean_q \
 --actions act_far act_other act_zero act_neg act_gripper act_scaled act_uniform
```

- Wording-only changes (no bootstrap rerun): rewrite the `bf_*` tables from the JSON

```bash
uv run python src/scripts/write_perturbation_thesis_tables.py \
  --json src/results/perturbation-sensitivity-bridge-fractal/analysis_thesis/perturbation_thesis.json \
  --out_dir src/results/perturbation-sensitivity-bridge-fractal/analysis_thesis
```

### 2c. Seen vs. unseen tasks (`bf_seen_unseen_task_auroc`)

```bash
C=src/results/perturbation-sensitivity-bridge-fractal/controls; O=src/results/instruction-overlap-bridge-fractal
A='SayNo'; R='SayNo-CR'; C1='ResNet-MUSE 1'
uv run python src/scripts/instruction_overlap.py --walk $O/instructions_bridge_fractal_full.hdf5 \
 --val "$A=$C/nonegdemo_val_n200.hdf5" --inpainting "$A=$C/nonegdemo_n300_inpainted.hdf5" \
 --val "$R=$C/negdemo0.10_val_n200.hdf5" --inpainting "$R=$C/negdemo0.10_n300_inpainted.hdf5" \
 --val "$C1=$C/cls_resnet_muse_1_val_n200.hdf5" --inpainting "$C1=$C/cls_resnet_muse_1_n300_inpainted.hdf5" \
 --train "$A=$O/dumps/nonegdemo_train_n200_inpainted.hdf5" --train "$R=$O/dumps/negdemo0.10_train_n200_inpainted.hdf5" \
 --domain_models "$A" "$R" --table_out $O/bf_seen_unseen_task_auroc.tex \
 --pkl val=src/data/bridge_fractal_n300_with_inpainted_cleaned.pkl --pkl val=src/data/bridge_fractal_val_n200.pkl \
 --pkl train=src/data/bridge_fractal_train_n200.pkl \
 --n_boot 2000 --seed 0 --n_folds 2 --min_rows 10 --out_json $O/instruction_overlap.json
```

## 3. IVA tables (host)

- Tables: `iva_test_split_composition`, `iva_frame_auroc_reward_design`, `iva_frame_vs_episode_calibration`, `iva_episode_operating_points`, `iva_episode_statistic_comparison`, `iva_detection_delay`, `iva_per_task_operating_point`
- Conformal code: `src/cp/bands.py` (split CP, episode statistics, CUSUM), `src/cp/episode_eval.py` (calibration/test protocol), `src/cp/metrics.py` (AUROC, paired bootstrap CI)

```bash
D=src/data/iva_cp/three_way_prompt
# per-model AUROCs + operating points
uv run python src/scripts/cp/analyse_three_way_prompt.py \
  --dump "reference · sr -1 · negdemo 0.10 · ens8 · 50k=$D/reference_ens8_50k.pkl" \
  --dump "no-negdemo · sr -1 · ens8 · 50k=$D/no_relabeling_ens8_50k.pkl" \
  --dump "with_step_reward · sr -1 · negdemo 0.10 · ens4 · 200k=$D/with_step_reward_200k.pkl" \
  --dump "without_step_reward · sr 0 · negdemo 0.10 · ens4 · 200k=$D/without_step_reward_200k.pkl" \
  --alphas 0.10 --pool-tasks --out src/results/three-way-prompt-eval-iva/aurocs_and_operating_points_4checkpoints.json
# episode-level conformal detection
uv run python src/scripts/cp/analyse_episode_statistics.py --dump $D/without_step_reward_200k.pkl \
  --out-dir src/results/episode-operating-point-iva/thesis --alpha 0.05 --tau-alpha 0.05 --sweep-alphas 0.10,0.05,0.03 \
  --replicate $D/without_step_reward_50k.pkl --replicate $D/without_step_reward_seed45_50k.pkl \
  --three-way-json src/results/three-way-prompt-eval-iva/aurocs_and_operating_points_4checkpoints.json
# the 7 iva_* tables
uv run python src/scripts/cp/write_iva_thesis_tables.py \
  --stats-json src/results/episode-operating-point-iva/thesis/episode_statistics.json \
  --out-dir src/results/episode-operating-point-iva/thesis
```

### 3b. VLA-rollout tables

- Tables `vla_safe_detection`, `vla_libero_detection`, `vla_safe_folds`, `vla_safe_per_fold` (+ `results.{json,md}` per fold and view)
- Detectors: SayNo value score and ensemble disagreement; baselines gripper-open steps, action extremeness, OpenVLA token entropy (WidowX)
- Protocol: SAFE's functional conformal band (α = 0.15, 10 calibration splits) per fold; seen tasks = view B (calibration on `holdout`, test on `eval_seen`), unseen tasks = view A (SAFE's: calibration on `eval_seen` successes, test on `unseen`); LIBERO episodes cut at the shortest calibration success of their task; mean ± std over the 3 folds at step 50000

```bash
R=src/results/vla-rollout-failure-detection
uv run python src/scripts/run_vla_rollouts_safe_protocol.py --root $R/safe_protocol --datasets safe_widowx libero_pi0fast \
  --actions_npz safe_widowx=$R/safe_widowx/dumps/actions_raw.npz libero_pi0fast=$R/libero_pi0fast/dumps/actions_raw.npz \
  --policy_csv_dir src/data/safe_widowx/openvla_widowx --alpha 0.15 --n_boot 1000
uv run python src/scripts/vla_rollout_thesis_tables.py --root $R/safe_protocol \
  --safe_index src/data/safe_widowx/index_safe_protocol_fold{0,1,2}.json --checkpoint step50000 --alpha 0.15 \
  --out_dir $R/safe_protocol/thesis_tables
```

- Earlier single-split analysis (not in the thesis): `src/scripts/run_vla_rollout_analyses.sh`

## 4. Figures

- Registry `src/scripts/render_thesis_figures.py`: command, output file, include width of every figure

```bash
uv run python src/scripts/render_thesis_figures.py --dry-run         # list figures + commands
uv run python src/scripts/render_thesis_figures.py                   # draw all (needs §2–3 outputs)
uv run python src/scripts/render_thesis_figures.py --only sayno_overview.pdf --copy <dir>
```

- Data figures: `vla_detection_over_finetuning`, `vla_example_*` (§1c / 3b outputs + `src/data/safe_widowx/frames`), `auroc_*`, `bf_detector_score_violins`, `bf_seen_unseen_task_violins`, `object_removal_controls_examples`, `disagreement_original_vs_object_removal`, `iva_example_episode_three_instructions`, `sayno_overview`
- Training-curve figures (appendix): `baseline_*`, `contrastive_*`, `reward_penalty_critic_loss`, `roboreward_inpainting_auroc`
  - W&B API via `src/scripts/render_plot.py` + `src/scripts/plots/thesis_*.yaml`
  - set `run: <wandb-entity>/<project>/<run_id>` in each YAML to your training runs; `WANDB_API_KEY` in `.env`
- SigLIP appendix table: object-removal AUROC logged to W&B during SigLIP training (no separate script)

## 5. Tests

```bash
uv run pytest -q          # -m "not slow" skips tests loading > 0.5 GB files
```

- Thesis numbers recomputed from §1 outputs and pinned: `test_perturbation_numbers.py`, `test_instruction_overlap_numbers.py`, `test_episode_statistics_numbers.py`, `test_three_way_prompt_numbers.py`, `test_iva_thesis_tables.py` (each `.tex` = regeneration from its JSON)
- VLA rollouts: `test_vla_rollout_failure_detection_numbers.py` (pins §3b results), `test_vla_rollouts.py` (protocol code), `test_split_vla_rollouts.py` (fold splits)
- Score violins: `test_perturbation_score_violins.py`
- Method code on synthetic data: `test_metrics.py`, `test_episode_cp.py`, `test_embedding_baselines.py`, `test_iva_step_reward.py`
- IVA dataset statistics: `test_iva_injection_composition.py`, `test_iva_instruction_inventory.py`
- Figure style: `test_figstyle.py`, `test_figure_style_rules.py`
- Missing inputs → test skips with a message
