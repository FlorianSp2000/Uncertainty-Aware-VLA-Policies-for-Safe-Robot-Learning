# Data

All paths are relative to the repository root. Submit every sbatch file from the repository root, because each job resolves its paths from `$(pwd)`. None of the data lives in git; `.gitignore` excludes it.

| What | Where it goes | Used by |
|---|---|---|
| BridgeData V2 + Fractal (RLDS, 256×256) | `datasets/open_x/{bridge_dataset/1.0.0,fractal20220817_data/0.1.0}` | all Bridge/Fractal training and evaluation |
| IVA (RLBench false-premise episodes) | `datasets/iva_dataset/{Train,Eval}` | IVA training and evaluation |
| SAFE / LIBERO VLA rollouts → rollout caches | `datasets/{safe_widowx,pi0_libero}/` | VLA-rollout experiment |
| RoboReward (HF → RLDS) | `datasets/roboreward/rlds` | appendix: RoboReward and SigLIP ensembles |
| π0 base checkpoint | `external/V-GPS/checkpoints/pi0_base` | appendix: SigLIP ensemble |
| Evaluation trajectory sets (pkl) | cluster: `external/V-GPS/experiments/data/`; host: `src/data/` | perturbation evaluation, figures |
| Evaluation dumps (hdf5/pkl) | host: `src/results/`, `src/data/iva_cp/` | all thesis tables and figures |

## 1. BridgeData V2 and Fractal

1. Build the download image once:

   ```bash
   singularity build oxe_download.sif containers/get-oxe.def
   ```

2. Submit the download job:

   ```bash
   sbatch slurm/get-oxe.sbatch   # ~1 day, CPU only
   ```

The job fetches the two datasets from different sources.
- **Fractal:** it downloads `fractal20220817_data` 0.1.0 from `gs://gresearch/robotics`. It then runs `external/rlds_dataset_mod/modify_rlds_dataset.py --mods=resize_and_jpeg_encode` to resize the images to 256×256 and JPEG-encode them.
- **Bridge:** it fetches BridgeData V2 as `bridge_dataset` 1.0.0, already at 256×256, from the Berkeley release (https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/). The data loader (`octo/data/oxe`) expects this dataset name. It does not use the OXE `bridge` 0.1.0 copy.

Both end up in `datasets/open_x/`. Train/val splits are the loader defaults of `external/V-GPS/experiments/configs/data_config.py`. Fractal has no validation split, so the loader holds out the last 5 % of train.

## 2. IVA (Instruct, Verify, Act)

The IVA dataset comes from Hsieh et al., *Do What? Teaching Vision-Language-Action Models to Reject the Impossible*, Findings of EMNLP 2025 ([project page](https://wen-hanhsieh.github.io/iva.github.io/), [arXiv:2508.16292](https://arxiv.org/abs/2508.16292)). At the time of writing the authors had not released it publicly. We obtained it from them. Place it as:

```
datasets/iva_dataset/
├── Train/iva_train_{tp,fp}.json   + {task}/episode*/{step}.png
└── Eval/iva_eval_{task}_{tp,fp}.json + {task}/episode*/{step}.png
```

`tp` files contain feasible episodes. `fp` files contain the same episodes with false-premise instructions injected at scattered steps. The loader is `external/V-GPS/octo/octo/data/iva.py`, and its docstring documents the JSON schema. `tests/test_iva_injection_composition.py` and `tests/test_iva_instruction_inventory.py` check the dataset statistics quoted in the thesis. They skip when the data is absent.

## 2b. VLA rollouts (SAFE OpenVLA-WidowX, pi0-FAST LIBERO-10)

- **WidowX:** `openvla_widowx.zip` released with SAFE ([download](https://drive.google.com/file/d/1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp/view?usp=sharing), 1.4 GB; 532 rollouts, 8 tasks, 50 steps at 5 Hz) → unzip to `datasets/safe_widowx/openvla_widowx/`
- **LIBERO-10:** HF `oldTOM/pi0-libero-rollouts`, `pi0fast-libero_10.zip` (16.7 GB; 306 successes / 194 failures) → unzip to `datasets/pi0_libero/pi0fast-libero_10/`
- Convert to per-episode caches (PNG frames, raw actions, prompts, `index.json`); run with `bash`, not `sbatch`; image `train_q_ood.sif` (SAFE pickles need torch)
- Re-split each cache by SAFE's seen / unseen-task protocol: 3 folds, per fold 2 of 8 (WidowX) / 3 of 10 (LIBERO) tasks unseen; seen tasks → `finetune` / `holdout` (successes only) / `eval_seen`; unseen tasks → `unseen`. Script `external/V-GPS/experiments/split_vla_rollouts.py`; `.npz` files symlinked, no reconversion

```bash
singularity build train_q_ood.sif containers/train-q-ood.def
bash slurm/convert-safe-rollouts.sbatch                       # -> datasets/safe_widowx/processed_ft100_cal72_seed0
bash slurm/convert-libero-rollouts.sbatch                     # -> datasets/pi0_libero/processed_ft120_cal90_seed0
DATASET=safe_widowx    bash slurm/split-vla-rollouts.sbatch   # -> datasets/safe_widowx/safe_protocol_fold{0,1,2}
DATASET=libero_pi0fast bash slurm/split-vla-rollouts.sbatch   # -> datasets/pi0_libero/safe_protocol_fold{0,1,2}
```

- Host copies for the analysis:
  - `src/data/safe_widowx/openvla_widowx/` — SAFE's per-rollout CSVs (token-entropy baseline)
  - `src/data/safe_widowx/index_safe_protocol_fold{0,1,2}.json` — `index.json` of the three WidowX folds (task / split-size tables)
  - `src/data/safe_widowx/frames/<key>.npz` — cache entries of the example episodes (example figures)

## 3. RoboReward (appendix only)

```bash
singularity build roboreward.sif containers/roboreward.def
sbatch slurm/get-roboreward.sbatch      # HF teetone/RoboReward -> datasets/roboreward (HF cache layout)
sbatch slurm/convert_roboreward.sbatch  # MP4 -> RLDS per source dataset -> datasets/roboreward/rlds
```

The converter is `external/rlds_dataset_mod/roboreward_builder/roboreward_dataset_builder.py`. `SNAPSHOT_DIR` in the sbatch pins the HF revision used in the thesis.

## 4. π0 weights (appendix only, SigLIP backbone)

```bash
sbatch slurm/get-pi0-weights.sbatch     # gs://openpi-assets/checkpoints/pi0_base -> external/V-GPS/checkpoints/pi0_base
```

## 5. Evaluation trajectory sets (Bridge/Fractal)

Each pickle holds first and last frame, actions and instruction for a fixed set of trajectories: half Bridge, half Fractal, taken in deterministic loader order with seed 44.

| File | Split | Role in the thesis |
|---|---|---|
| `bridge_fractal_val_n200.pkl` | val | instruction relabeling, action perturbations |
| `bridge_fractal_n300_with_inpainted_cleaned.pkl` | val | object removal and artefact controls: 258 reviewed pairs. It is a superset of the 200 above. |
| `bridge_fractal_train_n200.pkl` | train | reference set for encoder novelty and CLIP novelty; seen/unseen split |
| `bridge_fractal_train_n200_inpainted_cleaned.pkl` | train | seen-task object removal (seen/unseen table); SigLIP / RoboReward training-time evaluation |

- Pickles go to both `external/V-GPS/experiments/data/` (cluster jobs) and `src/data/` (host analysis)
- Object-removal images were edited by hand (external image editor) → a rebuild gives a comparable, not bit-identical set

**To rebuild them from the RLDS data** (inside `train_q.sif`, `external/V-GPS` mounted at `/V-GPS`):

```bash
python /V-GPS/experiments/build_eval_sets.py --split val   --n 300 --out /V-GPS/experiments/data/bridge_fractal_n300.pkl
python /V-GPS/experiments/build_eval_sets.py --split val   --n 200 --out /V-GPS/experiments/data/bridge_fractal_val_n200.pkl
python /V-GPS/experiments/build_eval_sets.py --split train --n 200 --out /V-GPS/experiments/data/bridge_fractal_train_n200.pkl
```

Then add the object-removal images, which remove the instruction's target object from the first and last frame:

- **n300 (val):** `src/magic_create_inpaintings.ipynb`.
  1. Export the frames as PNG.
  2. Edit them with Google Photos Magic Editor.
  3. Merge the edited images back as `first_image_inpainted` and `last_image_inpainted`.
- **train n200:** `src/gemini_create_inpaintings.ipynb`. This uses the Gemini image-edit API (`gemini-2.5-flash-image-preview`), with prompts built by `src/utils/prompting.py`. It needs `GEMINI_API_KEY` in `.env`.
- **Review:** `src/review_inpaintings.ipynb`. Flag failed edits per frame. This writes the `*_cleaned.pkl` files with `failed_ids` and `failed_records`. The analysis keeps only reviewed pairs, which gives 258 of 300 for the n300 set.

## 6. Evaluation outputs

- Every thesis number is computed on the host from the evaluation outputs of the cluster jobs → [REPRODUCE.md](REPRODUCE.md) §1 (file names + locations)
