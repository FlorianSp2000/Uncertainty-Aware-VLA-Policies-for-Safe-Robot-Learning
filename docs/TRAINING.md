# Training

- All training on the cluster, from the repo root: `sbatch slurm/<job>.sbatch`
- `external/V-GPS` is bind-mounted at `/V-GPS`; checkpoints → `external/V-GPS/results/<project>/<run>/`
- Logging: W&B, project set in each sbatch (`WANDB_API_KEY` from `.env`)
- Stack: JAX 0.4.20 / Flax 0.7.5 (`train_q.sif`), Octo/dlimp data loading on TFDS RLDS, `ml_collections` configs in `external/V-GPS/experiments/configs/`
- Agent: `jaxrl_m/agents/continuous/sarsa_ensemble.py` — per member a ResNet-34 + FiLM image encoder and MUSE instruction encoder; members over GPUs (`pmap`), 2 Q-heads per member (`vmap`)
- Relabeling (negative demonstrations): fraction ρ of each batch gets a swapped instruction + terminal penalty reward (`experiments/utils/finetuning.py:swap_batch_prompts_tf`)

## Main models (Bridge + Fractal)

| Thesis model | Job | Script / config | Settings |
|---|---|---|---|
| SayNo | `train-ensemble-q.sbatch` | `train_ensemble.py`, `train_ensemble_config.py:ensemble_sarsa` | 8 members, batch 1024, γ 0.98, seed 44 |
| SayNo-CR (counterfactual relabeling, ρ = 10 %) | `train-ensemble-neg-dem.sbatch` | same | as above + `negative_demo_ratio=0.10`, empty instructions excluded |

- Resume an interrupted run: `sbatch --export=ALL,RESUME_PATH=/V-GPS/results/VGPS_ensemble/<run> ...`
- Evaluation uses the latest checkpoint

## Classifier baselines (Bridge + Fractal)

| Thesis model | Job | Script / config |
|---|---|---|
| ResNet–MUSE 1, 2 (per trajectory), 3 (per sample) | `train-vgps-baseline.sbatch` | `train_vgps_baseline.py`, `train_classifier_config.py:binary_classifier` |
| Octo (per sample) | `finetune-octo-baseline.sbatch` | `finetune_octo_baseline.py`, Octo `scripts/configs/finetune_config.py` + `data_config.py:baseline_classifier` |

- All classifiers: 50 % swapped instructions, 100k steps
- Current script swaps **per sample** (ResNet–MUSE 3, Octo); ResNet–MUSE 1/2 used an earlier per-trajectory variant (`create_negative_demonstrations_classifier`, now validation only); 1 and 2 differ only by training run
- Octo: full finetune of `hf://rail-berkeley/octo-small-1.5` with a binary classification head (`octo/model/components/action_heads.py`), Bridge training stream

## IVA (RLBench) models

- All finetune the Bridge/Fractal SayNo checkpoint (no relabeling) on feasible (`tp`) IVA episodes, ρ = 0.10 relabeling
- Config `train_iva_config.py:ensemble_sarsa_finetune`; 7-D → 8-D action head adapted automatically (`train_iva_ensemble.py:detect_checkpoint_action_dim`)
- `RESUME_PATH` = that checkpoint, relative to `external/V-GPS` (e.g. `results/VGPS_ensemble/<run>`)

| Thesis model | Job + overrides |
|---|---|
| **IVA model** (no step reward, 4 members, 200k steps) | `sbatch --export=ALL,ARM=without_step_reward,STEP_REWARD=0.0,RESUME_PATH=... slurm/train-iva-step-reward-ablation.sbatch` |
| with step reward (−1 per step) | same, `ARM=with_step_reward,STEP_REWARD=-1.0` |
| seed replicate | same as IVA model, `ARM=without_step_reward_seed45,SEED=45` |
| reference (8 members, 50k steps) | `sbatch --export=ALL,RESUME_PATH=... slurm/train-iva-ensemble.sbatch` |
| reference without relabeling | as reference, `CREATE_NEGATIVE_DEMOS=false` in the sbatch |

- Step-reward jobs: 4 members, batch 512, 200k steps; the IVA loader keeps all frames in RAM → 28 GB per CPU requested

## VLA-rollout models (SAFE WidowX, LIBERO-10)

- Finetune the Bridge/Fractal SayNo checkpoint (no relabeling) on the **successful** `finetune` rollouts of one fold (DATA.md §2b); no failures in training; `holdout` successes → validation TD loss
- Script `train_real_robot_ensemble.py`, config presets `train_real_robot_config.py:vla_safe_widowx` / `vla_libero_pi0fast` (every knob in the preset), job `slurm/train-vla-rollouts.sbatch`
- Settings: 8 members (2 per GPU on 4 GPUs; packing checked against a per-member update before training), batch 1024, 50k steps, checkpoint every 5000, reward −1 per step, γ = 0.98, per-step gripper binarisation; WidowX: Bridge action statistics (OpenVLA acts in Bridge units), 3 terminal frames; LIBERO: bounds of the finetuning rollouts, 1 terminal frame
- Thesis checkpoint: step 50000; zero-shot comparison: the Bridge/Fractal checkpoint itself (step 200k), no finetuning

```bash
export BASE_RUN=results/VGPS_ensemble/<Bridge/Fractal SayNo run>
for FOLD in 0 1 2; do
  sbatch --export=ALL,PRESET=vla_safe_widowx,FOLD=$FOLD slurm/train-vla-rollouts.sbatch      # run name safe_fold<k>
  sbatch --export=ALL,PRESET=vla_libero_pi0fast,FOLD=$FOLD slurm/train-vla-rollouts.sbatch   # run name libero_fold<k>
done
```

## Appendix models

| Thesis section | Job | Script / config | Settings |
|---|---|---|---|
| Reward-penalty ablation | `train-reward-ablation.sbatch` (`--array=0-3`) | `ablations/train_reward_ablation.py`, `ensemble_sarsa` | 4 members, ρ = 5 %; array index = no relabeling / penalty −1 / −10 / −100; penalty −1 run twice |
| Contrastive regulariser | `train-ensemble-contrastive.sbatch` | `train_ensemble.py` | contrastive weight 1.0, negative ratio 0.45, paired |
| RoboReward value ensemble | `train-ensemble-roboreward.sbatch` (`roboreward.sif`) | `train_ensemble_roboreward.py`, `ensemble_sarsa_roboreward` | 4 members, batch 512 |
| SigLIP (π0) backbone | `train-ensemble-pi0.sbatch` (`train_pi0.sif`) | `train_ensemble_roboreward_siglip.py`, `ensemble_sarsa_roboreward_siglip` | frozen: `FREEZE_BACKBONE=True,BATCH_SIZE=256,LEARNING_RATE=3e-4`; finetuned: defaults |

- SigLIP + RoboReward jobs log object-removal AUROC during training (`experiments/utils/evaluate_inpainting_uncertainty.py`); they read the pickles in `external/V-GPS/experiments/data/` (DATA.md §5)
- SigLIP job: needs the π0 checkpoint (DATA.md §4) and mounts `external/openpi` at `/openpi`
