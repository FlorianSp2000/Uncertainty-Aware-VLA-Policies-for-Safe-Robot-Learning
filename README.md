# Uncertainty-Aware VLA Policies for Safe Robot Learning

Code for the MSc thesis *Uncertainty-Aware Vision-Language-Action Policies for Safe Robot Learning* (Florian Sprick, University of Tübingen, 2026).

**SayNo** = ensemble of language-conditioned SARSA critics; flags a scene as infeasible when members disagree on Q(s, a, instruction).

- **Offline perturbation (Bridge V2 + Fractal):** frame-level AUROC under instruction relabeling, object removal (+ artefact controls), action perturbation; vs. classifier and CLIP baselines → `bf_*` tables, AUROC figures
- **False-premise episodes (IVA / RLBench, 9 tasks):** episode-level conformal detection → `iva_*` tables, example-episode figure
- **VLA rollouts (SAFE OpenVLA-WidowX, pi0-FAST LIBERO-10):** SayNo finetuned on successful rollouts, conformal failure detection on seen and unseen tasks (SAFE protocol, 3 folds) vs. gripper / action / token-entropy baselines → `vla_*` tables and figures
- **Appendix:** reward-penalty ablation, SigLIP (π0) backbone, RoboReward value ensemble, contrastive regulariser (W&B training curves)

## Layout

```
external/V-GPS/            JAX training + evaluation (vendored V-GPS; see THIRD_PARTY_NOTICES.md)
  experiments/             train_*.py, eval_*.py, build_eval_sets.py, configs/, utils/
  jaxrl_m/                 agents (SARSA ensemble: agents/continuous/sarsa_ensemble.py), encoders
  octo/                    data loaders (+ octo/data/iva.py), Octo baseline
external/rlds_dataset_mod/ dataset download / resize, RoboReward -> RLDS (vendored)
external/openpi/           pi0 SigLIP module, appendix only (vendored)
containers/                Singularity definitions + pinned package versions
slurm/                     SLURM jobs: download, training, evaluation
src/                       host-side analysis package `say_no`: conformal prediction, AUROC, tables, figures
  scripts/                 entry points writing every thesis table and figure
tests/                     unit tests + tests pinning thesis numbers to their evaluation outputs
docs/                      DATA.md, TRAINING.md, REPRODUCE.md
```

## Quick setup

**Host (analysis, CPU)** — Python ≥ 3.11, [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/FlorianSp2000/Uncertainty-Aware-VLA-Policies-for-Safe-Robot-Learning.git
cd Uncertainty-Aware-VLA-Policies-for-Safe-Robot-Learning
uv sync --extra analysis --extra plot   # + --extra clip for the CLIP baseline
uv run pytest -q                        # unit tests pass; number tests skip without evaluation outputs
```

**Cluster (training + evaluation, GPU, Singularity)** — run everything from the repo root:

```bash
singularity build train_q.sif      containers/train-q.def      # SayNo, classifiers, IVA
singularity build oxe_download.sif containers/get-oxe.def      # dataset download
singularity build train_q_ood.sif  containers/train-q-ood.def  # + torch: SAFE / LIBERO rollout conversion
singularity build roboreward.sif   containers/roboreward.def   # appendix: RoboReward
singularity build train_pi0.sif    containers/train-pi0.def    # appendix: SigLIP / pi0
cp .env.example .env                                           # W&B key; sourced by every sbatch
sbatch slurm/<job>.sbatch
```

- `#SBATCH` headers: `L40S` GPUs, partition `L40Sday` (Tübingen cluster) → adapt to your cluster
- Checkpoints, datasets and evaluation outputs are **not** part of this repository

## Reproduce in 4 steps

1. **Data** → [docs/DATA.md](docs/DATA.md): Bridge/Fractal, IVA, SAFE / LIBERO rollouts, RoboReward, evaluation trajectory sets with object-removal images ([download](https://drive.google.com/file/d/15a_hTGGLPi-Y_cSmMMrG1QxOi_4dmRgZ/view?usp=sharing), 373 MB)
2. **Train** → [docs/TRAINING.md](docs/TRAINING.md): one sbatch per thesis model, key settings
3. **Evaluate** (cluster) → [docs/REPRODUCE.md §1](docs/REPRODUCE.md#1-evaluation-cluster): checkpoints → evaluation outputs
4. **Analyse** (laptop) → [docs/REPRODUCE.md §2–5](docs/REPRODUCE.md#2-bridgefractal-tables-host): evaluation outputs → every table, figure, test

## Third-party code and references

- Origin + license per part: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), per directory: [external/README.md](external/README.md); vendored files carry a one-line source comment

**Code this repository is based on**

| Code base | Used for | Repository |
|---|---|---|
| V-GPS | training / evaluation code base (`external/V-GPS`) | [nakamotoo/V-GPS](https://github.com/nakamotoo/V-GPS) |
| jaxrl_m (BridgeData V2) | agents, encoders (`external/V-GPS/jaxrl_m`) | [rail-berkeley/bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2), originally [dibyaghosh/jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) |
| Octo | data loaders, Octo baseline (`external/V-GPS/octo`) | [octo-models/octo](https://github.com/octo-models/octo), via [nakamotoo/octo](https://github.com/nakamotoo/octo) |
| rlds_dataset_mod | dataset download / resize, RoboReward → RLDS (`external/rlds_dataset_mod`) | [kpertsch/rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod), fork [FlorianSp2000/rlds_dataset_mod](https://github.com/FlorianSp2000/rlds_dataset_mod) |
| openpi | π0 SigLIP vision tower (`external/openpi`) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |
| big_vision, robotics_transformer | ResNet-v2 and FiLM layers in `jaxrl_m/vision` | [google-research/big_vision](https://github.com/google-research/big_vision), [google-research/robotics_transformer](https://github.com/google-research/robotics_transformer) |
| dlimp | RLDS data pipeline (dependency) | [kvablack/dlimp](https://github.com/kvablack/dlimp) |
| SAFE, FAIL-Detect | conformal detection protocol, re-implemented in `src/cp`, `src/vla_rollouts.py` | [vla-safe/SAFE](https://github.com/vla-safe/SAFE), [CXU-TRI/FAIL-Detect](https://github.com/CXU-TRI/FAIL-Detect) |

**Code base** — training/evaluation code built on the V-GPS code base ([nakamotoo/V-GPS](https://github.com/nakamotoo/V-GPS)):
- M. Nakamoto, O. Mees, A. Kumar, S. Levine. *Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance.* CoRL 2024 (PMLR 270). [paper](https://proceedings.mlr.press/v270/nakamoto25a.html)

**Vendored models / tools**
- jaxrl_m, via BridgeData V2 ([rail-berkeley/bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2)): H. Walke et al. *BridgeData V2: A Dataset for Robot Learning at Scale.* CoRL 2023 (PMLR 229).
- Octo ([octo-models/octo](https://github.com/octo-models/octo)): D. Ghosh, H. Walke, K. Pertsch, et al. *Octo: An Open-Source Generalist Robot Policy.* RSS 2024. doi:10.15607/RSS.2024.XX.090
- π0 / openpi ([Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)): K. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* RSS 2025.
- rlds_dataset_mod ([kpertsch/rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod)) for the Open X-Embodiment data: Open X-Embodiment Collaboration. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models.* ICRA 2024. doi:10.1109/ICRA57147.2024.10611477

**Re-implemented methods** (no code copied)
- Functional conformal band, SAFE protocol: Q. Gu, Y. Ju, S. Sun, I. Gilitschenski, H. Nishimura, M. Itkina, F. Shkurti. *SAFE: Multitask Failure Detection for Vision-Language-Action Models.* NeurIPS 2025. [code](https://github.com/vla-safe/SAFE)
- Functional conformal band: C. Xu et al. *Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies.* RSS 2025. doi:10.15607/RSS.2025.XXI.073 [code](https://github.com/CXU-TRI/FAIL-Detect)

**Datasets**
- VLA rollouts released with SAFE (above): OpenVLA on WidowX ([download](https://drive.google.com/file/d/1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp/view?usp=sharing)); pi0-FAST on LIBERO-10 (HF `oldTOM/pi0-libero-rollouts`), collected by Foresight: H. Zhang et al. *Foresight: Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents.* arXiv:2606.23085, 2026. OpenVLA: M. J. Kim et al. *OpenVLA: An Open-Source Vision-Language-Action Model.* CoRL 2024. LIBERO: B. Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.* NeurIPS 2023 Datasets and Benchmarks.
- BridgeData V2: Walke et al., CoRL 2023 (above); Fractal / RT-1 via Open X-Embodiment (above)
- IVA: W.-H. Hsieh, E. Hsieh, D. Niu, T. Darrell, R. Herzig, D. M. Chan. *Do What? Teaching Vision-Language-Action Models to Reject the Impossible.* Findings of EMNLP 2025. doi:10.18653/v1/2025.findings-emnlp.635 — not publicly released; obtained from the authors on request (see docs/DATA.md §2).
- RoboReward: T. Lee, A. Wagenmaker, K. Pertsch, P. Liang, S. Levine, C. Finn. *RoboReward: General-Purpose Vision-Language Reward Models for Robotics.* arXiv:2601.00675, 2026.

**Baselines / encoders**
- CLIP: A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
- MUSE (instruction encoder): Y. Yang et al. *Multilingual Universal Sentence Encoder for Semantic Retrieval.* ACL 2020 (System Demonstrations).
