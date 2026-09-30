# Third-party code

This repository vendors code from the projects below. Each vendored file starts with a one-line comment that names its source, or its directory carries the original LICENSE. The last column says which files were changed for this thesis.

| Path | Origin | License | Changes |
|---|---|---|---|
| `external/V-GPS/` (`experiments/`, `jaxrl_m/`, `setup.py`, `requirements.txt`, `README.md`) | V-GPS, Nakamoto et al., CoRL 2024: https://github.com/nakamotoo/V-GPS | No license file upstream (see note) | Most of `experiments/` and the SARSA / classifier agents are new code for this thesis. Upstream files carry `# From V-GPS ...` with "Unmodified" or "Modified". |
| `external/V-GPS/jaxrl_m/` | jaxrl_m of bridge_data_v2: https://github.com/rail-berkeley/bridge_data_v2 (via V-GPS), derived from https://github.com/dibyaghosh/jaxrl_m | MIT | `agents/__init__.py`, `common/encoding.py`, `common/wandb.py`, `networks/actor_critic_nets.py`, `vision/__init__.py` modified. New: `agents/binary_classifier.py`, `agents/continuous/sarsa*.py`, `utils/siglip_weight_utils.py`, `vision/siglip_encoder.py` |
| `external/V-GPS/jaxrl_m/vision/bigvision_*.py` | big_vision: https://github.com/google-research/big_vision | Apache-2.0 (header kept) | unmodified |
| `external/V-GPS/jaxrl_m/vision/film_conditioning_layer.py` | robotics_transformer: https://github.com/google-research/robotics_transformer | Apache-2.0 | unmodified |
| `external/V-GPS/octo/` | Octo: https://github.com/octo-models/octo, via the V-GPS fork https://github.com/nakamotoo/octo | MIT (`octo/LICENSE`) | New: `octo/data/iva.py` (IVA loader). Modified: `octo/data/dataset.py`, `octo/data/oxe/*`, `octo/data/utils/data_utils.py`, `octo/model/components/action_heads.py`, `octo/model/octo_model.py`, `octo/utils/{train_callbacks,typing}.py`, `scripts/configs/finetune_config.py` |
| `external/openpi/` | openpi: https://github.com/Physical-Intelligence/openpi at commit `fdc03f5` | Apache-2.0 (`LICENSE`), Gemma terms (`LICENSE_GEMMA.txt`) | unmodified. Nested `third_party/{aloha,libero}` submodules omitted (unused). |
| `external/rlds_dataset_mod/` | https://github.com/kpertsch/rlds_dataset_mod, via the fork https://github.com/FlorianSp2000/rlds_dataset_mod | MIT (`LICENSE`) | New: `roboreward_builder/`, `convert_roboreward_all.sh` |

## Re-implemented methods (no code copied)

| Where | Method | Source |
|---|---|---|
| `src/cp/bands.py` (`functional_cp_band`, modulation, trajectory alignment) | functional conformal prediction band | FAIL-Detect, Xu et al. 2025 (https://github.com/CXU-TRI/FAIL-Detect, `UQ_test/timeseries_cp/methods/functional_predictor.py`); SAFE, Gu et al. 2025 (https://github.com/vla-safe/SAFE) |
| `src/cp/bands.py` (`zscore_cp_threshold`) | z-score conformal threshold | Dunn, Wasserman & Ramdas, arXiv:1809.07441 |
| `src/cp/bands.py` (`split_cp_threshold`, CUSUM) | split conformal quantile; Page's CUSUM | textbook |
| `src/cp/metrics.py` | rank-based AUROC / average precision | Mann–Whitney U; AP definition as in scikit-learn |
| `src/baselines/embedding.py` (`ledoit_wolf`) | Ledoit–Wolf shrinkage | re-implements `sklearn.covariance.ledoit_wolf_shrinkage` (BSD-3) |
| `src/utils/palette.py` | colour-blind-safe palette | Okabe & Ito (2008); Paul Tol |

## Note on V-GPS

The upstream V-GPS repository publishes no license file. Its `jaxrl_m` package descends from MIT-licensed bridge_data_v2, and its `octo` submodule is MIT-licensed. The V-GPS files are included here, credited in each file header, only so the thesis results can be reproduced. Rights remain with the original authors.
