# Vendored code

Copied into this repository, not linked as git submodules, so that one clone holds everything needed. Each directory keeps its upstream README and LICENSE. Files changed for the thesis carry a one-line source comment; see [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

| Directory | Based on | Upstream code | License |
|---|---|---|---|
| `V-GPS/` (training/eval code base) | V-GPS, Nakamoto et al., CoRL 2024 | https://github.com/nakamotoo/V-GPS | none published upstream |
| `V-GPS/jaxrl_m/` | jaxrl_m of BridgeData V2, derived from jaxrl_m | https://github.com/rail-berkeley/bridge_data_v2 · https://github.com/dibyaghosh/jaxrl_m | MIT |
| `V-GPS/jaxrl_m/vision/bigvision_*.py` | big_vision | https://github.com/google-research/big_vision | Apache-2.0 |
| `V-GPS/jaxrl_m/vision/film_conditioning_layer.py` | RT-1 FiLM layer | https://github.com/google-research/robotics_transformer | Apache-2.0 |
| `V-GPS/octo/` | Octo, via the V-GPS fork | https://github.com/octo-models/octo · https://github.com/nakamotoo/octo | MIT |
| `openpi/` (SigLIP module, appendix) | openpi, commit `fdc03f5` | https://github.com/Physical-Intelligence/openpi | Apache-2.0 + Gemma terms |
| `rlds_dataset_mod/` (dataset download / resize, RoboReward → RLDS) | rlds_dataset_mod, via the fork with the RoboReward converter | https://github.com/kpertsch/rlds_dataset_mod · https://github.com/FlorianSp2000/rlds_dataset_mod | MIT |

Installed as dependencies, not vendored: dlimp (https://github.com/kvablack/dlimp, pinned commit in `V-GPS/requirements.txt`), JAX, Flax, TensorFlow Datasets.
