# From V-GPS (https://github.com/nakamotoo/V-GPS), jaxrl_m of bridge_data_v2 (https://github.com/rail-berkeley/bridge_data_v2), MIT. Modified for this thesis.
from jaxrl_m.vision.bigvision_resnetv2 import resnetv2_configs
from jaxrl_m.vision.impala import impala_configs
from jaxrl_m.vision.resnet_v1 import resnetv1_configs
from jaxrl_m.vision.small_encoders import small_configs

encoders = dict()
encoders.update(impala_configs)
encoders.update(resnetv2_configs)
encoders.update(resnetv1_configs)
encoders.update(small_configs)

# SigLIP encoder (requires openpi on PYTHONPATH — only available in train_pi0.sif)
try:
    from jaxrl_m.vision.siglip_encoder import siglip_configs
    encoders.update(siglip_configs)
except ImportError:
    import logging as _log
    _log.getLogger(__name__).debug("SigLIP encoder unavailable (openpi not on PYTHONPATH)")
