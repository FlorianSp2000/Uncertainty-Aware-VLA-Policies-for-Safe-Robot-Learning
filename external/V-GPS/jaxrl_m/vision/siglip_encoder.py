# Wraps the SigLIP module of openpi (https://github.com/Physical-Intelligence/openpi, Apache-2.0).
"""
SigLIP So400m/14 vision encoder wrapper for SARSA ensemble Q-learning.

Wraps openpi's Flax Linen siglip._Module to match the V-GPS encoder interface:
    __call__(self, observations, train=False, cond_var=None) -> jnp.ndarray

The encoder outputs a (batch, 1152) GAP-pooled vector from the SigLIP ViT.
Language conditioning (cond_var) is accepted but ignored — SigLIP has no FiLM;
fusion happens in SigLIPLCEncodingWrapper instead.
"""
import functools

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from openpi.models import siglip

SIGLIP_INPUT_SIZE = 224  # So400m/14: 224/14=16 patches, pos_embedding=(1,256,1152)


class SigLIPEncoder(nn.Module):
    """Thin wrapper around openpi's SigLIP ViT for the V-GPS encoder registry."""

    variant: str = "So400m/14"
    pool_type: str = "gap"
    scan: bool = True
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False, cond_var=None) -> jnp.ndarray:
        # cond_var accepted for interface compat but unused — SigLIP has no FiLM
        chex.assert_rank(observations, 4)  # (B, H, W, C)

        image = observations.astype(jnp.float32)
        # Data pipeline delivers uint8 [0,255]; SigLIP expects float32 [-1,1]
        image = image / 127.5 - 1.0

        # Resize to 224x224 if input is different (e.g. inpainting eval images at 256x256)
        h, w = image.shape[1], image.shape[2]
        if h != SIGLIP_INPUT_SIZE or w != SIGLIP_INPUT_SIZE:
            image = jax.image.resize(image, (image.shape[0], SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE, 3), method="bilinear")

        x, _ = siglip.Module(
            num_classes=None,       # skip head layer — we want pre-logits only
            variant=self.variant,
            pool_type=self.pool_type,
            scan=self.scan,
            dtype_mm=self.dtype_mm,
            name="siglip",          # deterministic param path (avoids auto _Module_0)
        )(image, train=train)

        chex.assert_shape(x, (None, 1152))  # So400m output dim
        return x


# Encoder registry entry
siglip_configs = {
    "siglip-so400m-14": functools.partial(SigLIPEncoder),
}
