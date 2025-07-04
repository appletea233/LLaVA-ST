import torch

from .masked_drop import MaskedDrop
from .spatial_pool import SpatialPool
from .perceiver import PerceiverResampler
from .qformer import Qformer
from .token_packer import TokenPacker


class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}


def build_vision_resampler(model_args, delay_load=False, **kwargs):
    resampler_type = getattr(model_args, "mm_resampler_type", None)
    vision_tower = getattr(model_args, "mm_vision_tower", getattr(model_args, "vision_tower", None))
    model_args.vision_tower = vision_tower
    if resampler_type == "masked_drop":
        return MaskedDrop(model_args)
    elif resampler_type == "spatial_pool":
        return SpatialPool(model_args, **kwargs)
    elif resampler_type == "perceiver":
        return PerceiverResampler(model_args, **kwargs)
    elif resampler_type == "qformer":
        return Qformer(model_args, **kwargs)
    elif resampler_type == "fast_slow_resampler":
        return TokenPacker(model_args, **kwargs)
    elif resampler_type is None:
        return IdentityMap()

    raise ValueError(f"Unknown resampler type: {resampler_type}")
