import torch
import torch.nn as nn
import re
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class TokenPackerAttention(nn.Module):
    def __init__(self, *, dim, dim_head=128, heads=8, patch_devide_pattern='spatial_temporal', 
                 spatial_scale_factor=3, temporal_scale_factor=10):
        super().__init__()
        embed_dim = dim_head * heads
        self.to_q = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(embed_dim)
        self.ln_v = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, heads)

        self.to_out = nn.Linear(embed_dim, dim, bias=False)

        self.patch_devide_pattern = patch_devide_pattern
        self.spatial_scale_factor = spatial_scale_factor
        self.temporal_scale_factor = temporal_scale_factor

    def divide_feature(self, x, kernel_size_t, kernel_size_hw, temporal_token_num, spatial_token_num, c):
        h = w = int(spatial_token_num**0.5)
        t = temporal_token_num

        if t == 1:
            kernel_size_t = 1

        reshape_x = x.reshape(h // kernel_size_hw, kernel_size_hw, w // kernel_size_hw, kernel_size_hw, t // kernel_size_t, kernel_size_t, c) 
        reshape_x = reshape_x.permute(1, 3, 5, 0, 2, 4, 6)  
        reshape_x = reshape_x.reshape(kernel_size_hw ** 2 * kernel_size_t , -1, c)

        return reshape_x

    def forward(self, x, x_multi, attn_mask=None):
        q = self.ln_q(self.to_q(x)).permute(1, 0, 2)
        k = self.ln_k(self.to_k(x_multi)).permute(1, 0, 2)
        v = self.ln_v(self.to_v(x_multi)).permute(1, 0, 2)

        key_spatial_num, key_temporal_num, c = k.shape
        query_spatial_num, query_temporal_num, c = q.shape   # hw, t, c

        if self.patch_devide_pattern == 'spatial':
            q = self.divide_feature(q, 1, 1, query_temporal_num, query_spatial_num, c)   # 1, query_temporal_num*query_spatial_num, c
            k = self.divide_feature(k, 1, self.spatial_scale_factor, key_temporal_num, key_spatial_num, c) # kernel_size, query_temporal_num*query_spatial_num, c
            v = self.divide_feature(v, 1, self.spatial_scale_factor, key_temporal_num, key_spatial_num, c) # kernel_size, query_temporal_num*query_spatial_num, c
        elif self.patch_devide_pattern == 'temporal':
            q = self.divide_feature(q, 1, 1, query_temporal_num, query_spatial_num, c)   # 1, query_temporal_num*query_spatial_num, c
            k = self.divide_feature(k, self.temporal_scale_factor, 1, key_temporal_num, key_spatial_num, c)    # kernel_size, query_temporal_num*query_spatial_num, c
            v = self.divide_feature(v, self.temporal_scale_factor, 1, key_temporal_num, key_spatial_num, c)    # kernel_size, query_temporal_num*query_spatial_num, c
        elif self.patch_devide_pattern == 'spatial_temporal':
            q = self.divide_feature(q, 1, 1, query_temporal_num, query_spatial_num, c)   # 1, query_temporal_num*query_spatial_num, c
            k = self.divide_feature(k, self.temporal_scale_factor, self.spatial_scale_factor, key_temporal_num, key_spatial_num, c)    # kernel_size, query_temporal_num*query_spatial_num, c
            v = self.divide_feature(v, self.temporal_scale_factor, self.spatial_scale_factor, key_temporal_num, key_spatial_num, c)    # kernel_size, query_temporal_num*query_spatial_num, c
        else:
            raise ValueError("Unexpected patch devide pattern")
            
        out = self.attn(q, k, v, attn_mask=attn_mask)[0]

        out = out.reshape(query_spatial_num, query_temporal_num, -1)
        out = out.permute(1, 0, 2)
        out = self.to_out(out)

        return out


class TokenPackerModule(nn.Module):
    def __init__(
            self,
            *,
            raw_grid=27,
            num_latents=81,
            raw_frames=100,
            num_temporal_latents=20,
            embed_dim=1024,
            num_heads=8,
            visual_dim=3456,
            hidden_size=3584,
            patch_devide_pattern='spatial_temporal', 
            spatial_scale_factor=3,
            temporal_scale_factor=5
    ):
        super().__init__()
        self.raw_grid = raw_grid
        self.grid_size = int(num_latents**0.5)

        self.raw_frames = raw_frames
        self.num_temporal_latents = num_temporal_latents
        self.num_queries = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_factor = self.raw_grid // self.grid_size

        self.patch_devide_pattern = patch_devide_pattern

        self.devide_attention = TokenPackerAttention(
            dim=visual_dim, 
            dim_head=embed_dim//num_heads, 
            heads=num_heads, 
            patch_devide_pattern=patch_devide_pattern, 
            spatial_scale_factor=spatial_scale_factor,
            temporal_scale_factor=temporal_scale_factor
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        x_multi = x
        dtype = x.dtype

        t, hw, c = x.shape

        if self.patch_devide_pattern == 'spatial':
            x = x.reshape(x.shape[0], self.raw_grid, self.raw_grid, -1).float().permute(0,3,1,2)
            x = F.interpolate(x, size=(self.grid_size, self.grid_size), mode='bilinear').permute(0,2,3,1) ## fix
            x = x.reshape(x.shape[0], -1, x.shape[-1]).to(dtype)  # B, grid_size * grid_size, C
        elif self.patch_devide_pattern == 'temporal' and t != 1:
            x = x.reshape(t, -1, c).permute(1, 2, 0)   # [hw, c, t]
            x = F.interpolate(x, size=(self.num_temporal_latents,), mode='linear')  # [hw, c, t]
            x = x.permute(2, 0, 1).to(dtype)  # [t, hw, c]

        x = x + self.devide_attention(x, x_multi, attn_mask)
        N, token_num, c = x.shape

        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

class TokenPacker(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        self.depth = model_args.mm_perceiver_depth
        self.slow_num_latents = model_args.mm_perceiver_latents
        self.fast_num_latents = model_args.mm_perceiver_latents_fast
        self.ff_mult = model_args.mm_perceiver_ff_mult
        self.pretrained = model_args.mm_perceiver_pretrained

        self.token_packer = TokenPackerModule(
            raw_grid=27 if 'siglip' in model_args.vision_tower else 24,
            num_latents=self.slow_num_latents,
            embed_dim=1024,
            num_heads=8,
            visual_dim=3584,
            hidden_size=3584,
            patch_devide_pattern='spatial', 
            spatial_scale_factor=3,
            temporal_scale_factor=5
        )

        self.token_packer_slow = TokenPackerModule(
            raw_grid=27 if 'siglip' in model_args.vision_tower else 24,
            num_latents=self.slow_num_latents,
            embed_dim=1024,
            num_heads=8,
            visual_dim=3584,
            hidden_size=3584,
            patch_devide_pattern='temporal', 
            spatial_scale_factor=3,
            temporal_scale_factor=5
        )

        self.token_packer_fast = TokenPackerModule(
            raw_grid=9 if 'siglip' in model_args.vision_tower else 24,
            num_latents=self.fast_num_latents,
            embed_dim=1024,
            num_heads=8,
            visual_dim=3584,
            hidden_size=3584,
            patch_devide_pattern='spatial', 
            spatial_scale_factor=3,
            temporal_scale_factor=5
        )

        if self.pretrained is not None:
            self.load_state_dict(torch.load(self.pretrained))

    @property
    def config(self):
        return {
            "mm_resampler_type": "fast_slow_resampler",
            "mm_perceiver_depth": self.depth,
            "mm_perceiver_latents": self.slow_num_latents,
            "mm_perceiver_latents_fast": self.fast_num_latents,
            "mm_perceiver_ff_mult": self.ff_mult,
            "mm_perceiver_pretrained": self.pretrained,
        }
    
    def forward(self, image_features, slow=True, *args, **kwargs):
        image_features = self.token_packer(image_features) 
        if slow:
            slow_features = self.token_packer_slow(image_features)
            fast_features = self.token_packer_fast(image_features)
            # dummy for deepspeed
            slow_features = slow_features + fast_features.mean() * 0
            return slow_features
        else:
            slow_features = self.token_packer_slow(image_features)
            fast_features = self.token_packer_fast(image_features)  
            # dummy for deepspeed
            slow_features = slow_features + fast_features.mean() * 0
            return slow_features, fast_features