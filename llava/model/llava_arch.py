#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import *

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random

def position_transfer(position, num_temporal_tokens):
    position = np.clip(position, 0, 1)
    embed_position = position * (num_temporal_tokens - 1)
    floor_position = math.floor(embed_position)
    ceil_position = math.ceil(embed_position)
    ratio = embed_position - floor_position
    return floor_position, ceil_position, ratio

def token_transfer(position, temporal_embed_tokens, return_position=False):
    position = np.clip(position, 0, 1)
    floor_position, ceil_position, ratio = position_transfer(position, temporal_embed_tokens.shape[0])
    ret_feature = temporal_embed_tokens[floor_position] * (1 - ratio) + temporal_embed_tokens[ceil_position] * ratio
    if return_position:
        return ret_feature, floor_position, ceil_position, ratio
    else:
        return ret_feature

def reparam(weight, reparam_mat):
    reparam_weight = reparam_mat.to(weight.dtype).to(weight.device) @ weight
    return weight + reparam_weight - reparam_weight.detach()

class VisionConfig:
    def __init__(self):
        self.slow_token = False
        self.fast_token = False

        self.frame_size = 384
        self.patch_size = 14
        self.hidden_size = 1024
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None
        self.im_start_token = None
        self.im_end_token = None
        self.im_patch_token = None

        self.fast_token_num = 9
        self.slow_token_num = 81
        self.fast_frame_num = 100
        self.slow_frame_num = 20

        self.image_token_num = 729

        self.spatial_token_num = 100

        self.temporal_input_token_id = -1000
        self.temporal_output_token_id = -1001
        self.spatial_height_input_token_id = -1002
        self.spatial_height_output_token_id = -1003
        self.spatial_width_input_token_id = -1004
        self.spatial_width_output_token_id = -1005

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        self.has_init_specific_embeddings = False
        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
            self.vision_config = VisionConfig()
            if config.mm_resampler_type == "fast_slow_resampler":
                self.vision_config.slow_token = True
                self.vision_config.fast_token = True

        if hasattr(config, "num_spatial_tokens") and hasattr(config, "num_temporal_tokens"):
            self.init_special_embeddings()

    def init_special_embeddings(self):
        if self.has_init_specific_embeddings:
            return
        self.has_init_specific_embeddings = True
        # init spatial token embedding
        self.spatial_height_input_embeddings = torch.nn.Embedding(self.vision_config.spatial_token_num, self.config.hidden_size)
        self.spatial_height_output_embeddings = torch.nn.Linear(self.config.hidden_size, self.vision_config.spatial_token_num, bias=False)

        self.spatial_width_input_embeddings = torch.nn.Embedding(self.vision_config.spatial_token_num, self.config.hidden_size)
        self.spatial_width_output_embeddings = torch.nn.Linear(self.config.hidden_size, self.vision_config.spatial_token_num, bias=False)
        
        # init temporal token embedding
        self.temporal_input_embeddings = torch.nn.Embedding(self.vision_config.fast_frame_num, self.config.hidden_size)
        self.temporal_output_embeddings = torch.nn.Linear(self.config.hidden_size, self.vision_config.fast_frame_num, bias=False)

        index_vec = torch.arange(self.vision_config.spatial_token_num)
        self.spatial_width_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())
        self.spatial_height_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())
        index_vec = torch.arange(self.vision_config.fast_frame_num)
        self.temporal_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())
        self.config.num_temporal_tokens = self.vision_config.spatial_token_num
        self.config.num_spatial_tokens = self.vision_config.fast_frame_num

    def init_vision_config(self, tokenizer):
        vision_config = self.vision_config
        vision_config.im_start_token, vision_config.im_end_token, vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN])
        vision_config.vid_start_token, vision_config.vid_end_token, vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN])
        if vision_config.slow_token:
            vision_config.slow_vid_start_token, vision_config.slow_vid_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_SLOW_VID_START_TOKEN, DEFAULT_SLOW_VID_END_TOKEN])
        vision_config.temporal_input_token_id = tokenizer.convert_tokens_to_ids([TEMPORAL_INPUT_TOKEN])[0]
        vision_config.temporal_output_token_id = tokenizer.convert_tokens_to_ids([TEMPORAL_OUTPUT_TOKEN])[0]

        vision_config.spatial_height_input_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_HEIGHT_INPUT_TOKEN])[0]
        vision_config.spatial_height_output_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_HEIGHT_OUTPUT_TOKEN])[0]

        vision_config.spatial_width_input_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_WIDTH_INPUT_TOKEN])[0]
        vision_config.spatial_width_output_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_WIDTH_OUTPUT_TOKEN])[0]
        index_vec = torch.arange(self.vision_config.spatial_token_num)
        self.spatial_width_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())
        self.spatial_height_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())
        # Neighboring Token Propagation (NTP)
        index_vec = torch.arange(self.vision_config.fast_frame_num)
        self.temporal_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())
        return self.vision_config

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, tokenizer, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            vision_tower = self.vision_tower
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_resampler = vision_resampler

            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
        
        self.vision_config = VisionConfig()
        if model_args.mm_resampler_type == "fast_slow_resampler":
            self.vision_config.slow_token = True

        return self.vision_config

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature
    
    def is_siglip(self):
        if 'siglip' in self.get_model().config.mm_vision_tower.lower():
            return True
        return False

    def encode_images(self, images, split_sizes=None, modalities=None, 
        temporal_information_injection=None, spatial_height_information_injection=None, spatial_width_information_injection=None):

        if self.model.has_init_specific_embeddings:
            spatial_height_information_injection = spatial_height_information_injection.permute(1, 2, 0)
            spatial_width_information_injection = spatial_width_information_injection.permute(1, 2, 0)

        images = torch.split(images, split_sizes)

        slow_image_features = []
        fast_image_features = []
        
        for modality, image in zip(modalities, images):
            image_feature, image_forward_outs = self.get_model().get_vision_tower()(image)
            image_feature = self.get_model().mm_projector(image_feature)

            if self.model.has_init_specific_embeddings:
                t, hw, _ = image_feature.shape
                h = w = int(hw**0.5)
                image_feature = image_feature.reshape(t, h, w, -1)
                resize_spatial_height_information_injection = F.interpolate(spatial_height_information_injection, size=(h,), mode='linear').permute(0, 2, 1).unsqueeze(2)
                resize_spatial_width_information_injection = F.interpolate(spatial_width_information_injection, size=(w,), mode='linear').permute(0, 2, 1).unsqueeze(1)
                if modality == "video":
                    resize_temporal_information_injection = temporal_information_injection.unsqueeze(1)
                if modality == "image":
                    resize_temporal_information_injection = temporal_information_injection[0:1].unsqueeze(1) + temporal_information_injection.sum() * 0.
                
                # LAPE
                image_feature = image_feature + resize_spatial_height_information_injection + resize_spatial_width_information_injection + resize_temporal_information_injection
                
                image_feature = image_feature.reshape(t, hw, -1)
            if modality == 'video':
                if 'fast_slow_resampler' in self.get_model().config.mm_resampler_type:
                    image_feature = self.get_model().vision_resampler(image_feature, slow=False)
                    # slow_image_feature, fast_image_feature = self.get_model().vision_resampler(image_feature, slow=False)
                    if type(image_feature) == tuple:
                        # image_feature = [self.get_model().mm_projector(i) for i in image_feature]
                        slow_image_features.append(image_feature[0])
                        fast_image_features.append(image_feature[1])
                    else:
                        slow_image_features.append(None)
                        fast_image_features.append(image_feature)
                else:
                    slow_image_features.append(image_feature)
            else:
                if 'fast_slow_resampler' in self.get_model().config.mm_resampler_type:
                    slow_image_feature = self.get_model().vision_resampler(image_feature)
                    # image_feature = self.get_model().mm_projector(image_feature)

                    use_downsample_image = getattr(self.config, "use_downsample_image", False)
                    if use_downsample_image:
                        image_feature = slow_image_feature
                    else:
                        image_feature = image_feature + slow_image_feature.mean() * 0.

                    slow_image_features.append(image_feature)
                    fast_image_features.append(None)
                else:
                    slow_image_features.append(image_feature)
        return slow_image_features, fast_image_features

    def prepare_inputs_labels_for_multimodal_video(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images, 
        modalities=["image"], 
        image_sizes=None,
        variables=None,
    ):
        # orig_embeds_params = getattr(self.get_model(), 'orig_embeds_params', None)
        orig_embeds_params = None
        if self.model.has_init_specific_embeddings:
            temporal_input_embeddings = reparam(self.model.temporal_input_embeddings.weight, self.model.temporal_reparam_mat)
            temporal_output_embeddings = reparam(self.model.temporal_output_embeddings.weight, self.model.temporal_reparam_mat)
            spatial_height_input_embeddings = reparam(self.model.spatial_height_input_embeddings.weight, self.model.spatial_height_reparam_mat)
            spatial_height_output_embeddings = reparam(self.model.spatial_height_output_embeddings.weight, self.model.spatial_height_reparam_mat)
            spatial_width_input_embeddings = reparam(self.model.spatial_width_input_embeddings.weight, self.model.spatial_width_reparam_mat)
            spatial_width_output_embeddings = reparam(self.model.spatial_width_output_embeddings.weight, self.model.spatial_width_reparam_mat)

            temporal_information_injection = ((temporal_input_embeddings + temporal_output_embeddings)/2).unsqueeze(1)
            spatial_height_information_injection = ((spatial_height_input_embeddings + spatial_height_output_embeddings)/2).unsqueeze(1)
            spatial_width_information_injection = ((spatial_width_input_embeddings + spatial_width_output_embeddings)/2).unsqueeze(1)

            device = self.get_model().embed_tokens.weight.device
            inputs_embeds = F.embedding(input_ids, torch.cat(
                [self.get_model().embed_tokens.weight, temporal_input_embeddings.to(device), spatial_height_input_embeddings.to(device), spatial_width_input_embeddings.to(device)]
            ))
        else:
            temporal_information_injection = None
            spatial_height_information_injection = None
            spatial_width_information_injection = None
            
            device = self.get_model().embed_tokens.weight.device
            inputs_embeds = F.embedding(input_ids, self.get_model().embed_tokens.weight)
        
        if (input_ids.shape[1] != 1 or self.training):

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features, encoded_fast_image_features = self.encode_images(concat_images, split_sizes, modalities, temporal_information_injection, spatial_height_information_injection, spatial_width_information_injection)

            new_input_embeds = []
            for cur_video_idx,(cur_input_ids, cur_input_embeds, modality) in enumerate(zip(input_ids, inputs_embeds, modalities)):
                if modality == 'image':
                    image_start_tokens = torch.where(cur_input_ids == self.vision_config.im_start_token)[0]

                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = encoded_image_features[cur_video_idx].to(device=cur_input_embeds.device)
                        cur_image_features = cur_image_features.flatten(0, 1)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches - 1] != self.vision_config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:image_start_token_pos],
                                cur_image_features, 
                                cur_input_embeds[image_start_token_pos + num_patches:],
                            ), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:image_start_token_pos],
                                cur_image_features,
                                cur_input_embeds[image_start_token_pos + num_patches:]
                            ), dim=0)

                elif modality == 'video':
                    video_start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                    
                    for video_start_token_pos in video_start_tokens:
                        cur_video_features = encoded_fast_image_features[cur_video_idx].to(device=cur_input_embeds.device)

                        cur_video_features = cur_video_features.flatten(0, 1)
                        num_patches = cur_video_features.shape[0]
                        if cur_input_ids[video_start_token_pos + num_patches - 1] != self.vision_config.vid_end_token:
                            raise ValueError("The video end token should follow the video start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:video_start_token_pos],
                                cur_video_features, 
                                cur_input_embeds[video_start_token_pos + num_patches:],
                            ), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:video_start_token_pos],
                                cur_video_features,
                                cur_input_embeds[video_start_token_pos + num_patches:]
                            ), dim=0)
                        # cur_video_idx += 1
                    
                    if self.vision_config.slow_token:
                        slow_video_start_tokens = torch.where(cur_input_ids == self.vision_config.slow_vid_start_token)[0]
                        
                        for video_start_token_pos in slow_video_start_tokens:
                            cur_video_features = encoded_image_features[cur_video_idx].to(device=cur_input_embeds.device)

                            cur_video_features = cur_video_features.flatten(0, 1)
                            num_patches = cur_video_features.shape[0]
                            if cur_input_ids[video_start_token_pos + num_patches - 1] != self.vision_config.slow_vid_end_token:
                                raise ValueError("The video end token should follow the video start token.")
                            if orig_embeds_params is not None:
                                cur_new_input_embeds = torch.cat((
                                    cur_input_embeds[:video_start_token_pos],
                                    cur_video_features, 
                                    cur_input_embeds[video_start_token_pos + num_patches:],
                                ), dim=0)
                            else:
                                cur_new_input_embeds = torch.cat((
                                    cur_input_embeds[:video_start_token_pos],
                                    cur_video_features,
                                    cur_input_embeds[video_start_token_pos + num_patches:]
                                ), dim=0)
                    
                else:
                    raise ValueError("Unexpected modality besides image and video.")
                new_input_embeds.append(cur_new_input_embeds)

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        if (input_ids.shape[1] != 1 or self.training) and all(variables):
            new_input_embeds = []

            for cur_video_idx, (cur_input_ids, cur_input_embeds, cur_variables) in enumerate(zip(input_ids, inputs_embeds, variables)):
                cur_temporal_input_locations = cur_variables['temporal_input_locations']
                cur_temporal_output_locations = cur_variables['temporal_output_locations']
                cur_spatial_height_input_locations = cur_variables['spatial_height_input_locations']
                cur_spatial_height_output_locations = cur_variables['spatial_height_output_locations']
                cur_spatial_width_input_locations = cur_variables['spatial_width_input_locations']
                cur_spatial_width_output_locations = cur_variables['spatial_width_output_locations']

                cur_new_input_embeds = inputs_embeds[cur_video_idx].clone()

                if (cur_input_ids == self.vision_config.temporal_input_token_id).sum() \
                    + (cur_input_ids == self.vision_config.temporal_output_token_id).sum() \
                    + (cur_input_ids == self.vision_config.spatial_height_input_token_id).sum() \
                    + (cur_input_ids == self.vision_config.spatial_height_output_token_id).sum() \
                    + (cur_input_ids == self.vision_config.spatial_width_input_token_id).sum() \
                    + (cur_input_ids == self.vision_config.spatial_width_output_token_id).sum() == 0:
                    new_input_embeds.append(cur_input_embeds)
                else:
                    if (cur_input_ids == self.vision_config.temporal_input_token_id).sum() > len(cur_temporal_input_locations) \
                        or (cur_input_ids == self.vision_config.temporal_output_token_id).sum() > len(cur_temporal_output_locations):
                        raise ValueError("The number of temporal tokens and input temporal location features should be the same.")
                    if (cur_input_ids == self.vision_config.temporal_input_token_id).sum() > len(cur_temporal_input_locations) \
                        or (cur_input_ids == self.vision_config.temporal_output_token_id).sum() > len(cur_temporal_output_locations) \
                        or (cur_input_ids == self.vision_config.spatial_height_input_token_id).sum() > len(cur_spatial_height_input_locations) \
                        or (cur_input_ids == self.vision_config.spatial_height_output_token_id).sum() > len(cur_spatial_height_output_locations) \
                        or (cur_input_ids == self.vision_config.spatial_width_input_token_id).sum() > len(cur_spatial_width_input_locations) \
                        or (cur_input_ids == self.vision_config.spatial_width_output_token_id).sum() > len(cur_spatial_width_output_locations):
                        raise ValueError("The number of spatial tokens and input temporal location features should be the same.")
                    
                    temporal_input_token_indices = torch.where(cur_input_ids == self.vision_config.temporal_input_token_id)[0]
                    temporal_output_token_indices = torch.where(cur_input_ids == self.vision_config.temporal_output_token_id)[0]
                    for i, index in enumerate(temporal_input_token_indices):
                        cur_temporal_location_feature = token_transfer(cur_temporal_input_locations[i], temporal_input_embeddings)
                        cur_new_input_embeds[index] = cur_temporal_location_feature
                    for i, index in enumerate(temporal_output_token_indices):
                        cur_temporal_location_feature = token_transfer(cur_temporal_output_locations[i], temporal_input_embeddings)
                        cur_new_input_embeds[index] = cur_temporal_location_feature

                    spatial_height_input_token_indices = torch.where(cur_input_ids == self.vision_config.spatial_height_input_token_id)[0]
                    spatial_height_output_token_indices = torch.where(cur_input_ids == self.vision_config.spatial_height_output_token_id)[0]
                    for i, index in enumerate(spatial_height_input_token_indices):
                        cur_spatial_height_location_feature = token_transfer(cur_spatial_height_input_locations[i], spatial_height_input_embeddings)
                        cur_new_input_embeds[index] = cur_spatial_height_location_feature
                    for i, index in enumerate(spatial_height_output_token_indices):
                        cur_spatial_height_location_feature = token_transfer(cur_spatial_height_output_locations[i], spatial_height_input_embeddings)
                        cur_new_input_embeds[index] = cur_spatial_height_location_feature

                    spatial_width_input_token_indices = torch.where(cur_input_ids == self.vision_config.spatial_width_input_token_id)[0]
                    spatial_width_output_token_indices = torch.where(cur_input_ids == self.vision_config.spatial_width_output_token_id)[0]
                    for i, index in enumerate(spatial_width_input_token_indices):
                        cur_spatial_width_location_feature = token_transfer(cur_spatial_width_input_locations[i], spatial_width_input_embeddings)
                        cur_new_input_embeds[index]
                    for i, index in enumerate(spatial_width_output_token_indices):
                        cur_spatial_width_location_feature = token_transfer(cur_spatial_width_output_locations[i], spatial_width_input_embeddings)
                        cur_new_input_embeds[index] = cur_spatial_width_location_feature
                    
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels
            
    def initialize_image_tokenizer(self, tokenizer):
        vision_config = self.get_model().vision_config

        num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        vision_config.im_start_token, vision_config.im_end_token, vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN])

        return num_new_tokens
    
    def initialize_video_tokenizer(self, tokenizer):
        vision_config = self.get_model().vision_config

        num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        vision_config.vid_start_token, vision_config.vid_end_token, vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN])
        if vision_config.slow_token:
            num_new_tokens += tokenizer.add_tokens([DEFAULT_SLOW_VID_START_TOKEN, DEFAULT_SLOW_VID_END_TOKEN], special_tokens=True)
            vision_config.slow_vid_start_token, vision_config.slow_vid_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_SLOW_VID_START_TOKEN, DEFAULT_SLOW_VID_END_TOKEN])

        return num_new_tokens

    def initialize_spatial_temporal_tokens(self, tokenizer, num_temporal_tokens, num_spatial_tokens):
        vision_config = self.get_model().vision_config
        self.num_temporal_tokens = num_temporal_tokens
        self.num_spatial_tokens = num_spatial_tokens
        self.temporal_tokens = [TEMPORAL_TOKEN_FORMAT.format(i) for i in range(num_temporal_tokens)]
        self.spatial_height_tokens = [SPATIAL_HEIGHT_TOKEN_FORMAT.format(i) for i in range(num_spatial_tokens)]
        self.spatial_width_tokens = [SPATIAL_WIDTH_TOKEN_FORMAT.format(i) for i in range(num_spatial_tokens)]
        num_new_tokens = tokenizer.add_tokens([TEMPORAL_INPUT_TOKEN, TEMPORAL_OUTPUT_TOKEN, SPATIAL_HEIGHT_INPUT_TOKEN, SPATIAL_HEIGHT_OUTPUT_TOKEN, SPATIAL_WIDTH_INPUT_TOKEN, SPATIAL_WIDTH_OUTPUT_TOKEN], special_tokens=True)
        vision_config.temporal_input_token_id = tokenizer.convert_tokens_to_ids([TEMPORAL_INPUT_TOKEN])[0]
        vision_config.temporal_output_token_id = tokenizer.convert_tokens_to_ids([TEMPORAL_OUTPUT_TOKEN])[0]
        vision_config.spatial_height_input_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_HEIGHT_INPUT_TOKEN])[0]
        vision_config.spatial_height_output_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_HEIGHT_OUTPUT_TOKEN])[0]
        vision_config.spatial_width_input_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_WIDTH_INPUT_TOKEN])[0]
        vision_config.spatial_width_output_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_WIDTH_OUTPUT_TOKEN])[0]
        _ = tokenizer.add_tokens(self.temporal_tokens, special_tokens=True)
        _ = tokenizer.add_tokens(self.spatial_height_tokens, special_tokens=True)
        _ = tokenizer.add_tokens(self.spatial_width_tokens, special_tokens=True)

        # Neighboring Token Propagation (NTP)
        index_vec = torch.arange(num_spatial_tokens)
        self.model.spatial_width_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())
        self.model.spatial_height_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())
        index_vec = torch.arange(num_temporal_tokens)
        self.model.temporal_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())

        self.model.config.num_spatial_tokens = self.num_spatial_tokens
        self.model.config.num_temporal_tokens = self.num_temporal_tokens
        return num_new_tokens

    def initialize_temporal_tokens(self, tokenizer, num_temporal_tokens):
        vision_config = self.get_model().vision_config
        self.num_temporal_tokens = num_temporal_tokens
        self.temporal_tokens = [TEMPORAL_TOKEN_FORMAT.format(i) for i in range(num_temporal_tokens)]
        num_new_tokens = tokenizer.add_tokens([TEMPORAL_INPUT_TOKEN, TEMPORAL_OUTPUT_TOKEN], special_tokens=True)
        vision_config.temporal_input_token_id = tokenizer.convert_tokens_to_ids([TEMPORAL_INPUT_TOKEN])[0]
        vision_config.temporal_output_token_id = tokenizer.convert_tokens_to_ids([TEMPORAL_OUTPUT_TOKEN])[0]
        _ = tokenizer.add_tokens(self.temporal_tokens, special_tokens=True)
        
        # Neighboring Token Propagation (NTP)
        index_vec = torch.arange(num_temporal_tokens)
        self.model.temporal_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())

        self.model.config.num_temporal_tokens = self.num_temporal_tokens
        return num_new_tokens

    def initialize_spatial_tokens(self, tokenizer, num_spatial_tokens):
        vision_config = self.get_model().vision_config
        self.num_spatial_tokens = num_spatial_tokens
        self.spatial_height_tokens = [SPATIAL_HEIGHT_TOKEN_FORMAT.format(i) for i in range(num_spatial_tokens)]
        self.spatial_width_tokens = [SPATIAL_WIDTH_TOKEN_FORMAT.format(i) for i in range(num_spatial_tokens)]
        num_new_tokens = tokenizer.add_tokens([SPATIAL_HEIGHT_INPUT_TOKEN, SPATIAL_HEIGHT_OUTPUT_TOKEN, SPATIAL_WIDTH_INPUT_TOKEN, SPATIAL_WIDTH_OUTPUT_TOKEN], special_tokens=True)

        vision_config.spatial_height_input_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_HEIGHT_INPUT_TOKEN])[0]
        vision_config.spatial_height_output_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_HEIGHT_OUTPUT_TOKEN])[0]
        _ = tokenizer.add_tokens(self.spatial_height_tokens, special_tokens=True)

        vision_config.spatial_width_input_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_WIDTH_INPUT_TOKEN])[0]
        vision_config.spatial_width_output_token_id = tokenizer.convert_tokens_to_ids([SPATIAL_WIDTH_OUTPUT_TOKEN])[0]
        _ = tokenizer.add_tokens(self.spatial_width_tokens, special_tokens=True)

        # Neighboring Token Propagation (NTP)
        index_vec = torch.arange(num_spatial_tokens)
        self.model.spatial_width_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())
        self.model.spatial_height_reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())

        self.model.config.num_spatial_tokens = self.num_spatial_tokens
        return num_new_tokens

    def initialize_embedings(self, num_new_tokens, num_cur_tokens, pretrain_mm_mlp_adapter=None):

        self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone()]
        
        self.resize_token_embeddings(num_cur_tokens)
        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config.vocab_size = num_cur_tokens
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        for p in self.get_input_embeddings().parameters():
            p.requires_grad = True
        for p in self.get_output_embeddings().parameters():
            p.requires_grad = False

        if pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
            # assert num_new_tokens == 2
            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(
                    f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                    f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

    def init_special_embeddings(self):
        self.model.init_special_embeddings()

    @property
    def vision_config(self):
        return self.get_model().vision_config

