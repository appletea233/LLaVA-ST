#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, position_transfer, token_transfer, reparam
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM


# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        variables: Optional[list] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal_video(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, variables)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            
            loss = None
            vision_config = self.get_model().vision_config
            if labels is not None and all(variables) and self.model.has_init_specific_embeddings:
                logits = None
                temporal_output_embeddings = reparam(self.model.temporal_output_embeddings.weight, self.model.temporal_reparam_mat)
                spatial_height_output_embeddings = reparam(self.model.spatial_height_output_embeddings.weight, self.model.spatial_height_reparam_mat)
                spatial_width_output_embeddings = reparam(self.model.spatial_width_output_embeddings.weight, self.model.spatial_width_reparam_mat)

                for cur_video_idx, (cur_labels, cur_variables) in enumerate(zip(labels, variables)):
                    cur_temporal_output_locations = cur_variables['temporal_output_locations']
                    cur_spatial_height_output_locations = cur_variables['spatial_height_output_locations']
                    cur_spatial_width_output_locations = cur_variables['spatial_width_output_locations']

                    cur_logits = F.linear(hidden_states[cur_video_idx:cur_video_idx+1], torch.cat(
                        [
                            self.lm_head.weight, 
                            temporal_output_embeddings.to(self.lm_head.weight.device),
                            spatial_height_output_embeddings.to(self.lm_head.weight.device),
                            spatial_width_output_embeddings.to(self.lm_head.weight.device),
                        ], 0
                    ))
                    shift_logits = cur_logits[..., :-1, :].contiguous()
                    shift_logits = shift_logits.view(-1, self.config.vocab_size + self.num_temporal_tokens +  self.num_spatial_tokens * 2)

                    temporal_output_token_indices = torch.where(cur_labels == vision_config.temporal_output_token_id)[0]
                    spatial_height_output_token_indices = torch.where(cur_labels == vision_config.spatial_height_output_token_id)[0]
                    spatial_width_output_token_indices = torch.where(cur_labels == vision_config.spatial_width_output_token_id)[0]
                    if len(temporal_output_token_indices) > 0 or len(spatial_height_output_token_indices) > 0 or len(spatial_width_output_token_indices) > 0:
                        cur_onehot_labels = torch.zeros(*cur_labels.shape, self.vocab_size + self.num_temporal_tokens + self.num_spatial_tokens * 2)
                        for i in range(cur_labels.shape[0]):
                            if cur_labels[i] != -100:
                                cur_onehot_labels[i, cur_labels[i]] = 1
                        for i in range(temporal_output_token_indices.shape[0]):
                            floor_position, ceil_position, ratio = position_transfer(cur_temporal_output_locations[i], self.num_temporal_tokens)
                            cur_onehot_labels[[temporal_output_token_indices[i]]][vision_config.temporal_output_token_id] = 0
                            cur_onehot_labels[[temporal_output_token_indices[i]]][self.vocab_size + floor_position] = 1 - ratio
                            if floor_position != ceil_position:
                                cur_onehot_labels[[temporal_output_token_indices[i]]][self.vocab_size + ceil_position] = ratio

                        for i in range(spatial_height_output_token_indices.shape[0]):
                            floor_position, ceil_position, ratio = position_transfer(cur_spatial_height_output_locations[i], self.num_spatial_tokens)
                            cur_onehot_labels[[spatial_height_output_token_indices[i]]][vision_config.spatial_height_output_token_id] = 0
                            cur_onehot_labels[[spatial_height_output_token_indices[i]]][self.vocab_size + self.num_temporal_tokens + floor_position] = 1 - ratio
                            if floor_position != ceil_position:
                                cur_onehot_labels[[spatial_height_output_token_indices[i]]][self.vocab_size + self.num_temporal_tokens + ceil_position] = ratio
                            
                        for i in range(spatial_width_output_token_indices.shape[0]):
                            floor_position, ceil_position, ratio = position_transfer(cur_spatial_width_output_locations[i], self.num_spatial_tokens)
                            cur_onehot_labels[[spatial_width_output_token_indices[i]]][vision_config.spatial_width_output_token_id] = 0
                            cur_onehot_labels[[spatial_width_output_token_indices[i]]][self.vocab_size + self.num_temporal_tokens + self.num_spatial_tokens + floor_position] = 1 - ratio
                            if floor_position != ceil_position:
                                cur_onehot_labels[[spatial_width_output_token_indices[i]]][self.vocab_size + self.num_temporal_tokens + self.num_spatial_tokens + ceil_position] = ratio

                        shift_labels = cur_onehot_labels[1:].contiguous()
                        shift_labels = shift_labels.view(-1, self.vocab_size + self.num_temporal_tokens + self.num_spatial_tokens * 2)
                        shift_labels = shift_labels.to(shift_logits.device)
                        loss_mask = (cur_labels[1:] != -100)
                        origin_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
                        cur_loss = origin_loss[loss_mask.to(origin_loss.device)].mean()
                        loss = cur_loss if loss is None else loss + cur_loss
                    else:
                        shift_labels = cur_labels[..., 1:].contiguous()
                        shift_labels = shift_labels.view(-1)
                        shift_labels = shift_labels.to(shift_logits.device)
                        
                        loss = F.cross_entropy(shift_logits, shift_labels) if loss is None else loss + F.cross_entropy(shift_logits, shift_labels)
                    logits = cur_logits if logits is None else torch.cat([logits, cur_logits], 0)

            else:
                if self.model.has_init_specific_embeddings:
                    logits = F.linear(hidden_states, torch.cat([
                        self.lm_head.weight, 
                        self.model.temporal_output_embeddings.weight.to(self.lm_head.weight.device), 
                        self.model.spatial_height_output_embeddings.weight.to(self.lm_head.weight.device),
                        self.model.spatial_width_output_embeddings.weight.to(self.lm_head.weight.device),
                        ], 0
                    ))
                else:
                    logits = F.linear(hidden_states, self.lm_head.weight)
                logits = logits.float()
                loss = None
                if labels is not None:
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)


            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        variables: Optional[list] = [None],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal_video(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, variables=variables)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
