import argparse
import copy
import json
import math
import os
import os.path as osp
import re

import torch
import tqdm
import transformers

from inference.src.datasets import (
    CharadesSTGDataset, 
    RefCOCOEvalDataset,
    VidSTGChatDataset
)
from inference.src.set_args import set_args
from inference.src.utils import format_box_in_text, format_float_in_text, format_span_in_text, get_variables, replace_and_normalize
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN, 
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN, 
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_SLOW_VID_END_TOKEN,
    DEFAULT_SLOW_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN, 
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN, 
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX, 
    IMAGE_TOKEN_INDEX
)
from llava.model.builder import load_lora_model


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def preprocess_qwen(sources,
                    tokenizer: transformers.PreTrainedTokenizer,
                    has_image: bool = False,
                    max_len=2048,
                    system_message: str = "You are a helpful assistant."):
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [
        im_end
    ] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end
                                                                 ] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence[
                "value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i == IMAGE_TOKEN_INDEX
                        for i in _input_id]) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(
                    sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [
                im_end
            ] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [
                im_start
            ] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[
                len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


def preprocess_multimodal(sources, vision_config):
    is_multimodal = True
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if sentence["value"] is None:
                continue
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                # TODO maybe this should be changed for interleaved data?
                # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not
                # sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                # only check for num_im=1
                num_im = len(re.findall(DEFAULT_IMAGE_TOKEN,
                                        sentence["value"]))
                if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence[
                        "value"] and not sentence["value"].startswith(
                            DEFAULT_IMAGE_TOKEN):
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "").strip()
                    sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence[
                        "value"]
                    sentence["value"] = sentence["value"].strip()
                    if ("mmtag"
                            in conversation_lib.default_conversation.version):
                        sentence["value"] = sentence["value"].replace(
                            DEFAULT_IMAGE_TOKEN,
                            "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * (
                    vision_config.image_token_num - 2)
                replace_token = (DEFAULT_IM_START_TOKEN + replace_token +
                                 DEFAULT_IM_END_TOKEN)
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, replace_token)

                # For videoInstruct-100k noisy_data.
                # TODO: Ask Yuanhan to clean the data instead of
                # leaving the noise code here.
                sentence["value"] = sentence["value"].replace(
                    "QA_GT_caption_based_noisy", "")
            elif DEFAULT_VIDEO_TOKEN in sentence["value"]:
                num_vid = len(
                    re.findall(DEFAULT_VIDEO_TOKEN, sentence["value"]))
                if num_vid == 1 and DEFAULT_VIDEO_TOKEN in sentence[
                        "value"] and not sentence["value"].startswith(
                            DEFAULT_VIDEO_TOKEN):
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_VIDEO_TOKEN, "").strip()
                    sentence["value"] = DEFAULT_VIDEO_TOKEN + "\n" + sentence[
                        "value"]
                    sentence["value"] = sentence["value"].strip()

                replace_token = ""
                if vision_config.slow_token:
                    replace_token += (DEFAULT_VID_START_TOKEN +
                                      DEFAULT_VIDEO_PATCH_TOKEN *
                                      (vision_config.fast_token_num *
                                       vision_config.fast_frame_num - 2) +
                                      DEFAULT_VID_END_TOKEN)
                    replace_token += (DEFAULT_SLOW_VID_START_TOKEN +
                                      DEFAULT_VIDEO_PATCH_TOKEN *
                                      (vision_config.slow_token_num *
                                       vision_config.slow_frame_num - 2) +
                                      DEFAULT_SLOW_VID_END_TOKEN)
                else:
                    replace_token += (DEFAULT_VID_START_TOKEN +
                                      DEFAULT_VIDEO_PATCH_TOKEN *
                                      (vision_config.slow_token_num *
                                       vision_config.slow_frame_num) +
                                      DEFAULT_VID_END_TOKEN)
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_VIDEO_TOKEN, replace_token)

    return sources


class LocalDataset(torch.utils.data.Dataset):

    def __init__(self, base_dataset, ann_path: str, data_folder: str,
                 **kwargs):
        self.dataset = base_dataset(ann_path, data_folder, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem(self, idx):
        data = copy.deepcopy(self.dataset[idx])

        # For inputs, convert the format into special tokens form,
        # and add the parameter 'variables'
        # For ground truth, organize it into the large model format.
        # llava-st uses percentage decimals (this is not done here,
        # it's handled in eval)

        conversations = data["conversations"]
        for con in conversations:
            if con["value"] is None:
                continue
            in_out = "INPUT" if con["from"] == "human" else "OUTPUT"
            con["value"] = format_box_in_text(con["value"], in_out=in_out)
            con["value"] = format_span_in_text(con["value"], in_out=in_out)
            con["value"] = format_float_in_text(con["value"], in_out=in_out)
        conversations, variables = get_variables(conversations)
        data["conversations"] = conversations
        data["variables"] = variables
        return data

    def __getitem__(self, idx):
        try:
            return self.__getitem(idx)
        except Exception:
            new_idx = idx + 1 if idx + 1 < len(self.dataset) else 0
            return self.__getitem__(new_idx)


def collate_fn(batch_data):
    return batch_data


def inference_task(args,
                   model,
                   tokenizer,
                   image_processor=None,
                   vision_config=None):
    task = args.task.lower()
    dataset = args.dataset.lower()

    model.config.max_frame = 100

    if task == "rec" or dataset == "refcoco":
        dataset = dataset = LocalDataset(RefCOCOEvalDataset,
                                         data_folder=args.data_folder,
                                         ann_path=args.ann_path,
                                         sample_prompt=-1)
    elif task == "st-align":
        model.config.max_frame = 100
        dataset = LocalDataset(VidSTGChatDataset,
                               data_folder=args.data_folder,
                               ann_path=args.ann_path,
                               num_frames=model.config.max_frame,
                               task_id=args.task_id)
    elif task == "tvg" or dataset == "charades_sta":
        model.config.max_frame = 100
        dataset = LocalDataset(CharadesSTGDataset,
                               data_folder=args.data_folder,
                               ann_path=args.ann_path,
                               num_frames=model.config.max_frame)

    if isinstance(dataset, str):
        cls = globals()[args.cls]
        dataset = LocalDataset(cls,
                               data_folder=args.data_folder,
                               ann_path=args.ann_path,
                               num_frames=model.config.max_frame,
                               task=args.type)
    total_len = len(dataset)
    slice_len = math.ceil(total_len / args.chunk_num)
    chunk_id_lists = [
        range(i, i + slice_len) for i in range(0, total_len, slice_len)
    ]
    cur_chunk_list = chunk_id_lists[args.chunk_id]
    for num in tqdm.tqdm(cur_chunk_list):

        batch: dict = dataset[num]
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        if batch["data_type"] == "image":
            image = batch["image"]
            image_tensors = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'].half().cuda(
                )  # type: ignore
            batch["image"] = image_tensors
        elif batch["data_type"] == "video":
            video = batch["frames"]
            image_tensors = image_processor.preprocess(
                video, return_tensors="pt")["pixel_values"].cuda().half(
                )  # type: ignore
            batch["frames"] = image_tensors

        sources = [batch]
        sources = preprocess_multimodal(
            copy.deepcopy([e["conversations"] for e in sources]),
            vision_config)
        input_ids = preprocess_qwen(
            [sources[0][0], {
                'from': 'gpt',
                'value': None
            }],
            tokenizer,
            has_image=True).cuda()
        variables = [batch["variables"]]
        output_ids = model.generate(
            input_ids,
            images=[image_tensors],  # type: ignore
            do_sample=True,
            temperature=0.01,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            variables=variables,
            modalities=[batch["data_type"]],
            max_new_tokens=1024,
            use_cache=True)

        outputs = tokenizer.batch_decode(output_ids,
                                         skip_special_tokens=False)[0]
        outputs = outputs.replace("<|im_end|>", "")
        batch["outputs"] = outputs
        try:
            batch["converted_outputs"] = replace_and_normalize(outputs)
        except Exception:
            batch["converted_outputs"] = None

        output_dic = batch
        keys = list(output_dic.keys())
        for key in keys:
            if isinstance(output_dic[key], torch.Tensor):
                del output_dic[key]
        mode = "a"
        f = open(args.save_path, mode)
        json.dump(output_dic, f)
        f.write("\n")
        f.close()


def load_weights(model, weight_dict):
    model_state_dict = model.state_dict()

    for name, tensor in weight_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(tensor)
        else:
            raise KeyError(f"Key {name} not found in model's state_dict.")
    model.load_state_dict(model_state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument("--deepspeed", type=str, default="scripts/zero2.json")
    parser.add_argument("--model_path", type=str, default="appletea2333/LLaVA-ST-Qwen2-7B")
    parser.add_argument("--lora_path", type=str,  default="")
    parser.add_argument("--ann_path", type=str, default="")
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--task", type=str, default="rec")
    parser.add_argument("--dataset", type=str, default="refcoco")
    parser.add_argument("--all_name", type=str, default="moviechat_breakpoint")
    parser.add_argument("--sub_dir", type=str, default="debug")
    parser.add_argument("--save_dir", type=str, default="./eval_outputs")
    parser.add_argument("--save_name", type=str, default="debug.json")
    parser.add_argument("--chunk_num", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()

    if args.ddp:
        torch.distributed.init_process_group(backend='nccl')

    set_args(args)
    if not args.lora_path:
        args.lora_path = []
    else:
        args.lora_path = args.lora_path.split(" ")
    args.save_dir = osp.join(
        args.save_dir,
        args.model_path.split("/")[-1], 
        args.sub_dir,
        args.task, 
        args.dataset
    )
    os.makedirs(args.save_dir, exist_ok=True)
    args.save_name = args.all_name + ".json"
    args.save_path = osp.join(args.save_dir, args.save_name)

    print(args)

    if osp.exists(osp.join(args.save_dir, args.save_name + ".lock")):
        try:
            os.remove(osp.join(args.save_dir, args.save_name + ".lock"))
        except Exception as e:
            print(e)

    tokenizer, model, image_processor, max_length = load_lora_model(
        args.lora_path,
        args.model_path,
        "llava_qwen",
        device_map="auto",
        overwrite_config={
            "num_spatial_tokens": 100,
            "num_temporal_tokens": 100
        })  # Add any other thing you want to pass in llava_model_args

    vision_config = model.model.vision_config

    if not osp.exists(osp.join(args.save_dir, args.save_name + ".lock")):
        open(args.save_path, "w").close()
        open(osp.join(args.save_dir, args.save_name + ".lock"), "w").close()
    inference_task(args, model, tokenizer, image_processor, vision_config)
    if osp.exists(osp.join(args.save_dir, args.save_name + ".lock")):
        try:
            os.remove(osp.join(args.save_dir, args.save_name + ".lock"))
        except Exception as e:
            print(e)
