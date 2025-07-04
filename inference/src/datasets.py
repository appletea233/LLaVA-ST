import json
import os
import random

import cv2
import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset

from inference.src.prompts import ref_prompts, tvg_prompts
from inference.src.utils import (filter_svg_bboxes_according_to_frame_id)
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN


def load_video(video_path, n_frms=100, return_id=False, split=None, **kwargs):
    vr = VideoReader(video_path)
    if split is None:
        split = [0, len(vr) - 1]

    duration = split[1] - split[0] + 1

    num_frames = n_frms
    frame_idx = [
        split[0] + round((i / (num_frames - 1)) * (duration - 1))
        for i in range(num_frames)
    ]

    video = vr.get_batch(frame_idx).asnumpy()

    if return_id:
        return video, frame_idx
    return video


class CharadesSTGDataset(Dataset):

    def __init__(self, ann_path, data_folder, num_frames=64):
        super(CharadesSTGDataset, self).__init__()
        self.loaded_data = json.load(open(ann_path, "r"))
        self.data_folder = data_folder
        self.num_frames = num_frames

    def __len__(self):
        return len(self.loaded_data)

    def __getitem(self, idx):
        data = self.loaded_data[idx]

        if "activitynet" in self.data_folder.lower():
            img_id = data['image_id']
        else:
            img_id = data['image_id'].replace("v_", "")
        caption = data['caption'].strip(".")
        caption = caption.strip(" ").lower()
        video_path = os.path.join(self.data_folder, img_id)
        frames = load_video(
            video_path=video_path,
            n_frms=self.num_frames,
            height=224,
            width=224,
            sampling="uniform",
            return_msg=False
        )

        prompt = DEFAULT_VIDEO_TOKEN + "\n" + random.choice(
            tvg_prompts).replace("<event>", caption)
        gt = data["timestamp"]
        conversations = [{
            "from": "human",
            "value": prompt
        }, {
            "from": "gpt",
            "value": None
        }]
        return {
            "data_type": "video",
            "frames": frames,
            "video_path": video_path,
            "caption": caption,
            "prompt": prompt,
            "conversations": conversations,
            "file_name": img_id,
            "gt": gt
        }

    def __getitem__(self, idx):
        try:
            return self.__getitem(idx)
        except Exception:
            return self.__getitem__(idx + 1)


class RefCOCOEvalDataset(Dataset):
    REFPROMPTS = ref_prompts

    def __init__(self, ann_path, image_folder, sample_prompt=-1):
        super(RefCOCOEvalDataset, self).__init__()
        loaded_data = json.load(open(ann_path, "r"))
        self.loaded_data = self.process(loaded_data)
        self.image_folder = image_folder
        self.sample_prompt = sample_prompt

    def process(self, data):
        images = data["images"]
        annotations = data["annotations"]
        loaded_data = []
        for image, ann in zip(images, annotations):
            assert ann["id"] == image[
                "id"], "id of annotation and image don't match"
            loaded_data.append({
                "file_name": image["file_name"],
                "height": image["height"],
                "width": image["width"],
                "sents": image["caption"],
                "bbox": ann["bbox"],
                "id": image["id"]
            })
        return loaded_data

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['file_name']
        sent = data['sents']
        image_path = os.path.join(self.image_folder, data["file_name"])
        # image = load_image_square(image_path, OPENAI_CLIP_MEAN)
        image = Image.open(image_path).convert('RGB')
        width = data["width"]
        height = data["height"]

        if DEFAULT_IMAGE_TOKEN in sent:
            prompt = sent
        else:
            if self.sample_prompt == -1:
                REFPROMPT = random.choice(self.REFPROMPTS)
            else:
                REFPROMPT = self.REFPROMPTS[self.sample_prompt]
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + REFPROMPT.replace(
                "<event>", sent)
        conversations = [{
            "from": "human",
            "value": prompt
        }, {
            "from": "gpt",
            "value": None
        }]

        def clip(x):
            return x if x >= 0 and x <= 1 else (0 if x < 0 else 1)

        gt_box = data["bbox"]
        gt_box = [
            gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]
        ]

        gt_box = list(
            map(lambda x: clip(round(x, 3)), [
                gt_box[0] / width, gt_box[1] / height, gt_box[2] / width,
                gt_box[3] / height
            ]))
        if gt_box[0] >= 1:
            pass
        # gt_box: [x1, y1, x2, y2]
        return {
            "data_type": "image",
            "image": image,
            "image_path": image_path,
            "event": sent,
            "prompt": prompt,
            "conversations": conversations,
            "image_id": img_id,
            "gt_bbox": gt_box,
            "width": width,
            "height": height
        }


class VidSTGChatDataset(Dataset):
    tvg_tep = [
        "During the span of", 
        "In the time range", 
        "During the period",
        "Within the time frame of", 
        "In the time period", 
        "During", 
        "Between",
        "At"
    ]

    def __init__(self,
                 ann_path: str,
                 data_folder: str,
                 num_frames: int = 100,
                 task_id=0) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.loaded_data = json.load(open(ann_path, "r"))[:2000]
        self.num_frames = num_frames
        self.task_id = task_id

    def __len__(self):
        return len(self.loaded_data)

    def apply_tokens(self, text, tokens):
        text = text.replace("From <s> to <e>",
                            random.choice(self.tvg_tep) + " {<s>,<e>}")
        for k, v in tokens.items():
            if k in text:
                text = text.replace(k, str(v))
        return text

    def bboxes_xyhw2xyxy(self, boxes):

        def xyhw2xyxy(box):
            return [box[0], box[1], box[2] + box[0], box[3] + box[1]]

        if isinstance(boxes, list):
            return xyhw2xyxy(boxes)
        if isinstance(boxes, dict):
            temp = {}
            for k, v in boxes.items():
                temp[k] = xyhw2xyxy(v)
            return temp

    def __getitem__(self, i):
        sources = self.loaded_data[i]
        sources["video_path"] = os.path.join(self.data_folder,
                                             sources["video_path"])
        meta = sources["meta"]
        width = sources['meta']['width']
        height = sources['meta']['height']
        split = meta["split"]
        spatial_tokens = meta["spatial_token"]
        time_tokens = meta["time_token"]
        conversations = sources["conversations"]

        video, frame_id = load_video(sources["video_path"],
                                     num_frames=self.num_frames,
                                     return_id=True,
                                     split=split)

        keys = list(spatial_tokens.keys())
        for k in keys:
            bboxes = spatial_tokens[k]
            spatial_tokens[k] = self.bboxes_xyhw2xyxy(bboxes)

        time_tokens_float = {}
        for k, v in time_tokens.items():
            time_tokens_float[k] = round(
                (v - split[0]) / (split[1] - split[0] + 1 - 1), 3)
        spatial_tokens_float = {}
        for k, v in spatial_tokens.items():
            if isinstance(v, list):
                spatial_tokens_float[k] = list(
                    map(lambda x: round(x, 3), [
                        v[0] / width, v[1] / height, v[2] / width,
                        v[3] / height
                    ]))
        items = spatial_tokens.items()
        for k, v in items:
            spatial_tokens[k] = filter_svg_bboxes_according_to_frame_id(
                v, frame_id)

        prompt = conversations[0]["value"]
        gt = conversations[1]["value"]

        prompt = self.apply_tokens(prompt, spatial_tokens_float)
        prompt = self.apply_tokens(prompt, time_tokens_float)
        if self.task_id == 0:  # STVG
            prompt += ("Please firstly give the timestamps, and then give the "
                       "spatial bounding box corresponding to each timestamp "
                       "in the time period.")
        elif self.task_id == 1:  # SVG
            prompt += ("Please give the spatial bounding box corresponding to "
                       "each timestamp in the time period.")
        elif self.task_id == 2:  # ELC
            prompt += ("Please firstly give the end timestamp, then give the "
                "event associated with the object/subject, finally give "
                "the spatial bounding box corresponding to each timestamp "
                "in the time period.")
        conversations = [{
            "from": "human",
            "value": prompt
        }, {
            "from": "gpt",
            "value": None
        }]
        if self.task_id == 0:
            gt = {
                "tvg": [time_tokens["<s>"],
                        time_tokens["<e>"]],  # in frame_id currently,
                "svg": spatial_tokens["<bboxes>"]
            }
        elif self.task_id == 1:
            gt = {"svg": spatial_tokens["<bboxes>"]}
        elif self.task_id == 2:
            caption = gt.replace("Ends in <e>, ",
                                 "").replace(" Object bounding box: <bboxes>",
                                             "")
            gt = {
                "tvg": [time_tokens["<s>"],
                        time_tokens["<e>"]],  # in frame_id currently,
                "svg": spatial_tokens["<bboxes>"],
                "caption": caption
            }
        return {
            "data_type": "video",
            "frames": video,
            "video_path": sources["video_path"],
            "prompt": prompt,
            "conversations": conversations,
            "gt": gt,
            "width": sources['meta']['width'],
            "height": sources['meta']['height'],
            "frame_num": video.shape[0],
            "split": split
        }