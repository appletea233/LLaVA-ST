import sys
import json
import warnings
import copy
import os.path as osp
import os
import argparse
sys.path.append("./")

from llava.model.builder import load_pretrained_model
from llava.model import *
from llava.constants import *
from inference.multi_task_inference import preprocess_multimodal, preprocess_qwen
from inference.src.utils import *
from inference.src.datasets import *

from visualize import *
from utils import *

class Chat:
    def __init__(self, model_path):
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path, None, "llava_qwen", device_map="auto")  # Add any other thing you want to pass in llava_model_args
        self.model.model.init_vision_config(self.tokenizer)
        self.vision_config = self.model.model.vision_config

    def chat(self, query, data_path=None, modality="text", split=None):
        if modality == "image":
            image = Image.open(data_path).convert('RGB')
            image_tensors = self.image_processor.preprocess(image, return_tensors='pt')[
                'pixel_values'].half().cuda()  # type: ignore
            query = "<image>\n"+query
        elif modality == "video":
            video = load_video(data_path, split=split)
            image_tensors = self.image_processor.preprocess(video, return_tensors="pt")[
                "pixel_values"].cuda().half()  # type: ignore
            query = "<video>\n"+query
        else:
            image_tensors = None

        conversations = [{"from": "human", "value": query},
                         {"from": "gpt", "value": None}]
        conversations, variables = format_conversations(conversations)
        sources = preprocess_multimodal(
            copy.deepcopy([conversations]), self.vision_config)
        input_ids = preprocess_qwen([sources[0][0], {
                                    'from': 'gpt', 'value': None}], self.tokenizer, has_image=True).cuda()
        variables = [variables]
        output_ids = self.model.generate(
            input_ids,
            images=[image_tensors],  # type: ignore
            do_sample=True,
            temperature=0.01,
            top_p=None,
            num_beams=1,
            variables=variables,
            modalities=[modality],
            max_new_tokens=1024,
            use_cache=True)

        outputs = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=False)[0]
        outputs = outputs.replace("<|im_end|>", "")
        converted_outputs = replace_and_normalize(outputs)
        return converted_outputs


def visualize(conversations, args):
    task = args.task
    data_path = args.data_path
    split = eval(args.split)
    output_path = osp.join(args.output_dir, args.task, osp.basename(data_path))
    func = globals()[task]
    if args.task.lower() in ["reg", "tr"]:
        outputs = conversations["human"]
    else:
        outputs = conversations["gpt"]
    results = parse_results(outputs, task)
    func(results, data_path, output_path, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="appletea2333/LLaVA-ST-Qwen2-7B")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--modality", type=str, default="video")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--split", type=str, default="None")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--task", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="demo_outputs_hf")
    args = parser.parse_args()

    chat = Chat(args.model_path)
    outputs = chat.chat(
        args.query, 
        args.data_path,
        args.modality, 
        split=eval(args.split)
    )
    query = parse_inputs(args.query)
    print("############")
    print(f"human: {query}\ngpt: {outputs}")
    print("############")

    conversations = {"human": query, "gpt": outputs}
    file_name = osp.basename(args.data_path)
    os.makedirs(osp.join(args.output_dir, args.task, file_name), exist_ok=True)
    chat_file = osp.join(args.output_dir, args.task, file_name, "chat.json")

    with open(chat_file, "a") as fp:
        json.dump(conversations, fp)
        fp.write("\n")
    if args.show:
        visualize(conversations, args)
