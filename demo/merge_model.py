import argparse
from typing import Any, Tuple
import os
import os.path as osp
import json
import tqdm
import copy
import sys
sys.path.append("./")
sys.path.append("./inference")
import re
import math
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from inference.src.datasets import *
from inference.src.utils import *
import warnings

# 禁用所有警告
warnings.filterwarnings("ignore")
import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import sys
sys.path.append("../../")

import time
import random
import yaml
import math
import re
import torch

import transformers

from llava.constants import *
from llava import conversation as conversation_lib
from decord import VideoReader,cpu

from llava.model import *
import tqdm
from peft import PeftModel
from llava.model.builder import load_pretrained_model, load_lora_model
from accelerate import Accelerator

model_path = "checkpoints/stage1"
lora_path = ["checkpoints/stage2", "checkpoints/stage3"]

output_path = "checkpoints/llava_st_qwen2"
tokenizer, model, image_processor, max_length = load_lora_model(lora_path, model_path, "llava_qwen", device_map="auto", overwrite_config={"num_spatial_tokens":100, "num_temporal_tokens":100})  # Add any other thing you want to pass in llava_model_args

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)