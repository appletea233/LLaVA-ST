import argparse
import json
import os
import os.path as osp
import re

from inference.src.metrics import eval_rec, eval_stvg, eval_tvg
from inference.src.utils import load_jsonl

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, default="")
args = parser.parse_args()

output_dir = osp.join(args.result_dir, "total_eval.json")
result_files = []
for root, _, files in os.walk(args.result_dir):
    for file in files:
        if file.endswith(".json"):
            result_files.append(osp.join(root, file))

res_dic = {}

for file in result_files:
    if "/rec/" in file:
        try:
            results = load_jsonl(file)
        except Exception as e:
            print(f"Exception {e}")
            print(file)
        if len(results) == 0:
            continue
        miou, r_5 = eval_rec(results)
        res_dic[file.split("/")[-1][:-5]] = {
            "miou": miou,
            "R1@(0.5)": r_5
        }
    elif "/tvg/" in file:
        try:
            results = load_jsonl(file)
        except Exception as e:
            print(f"Exception {e}")
            print(file)
        if len(results) == 0:
            continue
        miou, r_3, r_5, r_7 = eval_tvg(results)
        res_dic[file.split("/")[-1][:-5]] = {
            "miou": miou,
            "R1@(0.3)": r_3,
            "R1@(0.5)": r_5,
            "R1@(0.7)": r_7
        }
    elif "/st-align/" in file:
        results = load_jsonl(file)
        if len(results) == 0:
            continue

        if "stvg" in file:
            task = "stvg"
        elif "svg" in file:
            task = "svg"
        elif "elc" in file:
            task = "elc"
        else:
            raise ValueError(f"Unknown task: {file}")

        metric = eval_stvg(results, task)
        res_dic[file.split("/")[-1][:-5]] = metric

for k, v in res_dic.items():
    print("-" * 100)
    print(f"task: {k}")
    for _k, _v in v.items():
        print(f"{_k}: {_v}")

with open(output_dir, "w") as fp:
    json.dump(res_dic, fp, indent=4)
