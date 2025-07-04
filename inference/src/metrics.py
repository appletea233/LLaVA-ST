import collections
import random
import string

import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from inference.src.utils import (iou, parse_box_from_text,
                                 parse_span_from_text, parse_stpair_from_text,
                                 replace_and_normalize, temporal_iou)


def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def eval_rec(results):
    total_iou = 0
    num = 0
    r_5 = 0
    for result in tqdm.tqdm(results):
        gt_box = result["gt_bbox"]
        pred_box = parse_box_from_text(result["converted_outputs"])
        if pred_box is None:
            print("no box predicted")
            continue
        _iou = iou(gt_box, pred_box)
        num += 1
        total_iou += _iou
        if _iou >= 0.5:
            r_5 += 1
    if num == 0:
        num = 1
    print("miou: ", str(total_iou / num))
    print("R1@(0.5): ", str(r_5 / num * 100))

    return total_iou / num, r_5 / num * 100


def eval_tvg(results):
    lis = results
    total_iou = 0
    r_3 = 0
    r_5 = 0
    r_7 = 0
    for i in tqdm.tqdm(range(len(lis))):
        pred = lis[i]["converted_outputs"]
        gt_box = lis[i]["gt"]
        pred_box = parse_span_from_text(pred)
        if pred_box == [0, 0]:
            continue
        _iou = temporal_iou(pred_box, gt_box)
        total_iou += _iou
        if _iou > 0.3:
            r_3 += 1
        if _iou > 0.5:
            r_5 += 1
        if _iou > 0.7:
            r_7 += 1

    miou = total_iou / len(lis)
    r_3 /= (len(lis) / 100)
    r_5 /= (len(lis) / 100)
    r_7 /= (len(lis) / 100)
    print("miou: ", str(miou))
    print("R@1(0.3): ", str(r_3))
    print("R@1(0.5): ", str(r_5))
    print("R@1(0.7): ", str(r_7))
    return miou, r_3, r_5, r_7


def eval_stvg(results, task="stvg", num_frames=100):
    anns = results

    mtiou = 0
    msiou = 0
    mstiou = 0
    str_3 = 0
    str_5 = 0
    str_7 = 0
    tr_3, tr_5, tr_7, sr_3, sr_5, sr_7 = 0, 0, 0, 0, 0, 0
    for ann in tqdm.tqdm(anns):
        if task == "stvg":
            split = ann["split"]
            duration = split[1] - split[0] + 1
            width, height = ann["width"], ann["height"]

            converted_outputs = replace_and_normalize(ann["outputs"],
                                                      return_token=True)
            tvg_pred = parse_span_from_text(ann["converted_outputs"])
            svg_pred = parse_stpair_from_text(converted_outputs)
            tvg_gt = ann["gt"]["tvg"]
            svg_gt = ann["gt"]["svg"]
            tvg_gt = [
                round((i - split[0]) / (duration - 1), 3) for i in tvg_gt
            ]
            frame_idx = [
                split[0] + round((i / (num_frames - 1)) * (duration - 1))
                for i in range(num_frames)
            ]
            temp = {}
            for k, v in svg_gt.items():
                assert int(k) in frame_idx
            for k, v in svg_pred.items():
                k = split[0] + round(
                    float(k) / (num_frames - 1) * (duration - 1))
                assert int(k) in frame_idx
            for k, v in svg_pred.items():
                t = str(
                    round(split[0] + float(k) / (num_frames - 1) *
                          (duration - 1)))
                v = [i / 99 for i in v]
                box = [
                    round(i) for i in
                    [v[0] * width, v[1] * height, v[2] * width, v[3] * height]
                ]
                temp[t] = box
            svg_pred = temp
            tiou = temporal_iou(tvg_gt, tvg_pred)

            timestamps_union = list(set(svg_gt.keys()) | set(svg_pred.keys()))
            timestamps_inter = list(set(svg_gt.keys()) & set(svg_pred.keys()))
            iou_sum = 0
            for time in timestamps_inter:
                iou_sum += iou(svg_gt[time], svg_pred[time])
            siou = iou_sum / max(len(timestamps_inter), 1)
            stiou = iou_sum / max(len(timestamps_union), 1)

        elif task == "svg":
            split = ann["split"]
            duration = split[1] - split[0] + 1
            width, height = ann["width"], ann["height"]

            converted_outputs = replace_and_normalize(ann["outputs"],
                                                      return_token=True)
            tvg_pred = parse_span_from_text(ann["converted_outputs"])
            svg_pred = parse_stpair_from_text(converted_outputs)
            svg_gt = ann["gt"]["svg"]
            frame_idx = [
                split[0] + round((i / (num_frames - 1)) * (duration - 1))
                for i in range(num_frames)
            ]
            temp = {}
            for k, v in svg_gt.items():
                assert int(k) in frame_idx
            for k, v in svg_pred.items():
                k = split[0] + round(
                    float(k) / (num_frames - 1) * (duration - 1))
                assert int(k) in frame_idx
            for k, v in svg_pred.items():
                t = str(
                    round(split[0] + float(k) / (num_frames - 1) *
                          (duration - 1)))
                v = [i / 99 for i in v]
                box = [
                    round(i) for i in
                    [v[0] * width, v[1] * height, v[2] * width, v[3] * height]
                ]
                temp[t] = box
            svg_pred = temp

            timestamps_union = list(set(svg_gt.keys()) | set(svg_pred.keys()))
            timestamps_inter = list(set(svg_gt.keys()) & set(svg_pred.keys()))
            iou_sum = 0
            for time in timestamps_inter:
                iou_sum += iou(svg_gt[time], svg_pred[time])
            siou = iou_sum / max(len(timestamps_inter), 1)
            stiou = iou_sum / max(len(timestamps_union), 1)
            tiou = 0

        elif task == "elc":
            split = ann["split"]
            duration = split[1] - split[0] + 1
            width, height = ann["width"], ann["height"]

            converted_outputs = replace_and_normalize(ann["outputs"],
                                                      return_token=True)
            tvg_pred = parse_span_from_text(ann["converted_outputs"])
            svg_pred = parse_stpair_from_text(converted_outputs)
            tvg_gt = ann["gt"]["tvg"]
            svg_gt = ann["gt"]["svg"]
            tvg_gt = [
                round((i - split[0]) / (duration - 1), 3) for i in tvg_gt
            ]
            frame_idx = [
                split[0] + round((i / (num_frames - 1)) * (duration - 1))
                for i in range(num_frames)
            ]
            temp = {}
            for k, v in svg_gt.items():
                assert int(k) in frame_idx
            for k, v in svg_pred.items():
                k = split[0] + round(
                    float(k) / (num_frames - 1) * (duration - 1))
                assert int(k) in frame_idx
            for k, v in svg_pred.items():
                t = str(
                    round(split[0] + float(k) / (num_frames - 1) *
                          (duration - 1)))
                v = [i / 99 for i in v]
                box = [
                    round(i) for i in
                    [v[0] * width, v[1] * height, v[2] * width, v[3] * height]
                ]
                temp[t] = box
            svg_pred = temp
            tiou = temporal_iou(tvg_gt, tvg_pred)

            tiou = temporal_iou(tvg_gt, tvg_pred)

            timestamps_union = list(set(svg_gt.keys()) | set(svg_pred.keys()))
            timestamps_inter = list(set(svg_gt.keys()) & set(svg_pred.keys()))
            iou_sum = 0
            for time in timestamps_inter:
                iou_sum += iou(svg_gt[time], svg_pred[time])
            siou = iou_sum / max(len(timestamps_inter), 1)
            stiou = iou_sum / max(len(timestamps_union), 1)

        mtiou += tiou
        msiou += siou
        mstiou += stiou
        str_3 += 1 if stiou >= 0.3 else 0
        str_5 += 1 if stiou >= 0.5 else 0
        str_7 += 1 if stiou >= 0.7 else 0
        tr_3 += 1 if tiou >= 0.3 else 0
        tr_5 += 1 if tiou >= 0.5 else 0
        tr_7 += 1 if tiou >= 0.7 else 0
        sr_3 += 1 if siou >= 0.3 else 0
        sr_5 += 1 if siou >= 0.5 else 0
        sr_7 += 1 if siou >= 0.7 else 0
    eval_scores = {}
    if task == "elc":
        for num, ann in enumerate(anns):
            ann["idx"] = num
            try:
                ann["gt_caption"] = ann["gt"]["caption"]
            except Exception:
                ann["gt_caption"] = ''
            ann["pred_caption"] = ann["outputs"].split(".")[0].split(",")[-1]
        scorers = [(Bleu(1), "Bleu_1"), (Bleu(2), "Bleu_2"),
                   (Bleu(3), "Bleu_3"), (Bleu(4), "Bleu_4"),
                   (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"),
                   (Cider(), "CIDEr")]  # [(Spice(), "SPICE")]
        scorers_dict = {s[1]: s for s in scorers}

        gts = collections.defaultdict(list)
        preds = collections.defaultdict(list)
        for ann in anns:
            gts[ann["idx"]].append({"caption": ann["gt_caption"]})
            preds[ann["idx"]].append({"caption": ann["pred_caption"]})

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(preds)

        for metric in scorers_dict.keys():
            score, scores = scorers_dict[metric][0].compute_score(gts, res)
            if isinstance(score, list):
                score = [round(i, 3) for i in score]
            else:
                score = round(score, 3)
            eval_scores[metric] = score
    mtiou /= len(anns)
    msiou /= len(anns)
    mstiou /= len(anns)
    str_3 /= (len(anns) / 100)
    str_5 /= (len(anns) / 100)
    str_7 /= (len(anns) / 100)
    tr_3 /= (len(anns) / 100)
    tr_5 /= (len(anns) / 100)
    tr_7 /= (len(anns) / 100)
    sr_3 /= (len(anns) / 100)
    sr_5 /= (len(anns) / 100)
    sr_7 /= (len(anns) / 100)
    print(f"mtiou: {mtiou}")
    print(f"msiou: {msiou}")
    print(f"mstiou: {mstiou}")
    print(f"viou@0.3: {str_3}")
    print(f"viou@0.5: {str_5}")
    return {
        "mtiou": mtiou,
        "tiou@0.3": tr_3,
        "tiou@0.5": tr_5,
        "tiou@0.7": tr_7,
        "msiou": msiou,
        "siou@0.3": sr_3,
        "siou@0.5": sr_5,
        "siou@0.7": sr_7,
        "mstiou": mstiou,
        "viou@0.3": str_3,
        "viou@0.5": str_5,
        "viou@0.7": str_7,
        "caption_score": eval_scores
    }
