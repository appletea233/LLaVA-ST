from inference.src.utils import *
import re
import sys


def parse_inputs(inputs):
    pattern = r"<([twh])(\d+(?:\.\d+)?)>"

    def replace_inside_braces(match):
        a = match.group(1)
        b = float(match.group(2))
        return str(round(b, 3))
    res = re.sub(pattern, replace_inside_braces, inputs)
    return res


def parse_results(outputs, task):
    if task.lower() in ["rec", "reg"]:
        results = parse_box_from_text(outputs)
    if task.lower() in ["tvg", "tr"]:
        results = parse_span_from_text(outputs)
    if task.lower() in ["stvg", "svg", "elc"]:
        results = parse_stpair_from_text(outputs)
    if task.lower() == "dgc":
        results = []
        lis = outputs.split("]")
        for i in lis:
            box = parse_box_from_text(i+"]")
            if box is not None:
                results.append(box)
    if task.lower() == "dvc":
        pred_lis = outputs.split("{")
        pred_lis = [p for p in pred_lis if '}' in p]
        results = []
        for pred in pred_lis:
            pred = "{"+pred
            caption = pred.split(",")[-1].split(".")[0]
            timestamp = parse_span_from_text(pred)
            if timestamp != [0, 0]:
                results.append({
                    'timestamp': timestamp,
                    'sentence': caption,
                })
    return results


def format_text(text: str, mode: str):
    mapp = {"t": "TEMP", "w": "WIDTH", "h": "HEIGHT"}
    pattern = r"<([twh])(\d+(?:\.\d+)?)>"

    def replace_inside_braces(match):
        a = match.group(1)
        b = float(match.group(2))
        return f"<{mapp[a]}-{mode}{round(b,3)}>"
    res = re.sub(pattern, replace_inside_braces, text)
    return res


def format_conversations(conversations):
    for con in conversations:
        if con["value"] is None:
            continue
        text = con["value"]
        mode = "INPUT" if con["from"] == "human" else "OUTPUT"
        text = format_text(text, mode)
        con["value"] = text
    return get_variables(conversations)


if __name__ == "__main__":
    query = "<t0.1> <t0.2> and [<w0.14>,<h0.35>,<w0.4>,<h0.9>]"
    conversations = [{"from": "human", "value": query},
                     {"from": "gpt", "value": None}]
    conversations, variables = format_conversations(conversations)
    print()
