import json
import re
from typing import Union


def print_cuda_memory(total_num):
    import pynvml
    pynvml.nvmlInit()
    for gpu_id in range(total_num):
        if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
            print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
            return 0, 0, 0

        handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
        total = round(int(meminfo.total) / (1024**3), 2)
        used = round(int(meminfo.used) / (1024**3), 2)
        free = round(int(meminfo.free) / (1024**3), 2)
        print(f"===== cuda {gpu_id} =====")
        print(f"total: {total} GB")
        print(f"used: {used} GB")
        print(f"free: {free} GB")
        print("==========================")


def temporal_iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    _iou = max(min1 - max0, 0) / (max1 - min0)
    return max(0, _iou)


def iou(box1, box2):

    def s(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    intersection = max(0,
                       (min(box1[2], box2[2]) - max(box1[0], box2[0]))) * max(
                           0, (min(box1[3], box2[3]) - max(box1[1], box2[1])))
    intersection = max(0, intersection)
    union = s(box1) + s(box2) - intersection
    return intersection / union if union != 0 else 0


def load_jsonl(ann_path):
    lis = []
    with open(ann_path, "r") as fp:
        for line in fp:
            try:
                lis.append(json.loads(line))
            except Exception as e:
                print(f"Find expception {e}, ignore line.")
                continue
    return lis


def load_json(ann_path):
    with open(ann_path, "r") as fp:
        anns = json.load(fp)
    return anns


def parse_span_from_text(s):
    pattern = r"{\s*(\d+(?:\.\d+)?)\,\s*(\d+(?:\.\d+)?)\s*}"
    match = re.search(pattern, s)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return [start_time, end_time]
    else:
        print("No match found.")
        return [0, 0]


def parse_box_from_text(
    text,
    coords_pattern=(r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
                    r"\s*(\d+(?:\.\d+)?)\]")):
    text = text.replace(" ", "")
    # print(text)
    raw_coords = re.findall(coords_pattern, text)
    if not raw_coords or len(raw_coords[0]) != 4:
        print(text)
        return None
    return list(map(float, raw_coords[0]))


def parse_stpair_from_text(
    text,
    pattern=r"(\d+(?:\.\d+)?)\,\:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
    r"\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]",
):
    text = text.replace(" ", "")
    # print(text)
    raw_coords = re.findall(pattern, text)
    dic = {}
    for i in raw_coords:
        dic[float(i[0])] = list(map(float, i[1:]))
    return dic


def parse_float_from_text(s, peer_str="Ends in", post_str=""):
    pattern = peer_str + r"\s*(\d+(?:\.\d+)?)" + post_str
    match = re.search(pattern, s)
    if match:
        end_time = float(match.group(1))
        return end_time
    else:
        print("No match found.")
        return 0


def replace_and_normalize(input_str, return_token=False):
    pattern = re.compile(r'(<WIDTH-(\d+)>|<HEIGHT-(\d+)>|<TEMP-(\d+)>)')

    def normalize(match):
        if match.group(2):
            value = int(match.group(2))
        elif match.group(3):
            value = int(match.group(3))
        elif match.group(4):
            value = int(match.group(4))

        normalized_value = value / 99.0

        if return_token:
            return '{:d},'.format(value)
        return '{:.5f},'.format(normalized_value)

    # 使用 re.sub 进行替换，调用 normalize 函数进行处理
    result_str = re.sub(pattern, normalize, input_str)

    return result_str.replace(",]", "]").replace(",}", "}")


def filter_svg_bboxes_according_to_frame_id(bboxes: Union[dict, list],
                                            frame_id: list[int]):
    if isinstance(bboxes, dict):
        keys = list(bboxes.keys())
        for k in keys:
            if int(k) not in frame_id:
                del bboxes[k]
    return bboxes


def everytype2str(a):
    pass


def format_1d_box(text, ):
    pattern = r"{\s*(\d+(?:\.\d+)?)\,\s*(\d+(?:\.\d+)?)\s*}"
    match = re.search(pattern, text)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return start_time, end_time
    else:
        # print("No match found.")
        return None


def bbox_post_refine(bbox, height, width):
    if height >= width:
        x1, y1, x2, y2 = (i * height for i in bbox)
        pad = (height - width) // 2
        x1 -= pad
        x2 -= pad
    else:
        x1, y1, x2, y2 = (i * width for i in bbox)
        pad = (width - height) // 2
        y1 -= pad
        y2 -= pad
    res = [x1 / width, y1 / height, x2 / width, y2 / height]
    return res


def format_2d_box(text):
    pattern = (r"\[\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
               r"\s*(\d+(?:\.\d+)?)\s*\]")
    match = re.search(pattern, text)
    if match:
        a = float(match.group(1))
        b = float(match.group(2))
        c = float(match.group(3))
        d = float(match.group(4))
        return [a, b, c, d]
    else:
        # print("No match found.")
        return None


def format_box_in_text(text: str, pad_proc=False, **kwargs):
    box = format_2d_box(text)
    if box is None:
        return text
    in_out = kwargs["in_out"]
    pattern = (r"\[\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),"
               r"\s*(\d+(?:\.\d+)?)\s*\]")

    def replace_inside_braces(match):
        a = float(match.group(1))
        b = float(match.group(2))
        c = float(match.group(3))
        d = float(match.group(4))
        box = [a, b, c, d]
        if pad_proc:
            box = bbox_post_refine(box, kwargs["h"], kwargs["w"])

        def clip(x):
            return x if x >= 0 and x <= 1 else (0 if x < 0 else 1)

        box = [round(clip(x), 3) for x in box]
        return (f" [<WIDTH-{in_out}{box[0]}><HEIGHT-{in_out}{box[1]}>"
                f"<WIDTH-{in_out}{box[2]}><HEIGHT-{in_out}{box[3]}>]")

    res = re.sub(pattern, replace_inside_braces, text)
    return res


def format_span_in_text(text: str, in_out: str):
    span = format_1d_box(text)
    if span is None:
        return text
    pattern = r"{\s*(\d+(?:\.\d+)?)\,\s*(\d+(?:\.\d+)?)\s*}"

    def replace_inside_braces(match):
        s = float(match.group(1))
        e = float(match.group(2))
        if s >= e:
            print(f"start {s} >= end {e}")
            return "{<error>}"
        return (f" {'{'}<TEMP-{in_out}{round(s,3)}>"
                f"<TEMP-{in_out}{round(e,3)}>{'}'} ")

    res = re.sub(pattern, replace_inside_braces, text)
    return res


def format_float_in_text(text: str, in_out: str):
    pattern = r"Starts in (\d+(?:\.\d+)?)"

    def replace_inside_braces(match):
        s = float(match.group(1))
        return f" Starts in <TEMP-{in_out}{round(s,3)}> "

    res = re.sub(pattern, replace_inside_braces, text)
    return res


def seperate_token_number(text, token):
    pattern = rf'<{token}(.*?)>'
    matches = re.finditer(pattern, text)
    lis = [(i.group(1), i.start()) for i in matches]
    lis = sorted(lis, key=lambda x: x[-1])
    values = []
    for k, _ in lis:
        text = re.sub(re.escape(f"<{token}{k}>"), f"<{token}>", text, count=1)
        values.append(eval(k))
    return text, values


def get_variables(conversations):
    # Extract numerical information from tokens
    variables_dict = {
        "TEMP-INPUT": [],
        "TEMP-OUTPUT": [],
        "HEIGHT-INPUT": [],
        "HEIGHT-OUTPUT": [],
        "WIDTH-INPUT": [],
        "WIDTH-OUTPUT": []
    }
    for con in conversations:
        text = con["value"]
        if text is None:
            continue
        for key in variables_dict.keys():
            text, lis = seperate_token_number(text, key)
            variables_dict[key].extend(lis)
        con["value"] = text
    variables = {
        "temporal_input_locations": variables_dict["TEMP-INPUT"],
        "temporal_output_locations": variables_dict["TEMP-OUTPUT"],
        "spatial_height_input_locations": variables_dict["HEIGHT-INPUT"],
        "spatial_height_output_locations": variables_dict["HEIGHT-OUTPUT"],
        "spatial_width_input_locations": variables_dict["WIDTH-INPUT"],
        "spatial_width_output_locations": variables_dict["WIDTH-OUTPUT"]
    }
    return conversations, variables
