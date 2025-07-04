import cv2
import os
import os.path as osp
import decord
from inference.src.datasets import *
import matplotlib.pyplot as plt
import numpy as np


def draw_boxes(image, boxes, labels=None):
    image_copy = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    color = (0, 255, 0)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)

        if labels is not None:
            label = labels[i]
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.rectangle(image_copy, (text_x, text_y -
                          text_size[1]), (text_x + text_size[0], text_y), color, -1)
            cv2.putText(image_copy, label, (text_x, text_y),
                        font, font_scale, (0, 0, 0), thickness)

    return image_copy


def rec(results, data_path, output_path, *args, **kwargs):
    image = cv2.imread(data_path)
    h, w, _ = image.shape
    x1, y1, x2, y2 = results
    results = [x1*w, y1*h, x2*w, y2*h]
    results = [int(i) for i in results]
    boxes = [results]
    result_image = draw_boxes(image, boxes)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(result_image)
    plt.axis('off')  # 不显示坐标轴
    plt.margins(0, 0)
    plt.savefig(osp.join(output_path, f"rec.jpg"),
                dpi=400, bbox_inches='tight', pad_inches=0)


def reg(results, data_path, output_path, *args, **kwargs):
    image = cv2.imread(data_path)
    h, w, _ = image.shape
    x1, y1, x2, y2 = results
    results = [x1*w, y1*h, x2*w, y2*h]
    results = [int(i) for i in results]
    boxes = [results]
    result_image = draw_boxes(image, boxes)
    cv2.imwrite(osp.join(output_path, "reg.jpg"), result_image)


def dgc(results, data_path, output_path, *args, **kwargs):
    image = Image.open(data_path).convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    boxes = []
    for box in results:
        x1, y1, x2, y2 = box
        box = [x1*w, y1*h, x2*w, y2*h]
        box = [int(i) for i in box]
        boxes.append(box)
    result_image = draw_boxes(image, boxes)
    cv2.imwrite(osp.join(output_path, "dgc.jpg"), result_image)


def draw_frames(frames, timestamps, output_path):
    for num, image in enumerate(frames):
        plt.figure()
        plt.imshow(image)
        plt.axis('off')  # 不显示坐标轴
        plt.margins(0, 0)
        plt.savefig(osp.join(
            output_path, f"{num}_{timestamps[num]}.jpg"), dpi=400, bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def tvg(results, data_path, output_path, split=None):
    fps = VideoReader(data_path).get_avg_fps()
    frames, frame_ids = load_video(data_path, split=split, return_id=True)
    timestamps = [round(
        i/fps, 2) if split is None else round((i-split[0])/fps, 2) for i in frame_ids]
    tvg_gt = [round(results[0]*(len(frames)-1)),
              round(results[1]*(len(frames)-1))]
    grid_size = 10
    plt.figure()
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    for i, ax in enumerate(axes.flat):
        img = frames[i]
        ax.imshow(img)
        ax.axis('off')  # 不显示坐标轴
        color = "red" if i >= tvg_gt[0] and i <= tvg_gt[1] else "blue"
        ax.text(5, 5, str(i+1), color=color, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(osp.join(output_path, "tvg.jpg"))
    plt.close()
    os.makedirs(osp.join(output_path, "tvg"), exist_ok=True)
    draw_frames(frames, timestamps, osp.join(output_path, "tvg"))


def tr(results, data_path, output_path, split=None):
    fps = VideoReader(data_path).get_avg_fps()
    frames, frame_ids = load_video(data_path, split=split, return_id=True)
    timestamps = [round(
        i/fps, 2) if split is None else round((i-split[0])/fps, 2) for i in frame_ids]
    tvg_gt = [round(results[0]*(len(frames)-1)),
              round(results[1]*(len(frames)-1))]
    grid_size = 10
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    for i, ax in enumerate(axes.flat):
        img = frames[i]
        ax.imshow(img)
        ax.axis('off')  # 不显示坐标轴
        color = "red" if i >= tvg_gt[0] and i <= tvg_gt[1] else "blue"
        ax.text(5, 5, str(i+1), color=color, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(osp.join(output_path, "tr.jpg"))
    os.makedirs(osp.join(output_path, "tr"), exist_ok=True)
    draw_frames(frames, timestamps, osp.join(output_path, "tr"))


def stvg(results, data_path, output_path, split):
    fps = VideoReader(data_path).get_avg_fps()
    frames, frame_ids = load_video(data_path, split=split, return_id=True)
    timestamps = [round(
        i/fps, 2) if split is None else round((i-split[0])/fps, 2) for i in frame_ids]
    split_duration = len(frames)
    svg_pred = {}
    _, height, width, _ = frames.shape
    for t, box in results.items():
        timestamp = round(t*(split_duration-1))
        box = list(
            map(int, [box[0]*width, box[1]*height, box[2]*width, box[3]*height]))
        svg_pred[timestamp] = box

    images = []
    for i, image in enumerate(frames):
        if i in svg_pred:
            box = svg_pred[i]
            image = cv2.rectangle(
                image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=10)
        images.append(image)

    grid_size = 10
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    for i, ax in enumerate(axes.flat):
        img = images[i]
        ax.imshow(img)
        ax.axis('off')  # 不显示坐标轴
        ax.text(5, 5, str(i+1), color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    plt.savefig(osp.join(output_path, "stvg.jpg"))
    os.makedirs(osp.join(output_path, "stvg"), exist_ok=True)
    draw_frames(frames, timestamps, osp.join(output_path, "stvg"))

def svg(results, data_path, output_path, split):
    fps = VideoReader(data_path).get_avg_fps()
    frames, frame_ids = load_video(data_path, split=split, return_id=True)
    timestamps = [round(
        i/fps, 2) if split is None else round((i-split[0])/fps, 2) for i in frame_ids]
    split_duration = len(frames)
    svg_pred = {}
    _, height, width, _ = frames.shape
    for t, box in results.items():
        timestamp = round(t*(split_duration-1))
        box = list(
            map(int, [box[0]*width, box[1]*height, box[2]*width, box[3]*height]))
        svg_pred[timestamp] = box

    images = []
    for i, image in enumerate(frames):
        if i in svg_pred:
            box = svg_pred[i]
            image = cv2.rectangle(
                image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=10)
        images.append(image)

    grid_size = 10
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    for i, ax in enumerate(axes.flat):
        img = images[i]
        ax.imshow(img)
        ax.axis('off')  # 不显示坐标轴
        ax.text(5, 5, str(i+1), color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    plt.savefig(osp.join(output_path, "svg.jpg"))
    os.makedirs(osp.join(output_path, "svg"), exist_ok=True)
    draw_frames(frames, timestamps, osp.join(output_path, "svg"))


def elc(results, data_path, output_path, split):
    fps = VideoReader(data_path).get_avg_fps()
    frames, frame_ids = load_video(data_path, split=split, return_id=True)
    timestamps = [round(
        i/fps, 2) if split is None else round((i-split[0])/fps, 2) for i in frame_ids]
    split_duration = len(frames)
    svg_pred = {}
    _, height, width, _ = frames.shape
    for t, box in results.items():
        timestamp = round(t*(split_duration-1))
        box = list(
            map(int, [box[0]*width, box[1]*height, box[2]*width, box[3]*height]))
        svg_pred[timestamp] = box

    images = []
    for i, image in enumerate(frames):
        if i in svg_pred:
            box = svg_pred[i]
            image = cv2.rectangle(
                image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=10)
        images.append(image)

    grid_size = 10
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    for i, ax in enumerate(axes.flat):
        img = images[i]
        ax.imshow(img)
        ax.axis('off')  # 不显示坐标轴
        ax.text(5, 5, str(i+1), color='red', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    plt.savefig(osp.join(output_path, "elc.jpg"))
    os.makedirs(osp.join(output_path, "elc"), exist_ok=True)
    draw_frames(frames, timestamps, osp.join(output_path, "elc"))
