import os
import pickle
from typing import Any, Dict, List, Tuple

import datasets
import numpy as np
import skia
import torch
from logzero import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from typography_generation.visualization.renderer_util import (
    get_skia_font,
    get_text_actual_width,
    get_texts,
)


def get_scaleinfo(
    element_data: Dict,
) -> List:
    svgid = element_data["id"]
    scaleinfo = svgid2scaleinfo[svgid]
    return list(scaleinfo)


def get_canvassize(
    bg_img: Any,
) -> List:
    img_size = (bg_img.size[1], bg_img.size[0])
    return list(img_size)


def get_canvas_bg_img_emb(bg_img: Any, **kwargs: Any) -> np.array:
    inputs = processor(images=[bg_img], return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    image_feature = model.get_image_features(**inputs)
    return list(image_feature.data.cpu().numpy().flatten())


def get_text_emb_list(
    element_data: Dict,
) -> List:
    text_emb_list: List[List]
    text_emb_list = []
    for k in range(len(element_data["text"])):
        text = element_data["text"][k]
        inputs = text_tokenizer([text], padding=True, return_tensors="pt")
        if inputs["input_ids"].shape[1] > 77:
            inp = inputs["input_ids"][:, :77]
        else:
            inp = inputs["input_ids"]
        inp = inp.to(device)
        text_features = model.get_text_features(inp).data.cpu().numpy()[0]
        text_emb_list.append(text_features)
    return text_emb_list


def _get_text_actual_width(
    element_data: Dict,
    W: int,
) -> List:
    svgid = element_data["id"]
    scaleinfo = svgid2scaleinfo[svgid]
    scale_h, scale_w = scaleinfo
    text_actual_width_list = []
    element_num = len(element_data["text"])

    for i in range(element_num):
        if element_data["text"][i] == "":
            text_actual_width_list.append(None)
        else:
            texts = get_texts(element_data, i)

            font_label = element_data["font"][i]
            font_name = fontlabel2fontname(int(font_label))
            font_name = font_name.replace(" ", "_")
            font_skia, _ = get_skia_font(
                font2ttf, fontmgr, element_data, i, font_name, scale_h
            )
            text_width = get_text_actual_width(
                element_data, i, texts, font_skia, scale_w
            )
            text_actual_width_list.append(text_width / float(W))
    return text_actual_width_list


def get_text_center_y(element_data: dict, text_index: int) -> float:
    text_height = element_data["height"][text_index]
    top = element_data["top"][text_index]
    center_y = (top + top + text_height) / 2.0
    return float(center_y)


def get_text_center_y_list(
    element_data: dict,
) -> List:
    text_center_y_list = []
    element_num = len(element_data["text"])
    for text_id in range(element_num):
        if element_data["text"][text_id] == "":
            text_center_y_list.append(None)
        else:
            text_center_y_list.append(get_text_center_y(element_data, text_id))
    return text_center_y_list


def get_text_center_x(
    element_data: dict,
    text_index: int,
    text_actual_width: float,
) -> float:
    left = element_data["left"][text_index]
    w = element_data["width"][text_index]
    textAlign = element_data["text_align"][text_index]
    right = left + w
    if textAlign == 1:
        center_x = (left + right) / 2.0
    elif textAlign == 3:
        center_x = right - text_actual_width / 2.0
    elif textAlign == 2:
        center_x = left + text_actual_width / 2.0
    return float(center_x)


def get_text_center_x_list(
    element_data: dict,
    text_actual_width: float,
) -> List:
    svgid = element_data["id"]
    text_center_x_list = []
    element_num = len(element_data["text"])
    for text_id in range(element_num):
        if element_data["text"][text_id] == "":
            text_center_x_list.append(None)
        else:
            _text_actual_width = text_actual_width[text_id]
            text_center_x_list.append(
                get_text_center_x(element_data, text_id, _text_actual_width)
            )
    return text_center_x_list


def get_text_local_img(
    img: Any,
    text_center_y: float,
    text_center_x: float,
    H: int,
    W: int,
) -> np.array:
    text_center_y = text_center_y * H
    text_center_x = text_center_x * W

    text_center_y = min(max(text_center_y, 0), H)
    text_center_x = min(max(text_center_x, 0), W)
    img = img.resize((640, 640))
    img = np.array(img)
    local_img_size = 64 * 5
    local_img_size_half = local_img_size // 2
    img_pad = np.zeros((640 + local_img_size, 640 + local_img_size, 3))
    img_pad[
        local_img_size_half : 640 + local_img_size_half,
        local_img_size_half : 640 + local_img_size_half,
    ] = img
    h_rate = 640 / float(H)
    w_rate = 640 / float(W)
    text_center_y = int(np.round(text_center_y * h_rate + local_img_size_half))
    text_center_x = int(np.round(text_center_x * w_rate + local_img_size_half))
    local_img = img_pad[
        text_center_y - local_img_size_half : text_center_y + local_img_size_half,
        text_center_x - local_img_size_half : text_center_x + local_img_size_half,
    ]
    return local_img


def get_text_local_img_emb_list(
    element_data: Dict,
    bg_img: Any,
    text_center_y: float,
    text_center_x: float,
) -> List:
    text_local_img_emb_list: List[List]
    text_local_img_emb_list = []
    for k in range(len(element_data["text"])):
        if element_data["text"][k] == "":
            text_local_img_emb_list.append([])
        else:
            H, W = bg_img.size[1], bg_img.size[0]
            local_img = get_text_local_img(
                bg_img.copy(), text_center_y[k], text_center_x[k], H, W
            )
            local_img = Image.fromarray(local_img.astype(np.uint8)).resize((224, 224))
            inputs = processor(images=[local_img], return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(device)
            image_feature = model.get_image_features(**inputs)
            text_local_img_emb_list.append(image_feature.data.cpu().numpy())

    return text_local_img_emb_list


def get_orderlist(
    center_y: List[float],
    center_x: List[float],
) -> List:
    """
    Sort elments based on the raster scan order.
    """
    center_y = [10000 if y is None else y for y in center_y]
    center_x = [10000 if x is None else x for x in center_x]
    center_y = np.array(center_y)
    center_x = np.array(center_x)
    sortedid = np.argsort(center_y * 1000 + center_x)
    return list(sortedid)


def add_features(
    element_data: Dict,
) -> Dict:
    svgid = element_data["id"]
    fn = os.path.join(data_dir, "generate_bg_png", f"{svgid}.png")
    bg_img = Image.open(fn).convert("RGB")  # background image
    element_data["scale_box"] = get_scaleinfo(element_data)
    element_data["canvas_bg_size"] = get_canvassize(bg_img)
    element_data["canvas_bg_img_emb"] = get_canvas_bg_img_emb(bg_img)
    element_data["text_emb"] = get_text_emb_list(element_data)
    text_actual_width = _get_text_actual_width(element_data, bg_img.size[0])
    text_center_y = get_text_center_y_list(element_data)
    text_center_x = get_text_center_x_list(element_data, text_actual_width)
    element_data["text_center_y"] = text_center_y
    element_data["text_center_x"] = text_center_x
    element_data["text_actual_width"] = text_actual_width
    element_data["text_local_img_emb"] = get_text_local_img_emb_list(
        element_data, bg_img, text_center_y, text_center_x
    )
    element_data["order_list"] = get_orderlist(text_center_y, text_center_x)
    return element_data


def map_features(
    _data_dir: str,
):
    fn = os.path.join(_data_dir, "svgid2scaleinfo.pkl")
    global svgid2scaleinfo
    global processor
    global text_tokenizer
    global model
    global device
    global data_dir
    global font2ttf
    global fontmgr
    global fontlabel2fontname

    dataset = datasets.load_dataset("cyberagent/crello")

    data_dir = _data_dir
    svgid2scaleinfo = pickle.load(open(fn, "rb"))
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fn = os.path.join(data_dir, "font2ttf.pkl")
    _font2ttf = pickle.load(open(fn, "rb"))
    font2ttf = {}
    for key in _font2ttf.keys():
        tmp = _font2ttf[key].split("/data/dataset/crello/")[1]
        fn = os.path.join(data_dir, tmp)
        font2ttf[key] = fn
    font2ttf = font2ttf
    fontmgr = skia.FontMgr()
    fontlabel2fontname = dataset["train"].features["font"].feature.int2str

    dataset_new = {}
    for dataset_division in ["train", "validation", "test"]:
        logger.info(f"{dataset_division=}")
        _dataset = dataset[dataset_division]
        dataset_new[dataset_division] = _dataset.map(add_features)
    dataset_new = datasets.DatasetDict(dataset_new)
    dataset_new.save_to_disk(f"{_data_dir}/crello_map_features")
