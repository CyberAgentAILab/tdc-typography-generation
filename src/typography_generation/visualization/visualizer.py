import copy
from typing import Dict, List, Tuple

import numpy as np
from typography_generation.tools.denormalizer import Denormalizer
from typography_generation.tools.tokenizer import Tokenizer
from typography_generation.visualization.renderer import TextRenderer

crelloattstr2pkgattstr = {
    "text_font": "font",
    "text_font_color": "color",
    "text_align_type": "text_align",
    "text_capitalize": "capitalize",
    "text_font_size": "font_size",
    "text_font_size_raw": "font_size",
    "text_angle": "angle",
    "text_letter_spacing": "letter_spacing",
    "text_line_height_scale": "line_height",
    "text_center_y": "text_center_y",
    "text_center_x": "text_center_x",
}


def get_text_ids(element_data: Dict) -> List:
    text_ids = []
    for k in range(len(element_data["text"])):
        if element_data["text"][k] == "":
            pass
        else:
            text_ids.append(k)
    return text_ids


def replace_style_data_by_prediction(
    prediction: Dict, element_data: Dict, text_ids: List
) -> Dict:
    element_data = copy.deepcopy(element_data)
    for prefix_pred, prefix_vec in crelloattstr2pkgattstr.items():
        if prefix_pred in prediction.keys():
            for i, t in enumerate(text_ids):
                element_data[prefix_vec][t] = prediction[prefix_pred][i][0]
    return element_data


def ordering_text_ids(order_list: List, text_ids: List) -> List:
    _text_ids = []
    for i in range(len(text_ids)):
        k = text_ids[int(order_list[i])]
        _text_ids.append(k)
    return _text_ids


def visualize_prediction(
    renderer: TextRenderer,
    element_data: Dict,
    prediction: Dict,
    bg_img: np.array,
) -> np.array:
    text_ids = get_text_ids(element_data)
    order_list = element_data["order_list"]
    scaleinfo = element_data["scale_box"]
    text_ids = ordering_text_ids(order_list, text_ids)
    element_data = replace_style_data_by_prediction(prediction, element_data, text_ids)
    img = renderer.draw_texts(element_data, text_ids, np.array(bg_img), scaleinfo)
    return img


def get_predicted_alphamaps(
    renderer: TextRenderer,
    element_data: Dict,
    prediction: Dict,
    image_size: Tuple[int, int],
    order_list: List = None,
) -> List:
    text_ids = get_text_ids(element_data)
    if order_list is None:
        order_list = element_data["order_list"]
    scaleinfo = element_data["scale_box"]
    text_ids = ordering_text_ids(order_list, text_ids)
    element_data = replace_style_data_by_prediction(prediction, element_data, text_ids)
    alpha_list = renderer.get_text_alpha_list(
        element_data, text_ids, image_size, scaleinfo
    )
    return alpha_list


def get_element_alphamaps(
    renderer: TextRenderer,
    element_data: Dict,
) -> List:
    text_ids = get_text_ids(element_data)
    order_list = element_data["order_list"]
    scaleinfo = element_data["scale_box"]
    image_size = element_data["canvas_bg_size"]
    text_ids = ordering_text_ids(order_list, text_ids)
    alpha_list = renderer.get_text_alpha_list(
        element_data, text_ids, image_size, scaleinfo
    )
    return alpha_list


def visualize_data(
    renderer: TextRenderer,
    element_data: Dict,
    bg_img: np.array,
) -> np.array:
    text_ids = get_text_ids(element_data)
    scaleinfo = element_data["scale_box"]
    img = renderer.draw_texts(element_data, text_ids, np.array(bg_img), scaleinfo)
    return img


def tokenize(
    _element_data: Dict,
    tokenizer: Tokenizer,
    denormalizer: Denormalizer,
    text_ids: List,
    bg_img: np.array,
    scaleinfo: Tuple,
) -> Dict:
    element_data = copy.deepcopy(_element_data)
    h, w = bg_img.size[1], bg_img.size[0]
    for prefix_pred, prefix_vec in crelloattstr2pkgattstr.items():
        for i, t in enumerate(text_ids):
            data_info = {
                "element_data": _element_data,
                "text_index": t,
                "img_size": (h, w),
                "scaleinfo": scaleinfo,
                "text_actual_width": _element_data["text_actual_width"][t],
                "text": None,
            }
            data = getattr(denormalizer.dataset, f"get_{prefix_pred}")(**data_info)
            if prefix_pred in denormalizer.dataset.tokenizer.prefix_list:
                data = tokenizer.tokenize(prefix_pred, data)
                data = tokenizer.detokenize(prefix_pred, data)
            data = denormalizer.denormalize_elm(
                prefix_pred, data, h, w, scaleinfo[0], scaleinfo[1]
            )
            element_data[prefix_vec][t] = data
    return element_data


def visualize_tokenization(
    renderer: TextRenderer,
    tokenizer: Tokenizer,
    denormalizer: Denormalizer,
    element_data: Dict,
    bg_img: np.array,
) -> np.array:
    text_ids = get_text_ids(element_data)
    scaleinfo = element_data["scale_box"]
    element_data = tokenize(
        element_data, tokenizer, denormalizer, text_ids, bg_img, scaleinfo
    )
    img = renderer.draw_texts(element_data, text_ids, np.array(bg_img), scaleinfo)
    return img


def get_text_coords(element_data: Dict, text_index: int, img_size: Tuple) -> Tuple:
    h, w = img_size
    top = int(element_data["top"][text_index] * h)
    left = int(element_data["left"][text_index] * w)
    height = int(element_data["height"][text_index] * h)
    width = int(element_data["width"][text_index] * w)
    return top, left, top + height, left + width


def colorize_text(
    element_data: Dict,
    canvas: np.array,
    text_index: int,
    color: Tuple = (255, 0, 0),
    w: float = 0.5,
) -> np.array:
    text_ids = get_text_ids(element_data)
    order_list = element_data["order_list"]
    scaleinfo = element_data["scale_box"]
    text_ids = ordering_text_ids(order_list, text_ids)
    y0, x0, y1, x1 = get_text_coords(
        element_data, text_ids[text_index], canvas.shape[:2]
    )
    tmp = canvas.copy()
    tmp[y0:y1, x0:x1, :] = np.array(color)
    canvas[y0:y1, x0:x1, :] = (
        w * canvas[y0:y1, x0:x1, :] + (1 - w) * tmp[y0:y1, x0:x1, :]
    )
    return canvas
