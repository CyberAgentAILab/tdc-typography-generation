import time
from typing import List

import numpy as np
import PIL
from logzero import logger

from typography_generation.io.crello_util import CrelloProcessor
from typography_generation.io.data_object import CanvasContext, ElementContext


def get_canvas_context(
    element_data: dict,
    dataset: CrelloProcessor,
    canvas_prefix_list: List,
    bg_img: np.array,
    text_num: int,
) -> CanvasContext:
    img_size = (bg_img.size[1], bg_img.size[0])  # (h,w)
    scale_box = dataset.get_scale_box(element_data)
    canvas_context = CanvasContext(
        bg_img, text_num, img_size, scale_box, canvas_prefix_list
    )
    for prefix in canvas_prefix_list:
        data_info = {"element_data": element_data, "bg_img": bg_img}
        data = getattr(dataset, f"get_{prefix}")(**data_info)
        setattr(canvas_context, f"{prefix}", data)
        setattr(canvas_context, f"{prefix}_model_input", data)

    return canvas_context


def get_element_context(
    element_data: dict,
    bg_img: PIL.Image,
    dataset: CrelloProcessor,
    element_prefix_list: List,
    text_num: int,
    text_ids: List,
) -> ElementContext:
    scaleinfo = dataset.get_scale_box(element_data)
    img_size = (bg_img.size[1], bg_img.size[0])  # (h,w)
    element_context = ElementContext(
        element_prefix_list,
        dataset.tokenizer.rawdata2token,
        img_size,
        scaleinfo,
        dataset.seq_length,
    )
    for i in range(text_num):
        text_index = text_ids[i]
        for prefix in element_prefix_list:
            start = time.time()

            data_info = {
                "element_data": element_data,
                "text_index": text_index,
                "img_size": img_size,
                "scaleinfo": scaleinfo,
                "bg_img": bg_img,
            }
            if prefix == "text_emb":
                data = getattr(dataset, f"get_{prefix}")(**data_info)
                element_context.text_emb_model_input[i] = data
            elif prefix == "text_local_img_emb":
                data = getattr(dataset, f"get_{prefix}")(**data_info)
                element_context.text_local_img_emb_model_input[i] = data
            elif prefix == "text_font_emb":
                data = dataset.get_text_font_emb(element_data, text_index)
                getattr(element_context, prefix).append(data)
                element_context.text_font_emb_model_input[i] = data
            else:
                data = getattr(dataset, f"get_{prefix}")(**data_info)
                getattr(element_context, prefix).append(data)
                if prefix in dataset.tokenizer.prefix_list:
                    model_input = dataset.tokenizer.tokenize(prefix, data)
                else:
                    model_input = data
                getattr(element_context, f"{prefix}_model_input")[i] = model_input

            if prefix in dataset.tokenizer.rawdata_list:
                token = getattr(dataset, f"raw2token_{prefix}")(data)
                getattr(
                    element_context,
                    f"{dataset.tokenizer.rawdata2token[prefix]}_model_input",
                )[i] = token

            end = time.time()
            logger.debug(f"{prefix} {end - start}")

    return element_context
