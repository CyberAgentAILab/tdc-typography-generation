from typing import Any, List, Tuple
import skia
import os
import pickle
import numpy as np

from typography_generation.visualization.renderer_util import (
    get_color_map,
    get_skia_font,
    get_text_actual_height,
    get_text_actual_width,
    get_text_alpha,
    get_texts,
)


class TextRenderer:
    def __init__(
        self,
        data_dir: str,
        fontlabel2fontname: Any,
    ) -> None:
        self.fontmgr = skia.FontMgr()
        fn = os.path.join(data_dir, "font2ttf.pkl")
        _font2ttf = pickle.load(open(fn, "rb"))
        font2ttf = {}
        for key in _font2ttf.keys():
            tmp = _font2ttf[key].split("/data/dataset/crello/")[1]
            fn = os.path.join(data_dir, tmp)
            font2ttf[key] = fn
        self.font2ttf = font2ttf
        fn = os.path.join(data_dir, "fonttype2fontid_fix.pkl")
        fonttype2fontid = pickle.load(open(fn, "rb"))

        self.fontid2fonttype = {}
        for k, v in fonttype2fontid.items():
            self.fontid2fonttype[v] = k
        self.fontlabel2fontname = fontlabel2fontname

    def draw_texts(
        self,
        element_data: dict,
        text_ids: List,
        bg: np.array,
        scaleinfo: Tuple,
    ) -> np.array:
        H, W = bg.shape[0], bg.shape[1]
        h_rate, w_rate = scaleinfo
        canvas = bg.copy()
        for text_id in text_ids:
            font_label = element_data["font"][text_id]
            font_name = self.fontlabel2fontname(int(font_label))
            font_name = font_name.replace(" ", "_")
            texts = get_texts(element_data, text_id)
            font, _ = get_skia_font(
                self.font2ttf,
                self.fontmgr,
                element_data,
                text_id,
                font_name,
                h_rate,
            )
            text_alpha = get_text_alpha(
                element_data,
                text_id,
                texts,
                font,
                H,
                W,
                w_rate,
            )
            text_rgb_map = get_color_map(element_data, text_id, H, W)
            canvas = canvas * (1 - text_alpha) + text_alpha * text_rgb_map
        return canvas

    def get_text_alpha_list(
        self,
        element_data: dict,
        text_ids: List,
        image_size: Tuple[int, int],
        scaleinfo: Tuple[float, float],
    ) -> List:
        H, W = image_size
        h_rate, w_rate = scaleinfo
        text_alpha_list = []
        for text_id in text_ids:
            font_label = element_data["font"][text_id]
            font_name = self.fontlabel2fontname(int(font_label))
            font_name = font_name.replace(" ", "_")
            texts = get_texts(element_data, text_id)
            font, _ = get_skia_font(
                self.font2ttf,
                self.fontmgr,
                element_data,
                text_id,
                font_name,
                h_rate,
            )
            text_alpha = get_text_alpha(
                element_data,
                text_id,
                texts,
                font,
                H,
                W,
                w_rate,
            )
            text_alpha_list.append(text_alpha)
        return text_alpha_list

    def get_text_actual_height_list(
        self, element_data: dict, text_ids: List, scaleinfo: Tuple[float, float]
    ) -> List:
        text_actual_height_list = []
        h_rate, _ = scaleinfo
        for text_id in text_ids:
            font_label = element_data["font"][text_id]
            font_name = self.fontlabel2fontname(int(font_label))
            font_name = font_name.replace(" ", "_")
            font, _ = get_skia_font(
                self.font2ttf,
                self.fontmgr,
                element_data,
                text_id,
                font_name,
                h_rate,
            )
            text_actual_height = get_text_actual_height(font)
            text_actual_height_list.append(text_actual_height)
        return text_actual_height_list

    def compute_and_set_text_actual_width(
        self,
        element_data: dict,
        text_ids: List,
        scaleinfo: Tuple,
    ) -> None:
        h_rate, w_rate = scaleinfo
        text_actual_width = {}
        for text_id in text_ids:
            font_label = element_data["font"][text_id]
            font_name = self.fontlabel2fontname(int(font_label))
            font_name = font_name.replace(" ", "_")
            texts = get_texts(element_data, text_id)
            font, _ = get_skia_font(
                self.font2ttf,
                self.fontmgr,
                element_data,
                text_id,
                font_name,
                h_rate,
            )
            _text_actual_width = get_text_actual_width(
                element_data, text_id, texts, font, w_rate
            )
            text_actual_width[text_id] = _text_actual_width
        element_data["text_actual_width"] = text_actual_width

    def compute_and_set_text_center(
        self, element_data: dict, text_ids: List, image_size: Tuple[int, int]
    ) -> None:
        H, W = image_size
        text_center_x = {}
        text_center_y = {}
        for text_id in text_ids:
            left = element_data["left"][text_id] * W
            w = element_data["width"][text_id] * W
            textAlign = element_data["text_align"][text_id]
            right = left + w
            actual_w = element_data["text_actual_width"][text_id]
            if textAlign == 1:
                _text_center_x = (left + right) / 2.0
            elif textAlign == 3:
                _text_center_x = right - actual_w / 2.0
            elif textAlign == 2:
                _text_center_x = left + actual_w / 2.0
            text_height = element_data["height"][text_id] * H
            top = element_data["top"][text_id] * H
            _text_center_y = (top + top + text_height) / 2.0
            text_center_y[text_id] = _text_center_y
            text_center_x[text_id] = _text_center_x
        element_data["text_center_y"] = text_center_y
        element_data["text_center_x"] = text_center_x
