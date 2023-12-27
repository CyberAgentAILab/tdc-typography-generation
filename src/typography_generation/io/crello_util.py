import math
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import PIL
import skia
import torch
from einops import repeat
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from typography_generation.io.data_object import FontConfig
from typography_generation.tools.tokenizer import Tokenizer
from typography_generation.visualization.renderer_util import (
    get_skia_font,
    get_text_actual_width,
    get_texts,
)

fontmgr = skia.FontMgr()


class CrelloProcessor:
    def __init__(
        self,
        data_dir: str,
        tokenizer: Tokenizer,
        dataset: Any,
        font_config: FontConfig,
        use_extended_dataset: bool = True,
        seq_length: int = 50,
    ) -> None:
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.font_config = font_config
        self.dataset = dataset
        self.seq_length = seq_length
        if font_config is not None:
            fn = os.path.join(
                self.data_dir, "font_emb", f"{font_config.font_emb_name}.pkl"
            )
            self.fontid2fontemb = pickle.load(open(fn, "rb"))
        self.use_extended_dataset = use_extended_dataset
        if not use_extended_dataset:
            fn = os.path.join(data_dir, "font2ttf.pkl")
            _font2ttf = pickle.load(open(fn, "rb"))
            font2ttf = {}
            for key in _font2ttf.keys():
                tmp = _font2ttf[key].split("/data/dataset/crello/")[1]
                fn = os.path.join(data_dir, tmp)
                font2ttf[key] = fn
            self.font2ttf = font2ttf

            fn = os.path.join(data_dir, "svgid2scaleinfo.pkl")
            self.svgid2scaleinfo = pickle.load(open(fn, "rb"))
            self.fontlabel2fontname = self.dataset.features["font"].feature.int2str

            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.text_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device("cpu")
            self.model.to(self.device)

    def get_canvas_text_num(
        self, element_data: dict, **kwargs: Any
    ) -> Tuple[int, List]:
        text_num = 0
        for k in range(len(element_data["text"])):
            if element_data["text"][k] == "":
                pass
            else:
                text_num += 1
        text_num = min(text_num, self.seq_length)
        return text_num

    def get_canvas_text_ids(
        self, element_data: dict, **kwargs: Any
    ) -> Tuple[int, List]:
        text_ids = []
        for k in range(len(element_data["text"])):
            if element_data["text"][k] == "":
                pass
            else:
                text_ids.append(k)
        return text_ids

    def get_canvas_bg_size(self, element_data: dict) -> Tuple[int, int]:
        return element_data["canvas_bg_size"]

    def get_scale_box(self, element_data: dict) -> List:
        if self.use_extended_dataset:
            return tuple(element_data["scale_box"])
        else:
            svgid = element_data["id"]
            return self.svgid2scaleinfo[svgid]

    def get_text_font(self, element_data: dict, text_index: int, **kwargs: Any) -> int:
        font = element_data["font"][text_index] - 1
        return int(font)

    def denorm_text_font(self, val: int, **kwargs: Any) -> int:
        val += 1
        return val

    def get_text_font_emb(
        self,
        element_data: dict,
        text_index: int,
        **kwargs: Any,
    ) -> np.array:
        font = element_data["font"][text_index]
        font_emb = self.fontid2fontemb[font - 1]
        return font_emb

    def raw2token_text_font_emb(
        self,
        val: np.array,
        **kwargs: Any,
    ) -> int:
        vec = repeat(val, "c -> n c", n=len(self.fontid2fontemb))
        diff = (vec - self.fontid2fontemb) ** 2
        diff = diff.sum(1)
        font = int(np.argsort(diff)[0])
        return font

    def denorm_text_font_emb(self, val: int, **kwargs: Any) -> int:
        val += 1
        return val

    def get_skia_font(
        self, element_data: dict, text_index: int, scaleinfo: Tuple, **kwargs: Any
    ) -> int:
        font_label = element_data["font"][text_index]
        font_name = self.fontlabel2fontname(int(font_label))
        font_name = font_name.replace(" ", "_")
        scale_h, _ = scaleinfo
        font_skia, _ = get_skia_font(
            self.font2ttf,
            fontmgr,
            element_data,
            text_index,
            font_name,
            scale_h,
        )
        return font_skia

    def get_text_font_size(
        self,
        element_data: dict,
        text_index: int,
        img_size: Tuple,
        scaleinfo: Tuple,
        **kwargs: Any,
    ) -> float:
        fs = element_data["font_size"][text_index]
        scale_h, _ = scaleinfo
        h, _ = img_size
        val = fs * scale_h / h
        return float(val)

    def denorm_text_font_size(
        self, val: float, img_height: int, scale_h: float, **kwargs: Any
    ) -> float:
        val = val * img_height / scale_h
        return val

    def get_text_font_size_raw(
        self,
        element_data: dict,
        text_index: int,
        img_size: Tuple,
        scaleinfo: Tuple,
        **kwargs: Any,
    ) -> float:
        val = self.get_text_font_size(element_data, text_index, img_size, scaleinfo)
        return val

    def raw2token_text_font_size_raw(
        self,
        val: float,
        **kwargs: Any,
    ) -> int:
        val = self.tokenizer.tokenize("text_font_size", float(val))
        return val

    def denorm_text_font_size_raw(
        self, val: float, img_height: int, scale_h: float, **kwargs: Any
    ) -> float:
        val = val * img_height / scale_h
        return val

    def get_text_font_color(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> Tuple[int, int, int]:
        B, G, R = element_data["color"][text_index]
        return (R, G, B)

    def get_text_height(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> float:
        height = element_data["height"][text_index]
        return float(height)

    def denorm_text_height(self, val: float, img_height: int, **kwargs: Any) -> float:
        val = val * img_height
        return val

    def get_text_width(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> float:
        width = element_data["width"][text_index]
        return float(width)

    def denorm_text_width(self, val: float, img_width: int, **kwargs: Any) -> float:
        val = val * img_width
        return val

    def get_text_top(self, element_data: dict, text_index: int, **kwargs: Any) -> float:
        top = element_data["top"][text_index]
        return float(top)

    def denorm_text_top(self, val: float, img_height: int, **kwargs: Any) -> float:
        val = val * img_height
        return val

    def get_text_left(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> float:
        left = element_data["left"][text_index]
        return float(left)

    def denorm_text_left(self, val: float, img_width: int, **kwargs: Any) -> float:
        val = val * img_width
        return val

    def get_text_actual_width(
        self,
        element_data: dict,
        text_index: int,
        texts: List[str],
        font_skia: skia.Font,
        scaleinfo: Tuple[float, float],
        **kwargs: Any,
    ) -> float:
        _, scale_w = scaleinfo
        text_width = get_text_actual_width(
            element_data, text_index, texts, font_skia, scale_w
        )
        return text_width

    def get_text_center_y(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> float:
        if self.use_extended_dataset:
            return element_data["text_center_y"][text_index]
        else:
            text_height = element_data["height"][text_index]
            top = element_data["top"][text_index]
            center_y = (top + top + text_height) / 2.0
            return float(center_y)

    def get_text_center_x(
        self,
        element_data: dict,
        text_index: int,
        scaleinfo: Tuple,
        **kwargs: Any,
    ) -> float:
        if self.use_extended_dataset:
            return element_data["text_center_x"][text_index]
        else:
            left = element_data["left"][text_index]
            w = element_data["width"][text_index]
            textAlign = element_data["text_align"][text_index]
            texts = get_texts(element_data, text_index)
            font_skia = self.get_skia_font(element_data, text_index, scaleinfo)
            text_actual_width = self.get_text_actual_width(
                element_data, text_index, texts, font_skia, scaleinfo
            )
            right = left + w
            if textAlign == 1:
                center_x = (left + right) / 2.0
            elif textAlign == 3:
                center_x = right - text_actual_width / 2.0
            elif textAlign == 2:
                center_x = left + text_actual_width / 2.0
            return float(center_x)

    def get_text_align_type(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> int:
        align_type = element_data["text_align"][text_index] - 1
        return int(align_type)

    def denorm_text_align_type(self, val: int, **kwargs: Any) -> int:
        val += 1
        return val

    def get_text_capitalize(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> int:
        capitalize = element_data["capitalize"][text_index]
        return int(capitalize)

    def get_text_angle(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> float:
        angle = element_data["angle"][text_index]
        angle = (float(angle) * 180 / math.pi) / 360.0
        angle = math.modf(angle)[0]
        return float(angle)

    def denorm_text_angle(self, val: float, **kwargs: Any) -> float:
        val = val * 360 / 180.0 * math.pi
        return val

    def get_text_letter_spacing(
        self,
        element_data: dict,
        text_index: int,
        scaleinfo: Tuple,
        img_size: Tuple,
        **kwargs: Any,
    ) -> float:
        _, scale_w = scaleinfo
        _, W = img_size
        letter_space = element_data["letter_spacing"][text_index] * scale_w / W
        return float(letter_space)

    def denorm_text_letter_spacing(
        self, val: float, img_width: int, scale_w: float, **kwargs: Any
    ) -> float:
        val = val * img_width / scale_w
        return val

    def get_text_line_height_scale(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> float:
        line_height_scale = element_data["line_height"][text_index]
        return float(line_height_scale)

    def get_text_char_count(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> int:
        text = self.get_text(element_data, text_index)
        texts = text.split(os.linesep)
        max_char_count = 0
        for t in texts:
            max_char_count = max(max_char_count, len(t))
        return min(max_char_count, 50 - 1)

    def get_text_line_count(
        self, element_data: dict, text_index: int, **kwargs: Any
    ) -> int:
        texts = element_data["text"][text_index].split(os.linesep)
        line_count = 0
        for t in texts:
            if t == "":
                pass
            else:
                line_count += 1
        return min(line_count, 49)

    def get_text(self, element_data: dict, text_index: int) -> str:
        text = element_data["text"][text_index]
        return str(text)

    def get_canvas_aspect_ratio(
        self, element_data: dict, bg_img: Any, **kwargs: Any
    ) -> float:
        h, w = bg_img.size[1], bg_img.size[0]
        ratio = float(h) / float(w)
        return ratio

    def get_canvas_group(self, element_data: dict, **kwargs: Any) -> int:
        return element_data["group"]

    def get_canvas_format(self, element_data: dict, **kwargs: Any) -> int:
        return element_data["format"]

    def get_canvas_category(self, element_data: dict, **kwargs: Any) -> int:
        return element_data["category"]

    def get_canvas_width(self, element_data: dict, **kwargs: Any) -> int:
        return element_data["canvas_width"]

    def get_canvas_height(self, element_data: dict, **kwargs: Any) -> int:
        return element_data["canvas_height"]

    def get_canvas_bg_img_emb(
        self, element_data: dict, bg_img: PIL.Image, **kwargs: Any
    ) -> np.array:
        if self.use_extended_dataset:
            return np.array(element_data["canvas_bg_img_emb"])
        else:
            inputs = self.processor(images=[bg_img], return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
            image_feature = self.model.get_image_features(**inputs)
            return image_feature.data.numpy()

    def get_text_emb(
        self,
        element_data: dict,
        text_index: int,
        **kwargs: Any,
    ) -> np.array:
        if self.use_extended_dataset:
            return np.array(element_data["text_emb"][text_index])
        else:
            text = element_data["text"][text_index]
            inputs = self.text_tokenizer([text], padding=True, return_tensors="pt")
            if inputs["input_ids"].shape[1] > 77:
                inp = inputs["input_ids"][:, :77]
            else:
                inp = inputs["input_ids"]
            text_features = self.model.get_text_features(inp).data.numpy()[0]
            return text_features

    def get_text_local_img(
        self,
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

    def get_text_local_img_emb(
        self,
        element_data: dict,
        text_index: int,
        scaleinfo: Tuple,
        img_size: Tuple,
        bg_img: Any,
        **kwargs: Any,
    ) -> np.array:
        if self.use_extended_dataset:
            return np.array(element_data["text_local_img_emb"][text_index])
        else:
            H, W = img_size
            text_center_x = self.get_text_center_x(element_data, text_index, scaleinfo)
            text_center_y = self.get_text_center_y(element_data, text_index)
            local_img = self.get_text_local_img(
                bg_img.copy(), text_center_y, text_center_x, H, W
            )
            local_img = Image.fromarray(local_img.astype(np.uint8)).resize((224, 224))
            inputs = self.processor(images=[local_img], return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
            image_feature = self.model.get_image_features(**inputs)
            return image_feature.data.numpy()

    def load_samples(self, index: int) -> Tuple[Dict, Any, str, int]:
        element_data = self.dataset[index]
        svg_id = element_data["id"]
        fn = os.path.join(self.data_dir, "generate_bg_png", f"{svg_id}.png")
        bg = Image.open(fn).convert("RGB")  # background image
        return element_data, bg, svg_id, index

    def __len__(self) -> int:
        return len(self.dataset)
