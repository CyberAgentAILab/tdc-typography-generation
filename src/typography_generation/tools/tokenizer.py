import pickle
from typing import Dict, Tuple, Union
from einops import repeat
import numpy as np
from torch import Tensor
import torch

default_cluster_num_dict = {
    "text_font_size": 16,
    "text_font_color": 64,
    "text_height": 16,
    "text_width": 16,
    "text_top": 64,
    "text_left": 64,
    "text_center_y": 64,
    "text_center_x": 64,
    "text_distance_y_from_prev": 16,
    "text_distance_x_from_prev": 16,
    "text_angle": 16,
    "text_letter_spacing": 16,
    "text_line_height_scale": 16,
    "text_line_height_size": 16,
    "canvas_aspect_ratio": 16,
}


class Tokenizer:
    def __init__(
        self,
        data_dir: str,
        cluster_num_dict: Union[Dict, None] = None,
        load_cluster: bool = True,
    ) -> None:
        self.prefix_list = [
            "text_font_size",
            "text_font_color",
            "text_height",
            "text_width",
            "text_top",
            "text_left",
            "text_center_y",
            "text_center_x",
            "text_angle",
            "text_letter_spacing",
            "text_line_height_scale",
            "text_line_height_size",
            "text_distance_y_from_prev",
            "text_distance_x_from_prev",
            "canvas_aspect_ratio",
        ]
        self.rawdata2token = {
            "text_font_emb": "text_font",
            "text_font_size_raw": "text_font_size",
        }
        self.rawdata_list = list(self.rawdata2token.keys())
        self.rawdata_out_format = {
            "text_font_emb": "token",
            "text_font_size_raw": "raw",
        }

        self.prediction_token_list = [
            "text_font_emb",
        ]
        if cluster_num_dict is None:
            cluster_num_dict = default_cluster_num_dict
        if load_cluster is True:
            for prefix in self.prefix_list:
                # fn = f"{data_dir}/stats/{prefix}_cluster.pkl"
                cluster_num = cluster_num_dict[prefix]
                fn = f"{data_dir}/cluster/{prefix}_{cluster_num}.pkl"
                if prefix == "text_font_color":
                    cluster = np.array(pickle.load(open(fn, "rb")))
                else:
                    cluster = np.array(pickle.load(open(fn, "rb"))).flatten()
                setattr(self, f"{prefix}_cluster", cluster)

    def assign_label(self, val: Union[float, int], bins: np.array) -> int:
        label = int(np.argsort(np.square(bins - val))[0])
        return label

    def assign_color_label(self, val: Union[float, int], bins: np.array) -> int:
        val = np.tile(np.array(val)[np.newaxis, :], (len(bins), 1))
        d = np.square(bins - val).sum(1)
        label = int(np.argsort(d, axis=0)[0])
        return label

    def tokenize(self, prefix: str, val: Union[float, int]) -> int:
        if prefix == "text_font_color":
            label = self.assign_color_label(val, getattr(self, f"{prefix}_cluster"))
        else:
            label = self.assign_label(val, getattr(self, f"{prefix}_cluster"))
        return label

    def detokenize(self, prefix: str, label: Union[int, float]) -> Union[Tuple, float]:
        if prefix == "text_font_color":
            b, g, r = getattr(self, f"{prefix}_cluster")[int(label)]
            return (r, g, b)
        elif prefix == "text_font_size_raw":
            val = float(getattr(self, f"text_font_size_cluster")[int(label)])
        else:
            val = float(getattr(self, f"{prefix}_cluster")[int(label)])
            return val
