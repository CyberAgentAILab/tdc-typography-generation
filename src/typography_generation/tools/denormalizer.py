from typing import Dict, List, Tuple, Union

import numpy as np
from typography_generation.io.crello_util import CrelloProcessor
from typography_generation.io.data_object import DesignContext
from logzero import logger


class Denormalizer:
    def __init__(self, dataset: CrelloProcessor):
        self.dataset = dataset

    def denormalize(
        self,
        prefix: str,
        text_num: int,
        prediction: np.array,
        design_context: DesignContext,
    ) -> Tuple:
        pred = prediction[prefix]
        gt = design_context.get_data(prefix)
        pred_token = []
        gt_token = []
        pred_denorm = []
        gt_denorm = []
        canvas_img_size_h, canvas_img_size_w = design_context.canvas_context.img_size
        canvas_h_scale, canvas_w_scale = design_context.canvas_context.scale_box
        for t in range(text_num):
            g = gt[t]
            if prefix in self.dataset.tokenizer.prefix_list:
                p_token = pred[t]
                p = self.dataset.tokenizer.detokenize(prefix, p_token)
                g_token = self.dataset.tokenizer.tokenize(prefix, g)
            elif prefix in self.dataset.tokenizer.rawdata_list:
                p = pred[t]
                p_token = getattr(self.dataset, f"raw2token_{prefix}")(p)
                g_token = getattr(self.dataset, f"raw2token_{prefix}")(g)
                if self.dataset.tokenizer.rawdata_out_format[prefix] == "token":
                    p = p_token
                    g = g_token
                else:
                    p = pred[t]
            else:
                p_token = pred[t]
                p = p_token
                g_token = g
            p = self.denormalize_elm(
                prefix,
                p,
                canvas_img_size_h,
                canvas_img_size_w,
                canvas_h_scale,
                canvas_w_scale,
            )
            g = self.denormalize_elm(
                prefix,
                g,
                canvas_img_size_h,
                canvas_img_size_w,
                canvas_h_scale,
                canvas_w_scale,
            )
            pred_token.append([p_token])
            gt_token.append(g_token)
            pred_denorm.append([p])
            gt_denorm.append(g)
        return pred_token, gt_token, pred_denorm, gt_denorm

    def denormalize_gt(
        self,
        prefix: str,
        text_num: int,
        design_context: DesignContext,
    ) -> Tuple:
        gt = design_context.get_data(prefix)
        gt_token = []
        gt_denorm = []
        canvas_img_size_h, canvas_img_size_w = design_context.canvas_context.img_size
        canvas_h_scale, canvas_w_scale = design_context.canvas_context.scale_box
        logger.debug(f"{prefix} {gt}")
        for t in range(text_num):
            g = gt[t]
            if prefix in self.dataset.tokenizer.prefix_list:
                g_token = self.dataset.tokenizer.tokenize(prefix, g)
            elif prefix in self.dataset.tokenizer.rawdata_list:
                g_token = getattr(self.dataset, f"raw2token_{prefix}")(g)
                if self.dataset.tokenizer.rawdata_out_format[prefix] == "token":
                    g = g_token
            else:
                g_token = g
            g = self.denormalize_elm(
                prefix,
                g,
                canvas_img_size_h,
                canvas_img_size_w,
                canvas_h_scale,
                canvas_w_scale,
            )
            gt_token.append(g_token)
            gt_denorm.append(g)
        return gt_token, gt_denorm

    def denormalize_elm(
        self,
        prefix: str,
        val: Union[int, float, Tuple],
        canvas_img_size_h: int,
        canvas_img_size_w: int,
        canvas_h_scale: float,
        canvas_w_scale: float,
    ) -> Union[int, float, Tuple]:
        if hasattr(self.dataset, f"denorm_{prefix}"):
            func = getattr(self.dataset, f"denorm_{prefix}")
            data_info = {
                "val": val,
                "img_height": canvas_img_size_h,
                "img_width": canvas_img_size_w,
                "scale_h": canvas_h_scale,
                "scale_w": canvas_w_scale,
            }
            val = func(**data_info)
        return val

    def convert_attributes(
        self,
        prefix: str,
        pred: List,
        element_data: Dict,
        text_ids: List,
        scale_h: float,
    ) -> Dict:
        converter = getattr(self.dataset, f"convert_{prefix}")
        converted_attributes = []
        for i, text_index in enumerate(text_ids):
            inputs = {
                "val": pred[i][0],
                "element_data": element_data,
                "text_index": text_index,
                "scale_h": scale_h,
            }
            _converted_attributes = converter(**inputs)
            converted_attributes.append(_converted_attributes)
        if len(text_ids) > 0:
            converted_attribute_prefixes = list(_converted_attributes.keys())
            converted_attribute_dict = {}
            for prefix in converted_attribute_prefixes:
                attribute_data = []
                for _converted_attributes in converted_attributes:
                    val = _converted_attributes[prefix]
                    attribute_data.append([val])
                converted_attribute_dict[prefix] = attribute_data
            return converted_attribute_dict
        else:
            return None
