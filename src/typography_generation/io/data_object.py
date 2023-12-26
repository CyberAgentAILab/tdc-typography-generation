from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from logzero import logger
from torch import Tensor

from typography_generation.config.attribute_config import (
    TextElementContextEmbeddingAttributeConfig,
)


@dataclass
class PrefixListObject:
    textelement: List
    canvas: List
    model_input: List
    target: List


@dataclass
class BinsData:
    color_bin: int
    small_spatio_bin: int
    large_spatio_bin: int


@dataclass
class FontConfig:
    font_num: int
    font_emb_type: str = "label"
    font_emb_weight: float = 1.0
    font_emb_dim: int = 40
    font_emb_name: str = "mfc"


@dataclass
class DataPreprocessConfig:
    order_type: str
    seq_length: int


@dataclass
class SamplingConfig:
    sampling_param: str
    sampling_param_geometry: float
    sampling_param_semantic: float
    sampling_num: int


class ModelInput:
    def __init__(self, design_context_list: List, model_input: List, gpu: bool) -> None:
        self.design_context_list = design_context_list
        self.model_input = model_input
        self.prefix_list = self.design_context_list[0].model_input_prefix_list
        self.gpu = gpu
        self.reset()

    def reset(self) -> None:
        self.setgt()
        if self.gpu is True:
            self.cuda()

    def setgt(self) -> None:
        if len(self.prefix_list) != len(self.model_input):
            raise ValueError("The length between list and input is different.")
        self.batch_num = self.model_input[0].shape[0]
        for prefix, elm in zip(self.prefix_list, self.model_input):
            setattr(self, f"{prefix}", elm.clone())
            if prefix == "canvas_text_num":
                self.canvas_text_num = elm.clone()

    def cuda(self) -> None:
        for prefix in self.prefix_list:
            tar = getattr(self, f"{prefix}")
            if type(tar) == Tensor:
                setattr(self, f"{prefix}", tar.cuda())

    def target_register(self, prefix: str, elm: Any) -> None:
        setattr(self, f"{prefix}", elm)

    def zeroinitialize_style_attributes(self, prefix_list: List) -> None:
        for prefix in prefix_list:
            if prefix == "canvas_text_num":
                continue
            tar = getattr(self, f"{prefix}")
            tar_rep = torch.zeros_like(tar)
            setattr(self, f"{prefix}", tar_rep)

    def zeroinitialize_specific_attribute(self, prefix: str) -> None:
        tar = getattr(self, f"{prefix}")
        if prefix != "canvas_text_num":
            setattr(self, f"{prefix}", torch.zeros_like(tar))

    def setgt_specific_attribute(self, prefix_tar: str) -> None:
        for prefix, elm in zip(self.prefix_list, self.model_input):
            if prefix == prefix_tar:
                tar = elm.clone()
                if self.gpu is True and type(tar) == Tensor:
                    tar = tar.cuda()
                setattr(self, f"{prefix}", tar)

    def update_th_style_attributes(
        self,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
        prefix_list: List,
        model_output: Tensor,
        text_index: int,
        batch_index: int = 0,
    ) -> None:
        for prefix in prefix_list:
            target_embedding_config = getattr(embedding_config_element, prefix)
            tar = getattr(self, f"{prefix}")
            out = model_output[f"{prefix}"]
            tar[batch_index, text_index] = out
            setattr(self, f"{target_embedding_config.input_prefix}", tar)

    def zeroinitialize_th_style_attributes(
        self,
        prefix_list: List,
        text_index: int,
        batch_index: int = 0,
    ) -> None:
        for prefix in prefix_list:
            tar = getattr(self, f"{prefix}")
            tar[batch_index, text_index] = 0
            setattr(self, f"{prefix}", tar)

    def additional_input_from_design_context_list(
        self, design_context_list: List
    ) -> None:
        self.texts = []
        for design_context in design_context_list:
            self.texts.append(design_context.text_context.texts)


@dataclass
class ElementContext:
    prefix_list: List
    rawdata2token: Dict
    img_size: Tuple
    scaleinfo: Tuple
    seq_length: int

    def __post_init__(self) -> None:
        for prefix in self.prefix_list:
            setattr(self, prefix, [])
            if prefix == "text_local_img":
                self.text_local_img_model_input = np.zeros(
                    (self.seq_length, 3, 224, 224), dtype=np.float32
                )
            elif prefix == "text_local_img_emb":
                self.text_local_img_emb_model_input = np.zeros(
                    (self.seq_length, 512), dtype=np.float32
                )
            elif prefix == "text_emb":
                self.text_emb_model_input = np.zeros(
                    (self.seq_length, 512), dtype=np.float32
                )
            elif prefix == "text_font_emb":
                self.text_font_emb_model_input = (
                    np.zeros((self.seq_length, 40), dtype=np.float32) - 10000
                )
            else:
                setattr(
                    self,
                    f"{prefix}_model_input",
                    np.zeros((self.seq_length), dtype=np.float32) - 1,
                )

            if prefix in self.rawdata2token.keys():
                setattr(
                    self,
                    f"{self.rawdata2token[prefix]}_model_input",
                    np.zeros((self.seq_length), dtype=np.float32) - 1,
                )


@dataclass
class CanvasContext:
    canvas_bg_img: np.array
    canvas_text_num: int
    img_size: Tuple
    scale_box: Tuple
    prefix_list: List


@dataclass
class DesignContext:
    element_context: ElementContext
    canvas_context: CanvasContext

    def __post_init__(self) -> None:
        self.prepare_keys()

    def prepare_keys(self) -> None:
        self.canvas_context_keys = dir(self.canvas_context)
        self.element_context_keys = dir(self.element_context)

    def get_text_num(self) -> int:
        return self.canvas_context.canvas_text_num

    def convert_target_to_torch_format(
        self,
        tar: Any,
    ) -> Any:
        if type(tar) == np.ndarray:
            tar = torch.from_numpy(tar)
        elif type(tar) == float or type(tar) == int:
            tar = torch.Tensor([tar])
        elif type(tar) == str or type(tar) == list:
            pass
        else:
            logger.info(tar)
            raise NotImplementedError()
        return tar

    def search_class(self, prefix: str) -> Union[ElementContext, CanvasContext]:
        if prefix in self.canvas_context_keys:
            return self.canvas_context
        elif prefix in self.element_context_keys:
            return self.element_context
        else:
            logger.info(
                f"{prefix}, {self.canvas_context_keys}, {self.element_context_keys}"
            )
            raise NotImplementedError()

    def get_model_inputs_from_prefix_list(self, prefix_list: List) -> List:
        self.model_input_prefix_list = prefix_list
        model_inputs = []
        for prefix in prefix_list:
            logger.debug(f"convert_target_to_torch_format {prefix}")
            tar_cls = self.search_class(f"{prefix}")
            tar = getattr(tar_cls, f"{prefix}_model_input")
            tar = self.convert_target_to_torch_format(tar)
            model_inputs.append(tar)
        return model_inputs

    def get_data(self, prefix: str) -> Any:
        tar_cls = self.search_class(f"{prefix}")
        tar = getattr(tar_cls, f"{prefix}")
        return tar

    def get_canvas_size(self) -> Tuple:
        canvas_size = (
            self.canvas_context.canvas_img_size_h,
            self.canvas_context.canvas_img_size_w,
        )
        return canvas_size

    def get_text_context(self) -> ElementContext:
        return self.element_context

    def get_bg(self) -> np.array:
        return self.canvas_context.canvas_bg_img

    def get_scaleinfo(self) -> Tuple:
        return (self.canvas_context.canvas_h_scale, self.canvas_context.canvas_w_scale)

    def convert_torch_format(self, prefix: str) -> None:
        tar_cls = self.search_class(prefix)
        tar = getattr(tar_cls, prefix)
        tar = self.convert_target_to_torch_format(tar)
        setattr(tar_cls, prefix, tar)
