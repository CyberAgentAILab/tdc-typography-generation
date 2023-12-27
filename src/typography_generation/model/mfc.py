from typing import Any, Dict, List

import numpy as np
import torch
from logzero import logger
from torch import Tensor, nn

from typography_generation.config.attribute_config import (
    CanvasContextEmbeddingAttributeConfig,
    TextElementContextEmbeddingAttributeConfig,
    TextElementContextPredictionAttributeConfig)
from typography_generation.io.crello_util import CrelloProcessor
from typography_generation.io.data_object import ModelInput
from typography_generation.model.mlp import MLP


class MFC(MLP):
    def __init__(
        self,
        prefix_list_element: List,
        prefix_list_canvas: List,
        prefix_list_target: List,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
        embedding_config_canvas: CanvasContextEmbeddingAttributeConfig,
        prediction_config_element: TextElementContextPredictionAttributeConfig,
        d_model: int = 256,
        dropout: float = 0.1,
        seq_length: int = 50,
        **kwargs: Any,
    ) -> None:
        logger.info(f"MFC model")
        super().__init__(
            prefix_list_element,
            prefix_list_canvas,
            prefix_list_target,
            embedding_config_element,
            embedding_config_canvas,
            prediction_config_element,
            d_model,
            dropout,
            seq_length,
        )
        for prefix in prefix_list_target:
            target_prediction_config = getattr(prediction_config_element, prefix)
            setattr(self, f"{prefix}_loss_type", target_prediction_config.loss_type)

    def get_label(
        self,
        out: Tensor,
        text_index: int,
        batch_index: int = 0,
    ) -> Tensor:
        sorted_label = torch.sort(input=out[batch_index], dim=1, descending=True)[1]
        target_label = 0  # top1
        out_label = sorted_label[text_index][target_label].item()
        return out_label

    def get_out(
        self,
        out: Tensor,
        text_index: int,
        batch_index: int = 0,
    ) -> Tensor:
        _out = out[batch_index, text_index].data.cpu().numpy()
        return _out

    def get_outs(
        self,
        model_outs: Dict,
        target_prefix_list: List,
        text_index: int,
        batch_index: int = 0,
    ) -> Dict:
        outs = {}
        for prefix in target_prefix_list:
            out = model_outs[f"{prefix}"]
            loss_type = getattr(self, f"{prefix}_loss_type")
            if loss_type == "cre":
                _out = self.get_label(out, text_index, batch_index)
            else:
                _out = self.get_out(out, text_index, batch_index)
            outs[f"{prefix}"] = _out
        return outs

    def store(
        self,
        out_all: Dict,
        out_labels: Dict,
        target_prefix_list: List,
        text_index: int,
    ) -> Dict:
        for prefix in target_prefix_list:
            out_all[f"{prefix}"][text_index, 0] = out_labels[f"{prefix}"]
        return out_all

    def prediction(
        self,
        model_inputs: ModelInput,
        dataset: CrelloProcessor,
        target_prefix_list: List,
        start_index: int = 0,
    ) -> Tensor:
        target_text_num = int(model_inputs.canvas_text_num[0].item())
        start_index = min(start_index, target_text_num)
        for t in range(start_index, target_text_num):
            model_inputs.zeroinitialize_th_style_attributes(target_prefix_list, t)

        out_all = {}
        for prefix in target_prefix_list:
            loss_type = getattr(self, f"{prefix}_loss_type")
            if loss_type == "cre":
                out_all[prefix] = np.zeros((target_text_num, 1))
            else:
                out_all[prefix] = []
            for t in range(0, start_index):
                tar = getattr(model_inputs, f"{prefix}")[0, t].item()
                if loss_type == "cre":
                    out_all[prefix][t, 0] = tar
                else:
                    out_all[prefix].append(tar)
        for t in range(start_index, target_text_num):
            _, _, feat_cat = self.emb(model_inputs)
            model_outs = self.head(feat_cat)
            out = self.get_outs(model_outs, target_prefix_list, t)
            for prefix in target_prefix_list:
                loss_type = getattr(self, f"{prefix}_loss_type")
                if loss_type == "cre":
                    out_all[f"{prefix}"][t, 0] = out[f"{prefix}"]
                else:
                    out_all[f"{prefix}"].append(out[f"{prefix}"])
        return out_all

    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
