import pickle
import random
from typing import Any, Dict, List, Union

import numpy as np
from logzero import logger
from torch import Tensor, nn

from typography_generation.config.attribute_config import (
    CanvasContextEmbeddingAttributeConfig,
    TextElementContextEmbeddingAttributeConfig,
    TextElementContextPredictionAttributeConfig,
)
from typography_generation.io.data_object import ModelInput
from typography_generation.model.bottleneck import ImlevelLF
from typography_generation.model.decoder import Decoder, MultiTask
from typography_generation.model.embedding import Embedding
from typography_generation.model.encoder import Encoder


class Baseline(nn.Module):
    def __init__(
        self,
        prefix_list_element: List,
        prefix_list_canvas: List,
        prefix_list_target: List,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
        embedding_config_canvas: CanvasContextEmbeddingAttributeConfig,
        prediction_config_element: TextElementContextPredictionAttributeConfig,
        d_model: int = 256,
        n_head: int = 8,
        dropout: float = 0.1,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        seq_length: int = 50,
        std_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        logger.info(f"CanvasVAE settings")
        logger.info(f"d_model: {d_model}")
        logger.info(f"n_head: {n_head}")
        logger.info(f"num_encoder_layers: {num_encoder_layers}")
        logger.info(f"num_decoder_layers: {num_decoder_layers}")
        logger.info(f"seq_length: {seq_length}")

        self.emb = Embedding(
            prefix_list_element,
            prefix_list_canvas,
            embedding_config_element,
            embedding_config_canvas,
            d_model,
            dropout,
            seq_length,
        )
        self.enc = Encoder(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            num_encoder_layers=num_encoder_layers,
        )
        self.lf = ImlevelLF(vae=True, std_ratio=std_ratio)

        self.dec = Decoder(
            prefix_list_target,
            embedding_config_element,
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            seq_length=seq_length,
            autoregressive_scheme=False,
        )
        self.head = MultiTask(
            prefix_list_target, prediction_config_element, d_model, bypass=False
        )
        self.initialize_weights()

    def forward(self, model_inputs: ModelInput) -> Tensor:
        (
            src,
            text_mask_src,
            feat_cat,
        ) = self.emb(model_inputs)
        z = self.enc(src, text_mask_src)
        z, vae_data = self.lf(z, text_mask_src)
        zd = self.dec(feat_cat, z, model_inputs)
        outs = self.head(zd, feat_cat)
        outs["vae_data"] = vae_data
        return outs

    def store(
        self,
        out_labels_all: Dict,
        out_labels: Dict,
        target_prefix_list: List,
        text_index: int,
    ) -> Dict:
        for prefix in target_prefix_list:
            out_labels_all[f"{prefix}"][text_index, 0] = out_labels[f"{prefix}"]
        return out_labels_all

    def prediction(
        self,
        model_inputs: ModelInput,
        target_prefix_list: List,
        start_index: int = 0,
    ) -> Tensor:
        target_text_num = int(model_inputs.canvas_text_num[0].item())
        start_index = min(start_index, target_text_num)
        for t in range(start_index, target_text_num):
            model_inputs.zeroinitialize_th_style_attributes(target_prefix_list, t)
        (
            src,
            text_mask_src,
            feat_cat,
        ) = self.emb(model_inputs)
        z = self.enc(src, text_mask_src)
        z = self.lf.prediction(z, text_mask_src)
        zd = self.dec(feat_cat, z, model_inputs)
        model_outs = self.head(zd, feat_cat)

        out_labels_all = {}
        for prefix in target_prefix_list:
            out_labels_all[prefix] = np.zeros((target_text_num, 1))
            for t in range(0, start_index):
                tar = getattr(model_inputs, f"{prefix}")[0, t].item()
                out_labels_all[prefix][t, 0] = tar

        for t in range(start_index, target_text_num):
            out_labels = self.get_labels(model_outs, target_prefix_list, t)
            out_labels_all = self.store(
                out_labels_all, out_labels, target_prefix_list, t
            )
        return out_labels_all

    def sample(
        self,
        model_inputs: ModelInput,
        target_prefix_list: List,
        start_index: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        target_text_num = int(model_inputs.canvas_text_num[0].item())
        start_index = min(start_index, target_text_num)
        for t in range(start_index, target_text_num):
            model_inputs.zeroinitialize_th_style_attributes(target_prefix_list, t)
        (
            src,
            text_mask_src,
            feat_cat,
        ) = self.emb(model_inputs)
        z = self.enc(src, text_mask_src)
        z = self.lf.sample(z, text_mask_src)
        zd = self.dec(feat_cat, z, model_inputs)
        model_outs = self.head(zd, feat_cat)
        out_labels_all = {}
        for prefix in target_prefix_list:
            out_labels_all[prefix] = np.zeros((target_text_num, 1))
            for t in range(0, start_index):
                tar = getattr(model_inputs, f"{prefix}")[0, t].item()
                out_labels_all[prefix][t, 0] = tar

        for t in range(start_index, target_text_num):
            out_labels = self.get_labels(model_outs, target_prefix_list, t)
            out_labels_all = self.store(
                out_labels_all, out_labels, target_prefix_list, t
            )
        return out_labels_all

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


class AllZero(Baseline):
    def __init__(
        self,
        prefix_list_element: List,
        prefix_list_canvas: List,
        prefix_list_target: List,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
        embedding_config_canvas: CanvasContextEmbeddingAttributeConfig,
        prediction_config_element: TextElementContextPredictionAttributeConfig,
        d_model: int = 256,
        n_head: int = 8,
        dropout: float = 0.1,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        seq_length: int = 50,
        std_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            prefix_list_element,
            prefix_list_canvas,
            prefix_list_target,
            embedding_config_element,
            embedding_config_canvas,
            prediction_config_element,
            d_model,
            n_head,
            dropout,
            num_encoder_layers,
            num_decoder_layers,
            seq_length,
            std_ratio,
        )

    def get_labels(
        self,
        model_outs: Union[Dict, None],
        target_prefix_list: List,
        text_index: int,
        batch_index: int = 0,
    ) -> Dict:
        out_labels = {}
        for prefix in target_prefix_list:
            out_labels[f"{prefix}"] = 0

        return out_labels


class AllRandom(Baseline):
    def __init__(
        self,
        prefix_list_element: List,
        prefix_list_canvas: List,
        prefix_list_target: List,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
        embedding_config_canvas: CanvasContextEmbeddingAttributeConfig,
        prediction_config_element: TextElementContextPredictionAttributeConfig,
        d_model: int = 256,
        n_head: int = 8,
        dropout: float = 0.1,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        seq_length: int = 50,
        std_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            prefix_list_element,
            prefix_list_canvas,
            prefix_list_target,
            embedding_config_element,
            embedding_config_canvas,
            prediction_config_element,
            d_model,
            n_head,
            dropout,
            num_encoder_layers,
            num_decoder_layers,
            seq_length,
            std_ratio,
        )

    def get_labels(
        self,
        model_outs: Dict,
        target_prefix_list: List,
        text_index: int,
        batch_index: int = 0,
    ) -> Dict:
        out_labels = {}
        for prefix in target_prefix_list:
            out = model_outs[f"{prefix}"]
            label_range = len(out[batch_index][text_index])
            out_labels[f"{prefix}"] = random.randint(0, label_range - 1)

        return out_labels


class Mode(Baseline):
    def __init__(
        self,
        prefix_list_element: List,
        prefix_list_canvas: List,
        prefix_list_target: List,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
        embedding_config_canvas: CanvasContextEmbeddingAttributeConfig,
        prediction_config_element: TextElementContextPredictionAttributeConfig,
        d_model: int = 256,
        n_head: int = 8,
        dropout: float = 0.1,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        seq_length: int = 50,
        std_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            prefix_list_element,
            prefix_list_canvas,
            prefix_list_target,
            embedding_config_element,
            embedding_config_canvas,
            prediction_config_element,
            d_model,
            n_head,
            dropout,
            num_encoder_layers,
            num_decoder_layers,
            seq_length,
            std_ratio,
        )
        self.prefix2mode = pickle.load(
            open(
                f"prefix2mode.pkl",
                "rb",
            )
        )

    def get_labels(
        self,
        model_outs: Dict,
        target_prefix_list: List,
        text_index: int,
        batch_index: int = 0,
    ) -> Dict:
        out_labels = {}
        for prefix in target_prefix_list:
            out = model_outs[f"{prefix}"]
            out_labels[f"{prefix}"] = self.prefix2mode[prefix]

        return out_labels
