import random
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from logzero import logger
from torch import Tensor, nn
from torch.functional import F
from typography_generation.config.attribute_config import (
    CanvasContextEmbeddingAttributeConfig,
    TextElementContextEmbeddingAttributeConfig,
    TextElementContextPredictionAttributeConfig,
)
from typography_generation.io.crello_util import CrelloProcessor
from typography_generation.io.data_object import ModelInput
from typography_generation.model.decoder import Decoder, MultiTask
from typography_generation.model.embedding import Embedding
from typography_generation.model.encoder import Encoder

FILTER_VALUE = -float("Inf")


def top_p(
    prior: Tensor,
    text_index: int,
    sampling_param: float,
) -> int:
    prior = F.softmax(prior, 1)
    sorted_prob, sorted_label = torch.sort(input=prior, dim=1, descending=True)
    prior = sorted_prob[text_index]
    sum_p = 0
    for k in range(len(prior)):
        sum_p += prior[k].item()
        if sum_p > sampling_param:  # prior
            break

    range_class = k
    if range_class == 0:
        index = 0
    else:
        index = random.randint(0, range_class)
    out_label = sorted_label[text_index][index].item()
    return out_label


def top_p_weight(
    logits: Tensor,
    text_index: int,
    sampling_param: float,
) -> int:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=1), dim=1)
    S = logits.size(1)
    indices = torch.arange(S).view(1, S).to(logits.device)
    # make sure to keep the first logit (most likely one)
    sorted_logits[(cumulative_probs > sampling_param) & (indices > 0)] = FILTER_VALUE
    logits = sorted_logits.gather(dim=1, index=sorted_indices.argsort(dim=1))
    probs = F.softmax(logits, dim=1)
    output = torch.multinomial(probs, num_samples=1)  # (B, 1)
    logger.debug(f"{output.shape}=")
    return output[text_index][0].item()


sampler_dict = {"top_p": top_p, "top_p_weight": top_p_weight}


def sample_label(
    logits: Tensor, text_index: int, sampling_param: float, mode: str = "top_p"
) -> int:
    return sampler_dict[mode](logits, text_index, sampling_param)


def get_structure(pred: np.array, indexes: List) -> Dict:
    index2samestructureindexes: Dict[str, List]
    index2samestructureindexes = {}
    for i in indexes:
        index2samestructureindexes[i] = []
        label_i = pred[i]
        for j in indexes:
            label_j = pred[j]
            if label_i == label_j:
                index2samestructureindexes[i].append(j)
    return index2samestructureindexes


def get_structure_dict(prefix_list: List, preds: Dict, indexes: List) -> Dict:
    index2samestructureindexes = {}
    for prefix in prefix_list:
        index2samestructureindexes[prefix] = get_structure(
            preds[prefix],
            indexes,
        )
    return index2samestructureindexes


def get_init_label_link(indexes: List) -> Dict:
    label_link: Dict[str, Union[int, None]]
    label_link = {}
    for i in indexes:
        label_link[i] = None
    return label_link


def initialize_link(prefix_list: List, indexes: List) -> Tuple[Dict, Dict]:
    label_link: Dict[str, Dict]
    label_link = {}
    used_labels: Dict[str, List]
    used_labels = {}
    for prefix in prefix_list:
        label_link[prefix] = get_init_label_link(indexes)
        used_labels[prefix] = []
    return label_link, used_labels


def label_linkage(
    prefix_list: List,
    text_num: int,
    preds: Dict,
) -> Tuple[Dict, Dict, Dict]:
    indexes = list(range(text_num))
    index2samestructureindexes = get_structure_dict(prefix_list, preds, indexes)
    label_link, used_labels = initialize_link(prefix_list, indexes)
    return index2samestructureindexes, label_link, used_labels


def update_label_link(label_link: Dict, samestructureindexes: List, label: int) -> Dict:
    for i in samestructureindexes:
        label_link[i] = label
    return label_link


class BART(nn.Module):
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
        bypass: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        logger.info(f"BART settings")
        logger.info(f"d_model: {d_model}")
        logger.info(f"n_head: {n_head}")
        logger.info(f"num_encoder_layers: {num_encoder_layers}")
        logger.info(f"num_decoder_layers: {num_decoder_layers}")
        logger.info(f"seq_length: {seq_length}")
        self.embedding_config_element = embedding_config_element

        self.emb = Embedding(
            prefix_list_element,
            prefix_list_canvas,
            embedding_config_element,
            embedding_config_canvas,
            d_model=d_model,
            dropout=dropout,
            seq_length=seq_length,
        )
        self.enc = Encoder(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            num_encoder_layers=num_encoder_layers,
        )
        self.dec = Decoder(
            prefix_list_target,
            embedding_config_element,
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            seq_length=seq_length,
        )
        self.head = MultiTask(
            prefix_list_target,
            prediction_config_element,
            d_model,
            bypass=bypass,
        )
        self.initialize_weights()

        for prefix in prefix_list_target:
            target_prediction_config = getattr(prediction_config_element, prefix)
            setattr(self, f"{prefix}_att_type", target_prediction_config.att_type)

    def forward(self, model_inputs: ModelInput) -> Tensor:
        start = time.time()
        (
            src,
            text_mask_src,
            feat_cat,
        ) = self.emb(model_inputs)
        logger.debug(f"{time.time()-start} sec emb")
        start = time.time()
        z = self.enc(src, text_mask_src)
        logger.debug(f"{time.time()-start} sec enc")
        start = time.time()
        zd = self.dec(feat_cat, z, model_inputs)
        logger.debug(f"{time.time()-start} sec dec")
        start = time.time()
        outs = self.head(zd, feat_cat)
        logger.debug(f"{time.time()-start} sec head")
        return outs

    def tokenize_model_out(
        self,
        dataset: CrelloProcessor,
        prefix: str,
        model_out: Tensor,
        batch_index,
        text_index,
    ) -> int:
        if prefix in dataset.tokenizer.rawdata_list:
            data = model_out[batch_index][text_index].data.cpu().numpy()
            out_label = getattr(dataset, f"raw2token_{prefix}")(data)
        else:
            sorted_label = torch.sort(
                input=model_out[batch_index], dim=1, descending=True
            )[1]
            target_label = 0  # top1
            out_label = sorted_label[text_index][target_label].item()
        return out_label

    def get_labels(
        self,
        model_outs: Dict,
        dataset: CrelloProcessor,
        target_prefix_list: List,
        text_index: int,
        batch_index: int = 0,
    ) -> Dict:
        out_labels = {}
        for prefix in target_prefix_list:
            out = model_outs[f"{prefix}"]
            out_label = self.tokenize_model_out(
                dataset, prefix, out, batch_index, text_index
            )

            out_labels[f"{prefix}"] = out_label

        return out_labels

    def get_outs(
        self,
        model_outs: Dict,
        dataset: CrelloProcessor,
        target_prefix_list: List,
        text_index: int,
        batch_index: int = 0,
    ) -> Dict:
        outs = {}
        for prefix in target_prefix_list:
            out = model_outs[f"{prefix}"]
            if prefix in dataset.tokenizer.rawdata_list:
                data = out[batch_index][text_index].data.cpu().numpy()
            else:
                sorted_label = torch.sort(
                    input=out[batch_index], dim=1, descending=True
                )[1]
                target_label = 0  # top1
                data = sorted_label[text_index][target_label].item()

            outs[f"{prefix}"] = data

        return outs

    def store(
        self,
        outs_all: Dict,
        outs: Dict,
        target_prefix_list: List,
        text_index: int,
    ) -> Dict:
        for prefix in target_prefix_list:
            outs_all[f"{prefix}"][text_index] = outs[f"{prefix}"]
        return outs_all

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
        src, text_mask_src, feat_cat = self.emb(model_inputs)
        z = self.enc(src, text_mask_src)

        outs_all = {}
        for prefix in target_prefix_list:
            outs_all[prefix] = {}
            for t in range(0, start_index):
                tar = getattr(model_inputs, f"{prefix}")[0, t].item()
                outs_all[prefix][t] = tar
        for t in range(start_index, target_text_num):
            zd = self.dec(feat_cat, z, model_inputs)
            model_outs = self.head(zd, feat_cat)
            out_labels = self.get_labels(model_outs, dataset, target_prefix_list, t)
            outs = self.get_outs(model_outs, dataset, target_prefix_list, t)
            model_inputs.update_th_style_attributes(
                self.embedding_config_element, target_prefix_list, out_labels, t
            )
            outs_all = self.store(outs_all, outs, target_prefix_list, t)
        return outs_all

    def get_transformer_weight(
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
        src, text_mask_src, feat_cat = self.emb(model_inputs)
        z = self.enc(src, text_mask_src)
        out_labels_all = {}
        for prefix in target_prefix_list:
            out_labels_all[prefix] = np.zeros((target_text_num, 1))
            for t in range(0, start_index):
                tar = getattr(model_inputs, f"{prefix}")[0, t].item()
                out_labels_all[prefix][t, 0] = tar
        weights = []
        for t in range(start_index, target_text_num):
            zd, _weights = self.dec.get_transformer_weight(feat_cat, z, model_inputs)
            weights.append(_weights[0, t])
            model_outs = self.head(zd, feat_cat)
            out_labels = self.get_labels(model_outs, dataset, target_prefix_list, t)
            model_inputs.update_th_style_attributes(
                self.embedding_config_element, target_prefix_list, out_labels, t
            )
            out_labels_all = self.store(
                out_labels_all, out_labels, target_prefix_list, t
            )
        if len(weights) > 0:
            weights = torch.stack(weights, dim=0)
            return weights
        else:
            dummy_weights = torch.zeros((1, 1)).to(src.device)
            return dummy_weights

    def sample_labels(
        self,
        model_outs: Dict,
        target_prefix_list: List,
        text_index: int,
        sampling_param_geometry: float = 0.5,
        sampling_param_semantic: float = 0.9,
        batch_index: int = 0,
    ) -> Dict:
        out_labels = {}
        for prefix in target_prefix_list:
            out = model_outs[f"{prefix}"][batch_index]
            if getattr(self, f"{prefix}_att_type") == "semantic":
                sampling_param = sampling_param_semantic
            elif getattr(self, f"{prefix}_att_type") == "geometry":
                sampling_param = sampling_param_geometry
            out_label = sample_label(out, text_index, sampling_param)
            out_labels[f"{prefix}"] = out_label

        return out_labels

    def sample(
        self,
        model_inputs: ModelInput,
        target_prefix_list: List,
        sampling_param_geometry: float = 0.7,
        sampling_param_semantic: float = 0.7,
        start_index: int = 0,
        **kwargs: Any,
    ) -> Tensor:
        target_text_num = int(model_inputs.canvas_text_num[0].item())
        start_index = min(start_index, target_text_num)
        for t in range(start_index, target_text_num):
            model_inputs.zeroinitialize_th_style_attributes(target_prefix_list, t)
        src, text_mask_src, feat_cat = self.emb(model_inputs)
        z = self.enc(src, text_mask_src)

        outs_all = {}
        for prefix in target_prefix_list:
            outs_all[prefix] = {}
        for t in range(start_index, target_text_num):
            zd = self.dec(feat_cat, z, model_inputs)
            model_outs = self.head(zd, feat_cat)
            out_labels = self.sample_labels(
                model_outs,
                target_prefix_list,
                t,
                sampling_param_geometry,
                sampling_param_semantic,
            )
            model_inputs.update_th_style_attributes(
                self.embedding_config_element, target_prefix_list, out_labels, t
            )
            outs_all = self.store(outs_all, out_labels, target_prefix_list, t)
        return outs_all

    def sample_labels_with_structure(
        self,
        model_outs: Dict,
        target_prefix_list: List,
        text_index: int,
        label_link: Dict,
        used_labels: Dict,
        index2samestructureindexes: Dict,
        sampling_param_geometry: float = 0.5,
        sampling_param_semantic: float = 0.9,
        batch_index: int = 0,
    ) -> Dict:
        outs_all = {}
        for prefix in target_prefix_list:
            _label_link = label_link[prefix][text_index]
            _used_labels = used_labels[prefix]
            if _label_link is None:
                out = model_outs[f"{prefix}"][batch_index]
                cnt = 0
                sampling_type = getattr(self, f"{prefix}_att_type")
                if sampling_type == "semantic":
                    sampling_param = sampling_param_semantic
                elif sampling_type == "geometry":
                    sampling_param = sampling_param_geometry

                out_label = sample_label(out, text_index, sampling_param)
                max_val = max(
                    torch.sum(F.softmax(out, 1)[text_index]).item(), sampling_param
                )
                while out_label in _used_labels:
                    out_label = sample_label(out, text_index, sampling_param)
                    cnt += 1
                    if cnt > 10:
                        sampling_param += abs((max_val - sampling_param) * 0.1)
                    if cnt > 1000:
                        sampling_param *= 2

                samestructureindexes = index2samestructureindexes[prefix][text_index]
                label_link[prefix] = update_label_link(
                    label_link[prefix], samestructureindexes, out_label
                )
                used_labels[prefix].append(out_label)
            else:
                out_label = _label_link
            outs_all[f"{prefix}"] = out_label

        return outs_all

    def structure_preserved_sample(
        self,
        model_inputs: ModelInput,
        dataset: CrelloProcessor,
        target_prefix_list: List,
        sampling_param_geometry: float = 0.7,
        sampling_param_semantic: float = 0.7,
        start_index: int = 0,
    ) -> Tensor:
        target_text_num = int(model_inputs.canvas_text_num[0].item())
        preds = self.prediction(model_inputs, dataset, target_prefix_list)
        index2samestructureindexes, label_link, used_labels = label_linkage(
            target_prefix_list, target_text_num, preds
        )

        for t in range(start_index, target_text_num):
            model_inputs.zeroinitialize_th_style_attributes(target_prefix_list, t)
        src, text_mask_src, feat_cat = self.emb(model_inputs)
        z = self.enc(src, text_mask_src)

        outs_all = {}
        for prefix in target_prefix_list:
            outs_all[prefix] = {}
        for t in range(start_index, target_text_num):
            zd = self.dec(feat_cat, z, model_inputs)
            model_outs = self.head(zd, feat_cat)
            out_labels = self.sample_labels_with_structure(
                model_outs,
                target_prefix_list,
                t,
                label_link,
                used_labels,
                index2samestructureindexes,
                sampling_param_geometry,
                sampling_param_semantic,
            )
            model_inputs.update_th_style_attributes(
                self.embedding_config_element, target_prefix_list, out_labels, t
            )
            outs_all = self.store(outs_all, out_labels, target_prefix_list, t)
        return outs_all

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
