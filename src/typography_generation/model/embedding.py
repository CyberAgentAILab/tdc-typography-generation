import time
from collections import OrderedDict
from typing import Any, List, Tuple

import torch
from einops import rearrange, repeat
from logzero import logger
from torch import Tensor, nn
from torch.functional import F

from typography_generation.config.attribute_config import (
    CanvasContextEmbeddingAttributeConfig, EmbeddingConfig,
    TextElementContextEmbeddingAttributeConfig)
from typography_generation.io.data_object import ModelInput
from typography_generation.model.common import _conv3x3_bn_relu, fn_ln_relu


def set_tensor_type(inp: Tensor, tensor_type: str) -> Tensor:
    if tensor_type == "float":
        inp = inp.float()
    elif tensor_type == "long":
        inp = inp.long()
    else:
        raise NotImplementedError()
    return inp


def setup_emb_layer(
    self: nn.Module, prefix: str, target_embedding_config: EmbeddingConfig
) -> None:
    if target_embedding_config.emb_layer is not None:
        kwargs = target_embedding_config.emb_layer_kwargs  # dict of args
        if target_embedding_config.emb_layer == "nn.Embedding":
            emb_layer = nn.Embedding(**kwargs)
            setattr(self, f"{prefix}_type", "long")
        elif target_embedding_config.emb_layer == "nn.Linear":
            emb_layer = nn.Linear(**kwargs)
            setattr(self, f"{prefix}_type", "float")
        else:
            raise NotImplementedError()
        emb_layer = fn_ln_relu(emb_layer, self.d_model, self.dropout)
        setattr(self, f"{prefix}_emb", emb_layer)
    else:
        setattr(self, f"{prefix}_type", "float")
    setattr(self, f"{prefix}_flag", target_embedding_config.flag)
    setattr(self, f"{prefix}_specific", target_embedding_config.specific_func)
    setattr(self, f"{prefix}_inp", target_embedding_config.input_prefix)


def get_output(
    self: nn.Module, prefix: str, model_inputs: ModelInput, batch_num: int
) -> Tensor:
    inp_prefix = getattr(self, f"{prefix}_inp")
    specific_func = getattr(self, f"{prefix}_specific")
    tensor_type = getattr(self, f"{prefix}_type")
    inp = getattr(model_inputs, inp_prefix)
    text_num = getattr(model_inputs, "canvas_text_num")
    inp = set_tensor_type(inp, tensor_type)
    if specific_func is not None:
        fn = getattr(self, specific_func)
        out = fn(
            inputs=inp,
            batch_num=batch_num,
            text_num=text_num,
        )
    else:
        fn = getattr(self, f"{prefix}_emb")
        out = self.get_features_via_fn(
            fn=fn, inputs=inp, batch_num=batch_num, text_num=text_num
        )
    return out


class Down(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(Down, self).__init__()
        self.conv1 = _conv3x3_bn_relu(input_dim, output_dim, kernel_size=3)
        self.drop = nn.Dropout()

    def forward(self, feat: Tensor) -> Tensor:
        feat = self.conv1(feat)
        return self.drop(feat)


class Embedding(nn.Module):
    def __init__(
        self,
        prefix_list_element: List,
        prefix_list_canvas: List,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
        embedding_config_canvas: CanvasContextEmbeddingAttributeConfig,
        d_model: int = 256,
        dropout: float = 0.1,
        seq_length: int = 50,
    ) -> None:
        super(Embedding, self).__init__()
        self.emb_element = EmbeddingElementContext(
            prefix_list_element, embedding_config_element, d_model, dropout, seq_length
        )
        self.emb_canvas = EmbeddingCanvasContext(
            prefix_list_canvas,
            embedding_config_canvas,
            d_model,
            dropout,
        )
        self.prefix_list_element = prefix_list_element
        self.prefix_list_canvas = prefix_list_canvas

        self.d_model = d_model
        self.seq_length = seq_length

        mlp_dim = (
            self.compute_modality_num(embedding_config_canvas, embedding_config_element)
            * d_model
        )
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("l1", nn.Linear(mlp_dim, d_model)),
                    ("r1", nn.LeakyReLU(0.2)),
                    ("l2", nn.Linear(d_model, d_model)),
                    ("r2", nn.LeakyReLU(0.2)),
                ]
            )
        )
        self.mlp_dim = mlp_dim

    def compute_modality_num(
        self,
        embedding_config_canvas: CanvasContextEmbeddingAttributeConfig,
        embedding_config_text: TextElementContextEmbeddingAttributeConfig,
    ) -> int:
        modality_num = 0
        for prefix in self.prefix_list_element:
            target_embedding_config = getattr(embedding_config_text, prefix)
            flag = target_embedding_config.flag
            if flag is False:
                pass
            else:
                modality_num += 1
        for prefix in self.prefix_list_canvas:
            target_embedding_config = getattr(embedding_config_canvas, prefix)
            flag = target_embedding_config.flag
            if flag is True:
                modality_num += 1
        return modality_num

    def flatten_feature(
        self,
        feats: List,
    ) -> Tensor:
        feats_flatten = []
        modarity_num = len(feats)
        for m in range(modarity_num):
            feats_flatten.append(feats[m])
        return feats_flatten

    def reshape_tbc(
        self,
        feat_elements: List,  # modality num(M) x [B, S, C]
        feat_canvas: List,  # canvas num(CV) x [B, C]
        batch_num: int,
        text_num: Tensor,
    ) -> Tensor:
        feat_elements = torch.stack(feat_elements)  # M, B, S, C
        feat_elements = rearrange(feat_elements, "m b s c -> s b (m c)")
        feat_canvas = torch.stack(feat_canvas)  # CV, B, C
        feat_canvas = rearrange(feat_canvas, "cv b c -> 1 b (cv c)")
        feat_canvas = repeat(feat_canvas, "1 b c -> s b c", s=self.seq_length)
        feat = torch.cat((feat_canvas, feat_elements), dim=2)
        feat = self.mlp(feat)
        return feat

    def get_transformer_inputs(
        self,
        feat_elements: List[Tensor],
        feat_canvas: Tensor,
        batch_num: int,
        text_num: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        feat_element, text_mask_element = self.lineup_features(
            feat_elements, batch_num, text_num
        )
        feat_canvas = torch.stack(feat_canvas)
        feat_canvas = feat_canvas.view(
            feat_canvas.shape[0], feat_element.shape[1], feat_element.shape[2]
        )
        src = torch.cat((feat_canvas, feat_element), dim=0)
        canvas_mask = torch.zeros(src.shape[1], feat_canvas.shape[0]) + 1
        canvas_mask = canvas_mask.float().to(text_num.device)
        text_mask_src = torch.cat((canvas_mask, text_mask_element), dim=1)
        return src, text_mask_src

    def lineup_features(
        self, feats: Tensor, batch_num: int, text_num: Tensor
    ) -> Tuple[Tensor, Tensor]:
        modality_num = len(feats)
        device = text_num.device
        feat = (
            torch.zeros(self.seq_length, modality_num, batch_num, self.d_model)
            .float()
            .to(device)
        )
        for m in range(modality_num):
            logger.debug(f"{feats[m].shape=}")
            feat[:, m, :, :] = rearrange(feats[m], "b t c -> t b c")
        feat = rearrange(feat, "t m b c -> (t m) b c")
        indices = rearrange(torch.arange(self.seq_length), "s -> 1 s").to(device)
        mask = (
            (indices < text_num).to(device).float()
        )  # (B, S), indicating valid attribute locations
        text_mask = repeat(mask, "b s -> b (s m)", m=modality_num)
        return feat, text_mask

    def forward(self, model_inputs: ModelInput) -> Tuple[Tensor, Tensor, Tensor]:
        feat_elements = self.emb_element(model_inputs)
        feat_canvas = self.emb_canvas(model_inputs)

        src, text_mask_src = self.get_transformer_inputs(
            feat_elements,
            feat_canvas,
            model_inputs.batch_num,
            model_inputs.canvas_text_num,
        )
        feat_cat = self.reshape_tbc(
            feat_elements,
            feat_canvas,
            model_inputs.batch_num,
            model_inputs.canvas_text_num,
        )

        return (src, text_mask_src, feat_cat)


class EmbeddingElementContext(nn.Module):
    def __init__(
        self,
        prefix_list_element: List,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
        d_model: int,
        dropout: float = 0.1,
        seq_length: int = 50,
    ):
        super(EmbeddingElementContext, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.seq_length = seq_length
        self.prefix_list_element = prefix_list_element

        for prefix in self.prefix_list_element:
            target_embedding_config = getattr(embedding_config_element, prefix)
            setup_emb_layer(self, prefix, target_embedding_config)
            if target_embedding_config.specific_build is not None:
                build_func = getattr(self, target_embedding_config.specific_build)
                build_func()

    def get_local_feat(
        self, inputs: Tensor, batch_num: int, text_num: Tensor, **kwargs: Any
    ) -> Tensor:
        inputs_fn = []
        for b in range(batch_num):
            tn = int(text_num[b].item())
            for t in range(tn):
                inputs_fn.append(inputs[b, t])
        outs = None
        if len(inputs_fn) > 0:
            inputs_fn = torch.stack(inputs_fn)
            with torch.no_grad():
                outs = self.imgencoder(inputs_fn).detach()
            outs = self.downdim(outs)
            outs = outs.view(len(outs), -1)
            outs = self.fcimg(outs)
        return outs

    # def get_features_via_fn(
    #     self, fn: Any, inputs: List, batch_num: int, text_num: Tensor
    # ) -> Tensor:
    #     inputs_layer = []
    #     for b in range(batch_num):
    #         tn = int(text_num[b].item())
    #         for t in range(tn):
    #             inputs_layer.append(inputs[b][t].view(-1))
    #     outs = None
    #     if len(inputs_layer) > 0:
    #         inputs_layer = torch.stack(inputs_layer)
    #         outs = fn(inputs_layer)
    #         outs = outs.view(len(inputs_layer), self.d_model)
    #     return outs

    def get_features_via_fn(
        self, fn: Any, inputs: Tensor, batch_num: int, text_num: Tensor
    ) -> Tensor:
        device = inputs.device
        feat = torch.zeros(batch_num, self.seq_length, self.d_model).float().to(device)
        indices = rearrange(torch.arange(self.seq_length), "s -> 1 s").to(device)
        mask = (indices < text_num).to(
            device
        )  # (B, S), indicating valid attribute locations
        feat[mask] = fn(inputs[mask])  # (B, S, C)
        return feat

    def text_emb_layer(
        self, inputs: Tensor, batch_num: int, text_num: Tensor, **kwargs: Any
    ) -> Tensor:
        inputs_fn = []
        for b in range(batch_num):
            tn = int(text_num[b].item())
            for t in range(tn):
                inputs_fn.append(inputs[b, t].view(-1))
        outs = None
        if len(inputs_fn) > 0:
            inputs_fn = torch.stack(inputs_fn)
            outs = self.text_emb_emb(inputs_fn)
            outs = outs.view(len(inputs_fn), self.d_model)
        return outs

    def text_local_img_emb_layer(
        self, inputs: Tensor, batch_num: int, text_num: Tensor, **kwargs: Any
    ) -> Tensor:
        inputs_fn = []
        for b in range(batch_num):
            tn = int(text_num[b].item())
            for t in range(tn):
                inputs_fn.append(inputs[b, t].view(-1))
        outs = None
        if len(inputs_fn) > 0:
            inputs_fn = torch.stack(inputs_fn)
            outs = self.text_local_img_emb_emb(inputs_fn)
            outs = outs.view(len(inputs_fn), self.d_model)
        return outs

    def text_font_emb_layer(
        self, inputs: Tensor, batch_num: int, text_num: Tensor
    ) -> Tensor:
        inputs_fn = []
        for b in range(batch_num):
            tn = int(text_num[b].item())
            for t in range(tn):
                inputs_fn.append(inputs[b, t].view(-1))
        outs = None
        if len(inputs_fn) > 0:
            inputs_fn = torch.stack(inputs_fn)
            outs = self.text_font_emb_emb(inputs_fn)
            outs = outs.view(len(inputs_fn), self.d_model)
        return outs

    def forward(self, model_inputs: ModelInput) -> Tensor:
        feats = []
        for prefix in self.prefix_list_element:
            flag = getattr(self, f"{prefix}_flag")
            if flag is True:
                start = time.time()
                logger.debug(f"{prefix=}")
                out = get_output(self, prefix, model_inputs, model_inputs.batch_num)
                logger.debug(f"{prefix=} {out.shape=} {time.time()-start}")
                feats.append(out)
        return feats


class EmbeddingCanvasContext(nn.Module):
    def __init__(
        self,
        canvas_prefix_list: List,
        embedding_config_canvas: CanvasContextEmbeddingAttributeConfig,
        d_model: int,
        dropout: float,
    ):
        super(EmbeddingCanvasContext, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.prefix_list = canvas_prefix_list
        for prefix in self.prefix_list:
            target_embedding_config = getattr(embedding_config_canvas, prefix)
            setup_emb_layer(self, prefix, target_embedding_config)
            if target_embedding_config.specific_build is not None:
                build_func = getattr(self, target_embedding_config.specific_build)
                build_func()

    def get_feat(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        with torch.no_grad():
            feat = self.imgencoder(inputs.float()).detach()
        feat = F.relu(self.avgpool(feat))
        feat = self.downdim(feat)
        feat = feat.view(feat.shape[0], feat.shape[1])
        return feat

    def canvas_bg_img_emb_layer(self, inputs: Tensor, **kwargs: Any) -> Tensor:
        feat = self.canvas_bg_img_emb_emb(inputs.float())
        feat = feat.view(feat.shape[0], feat.shape[1])
        return feat

    def get_features_via_fn(self, fn: Any, inputs: Tensor, **kwargs: Any) -> Tensor:
        out = fn(inputs)
        return out

    def forward(self, model_inputs: ModelInput) -> List:
        feats = []
        for prefix in self.prefix_list:
            flag = getattr(self, f"{prefix}_flag")
            if flag is True:
                logger.debug(f"canvas prefix {prefix}")
                out = get_output(self, prefix, model_inputs, model_inputs.batch_num)
                feats.append(out)
        return feats
