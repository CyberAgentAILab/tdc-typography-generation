from typing import Any, List, Tuple

import torch
from torch import Tensor, nn
from typography_generation.config.attribute_config import (
    TextElementContextEmbeddingAttributeConfig,
    TextElementContextPredictionAttributeConfig,
)
from typography_generation.io.data_object import ModelInput
from typography_generation.model.common import (
    ConstEmbedding,
    Linearx2,
    MyTransformerDecoder,
    MyTransformerDecoderLayer,
    fn_ln_relu,
)


class Decoder(nn.Module):
    def __init__(
        self,
        prefix_list_target: List,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
        d_model: int = 256,
        n_head: int = 8,
        dropout: float = 0.1,
        num_decoder_layers: int = 4,
        seq_length: int = 50,
        positional_encoding: bool = True,
        autoregressive_scheme: bool = True,
    ):
        super().__init__()

        self.prefix_list_target = prefix_list_target
        self.d_model = d_model
        self.dropout_element = dropout
        self.seq_length = seq_length
        self.autoregressive_scheme = autoregressive_scheme
        dim_feedforward = d_model * 2

        # Positional encoding in transformer
        position = torch.arange(0, 1, dtype=torch.long).unsqueeze(1)
        self.register_buffer("position", position)
        self.pos_embed = nn.Embedding(1, d_model)
        self.embedding = ConstEmbedding(d_model, seq_length, positional_encoding)

        # Decoder layer
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model, n_head, dim_feedforward, dropout
        # )
        decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = nn.TransformerDecoder(
        #     decoder_layer, num_decoder_layers, decoder_norm
        # )
        decoder_layer = MyTransformerDecoderLayer(
            d_model, n_head, dim_feedforward, dropout
        )
        self.decoder = MyTransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        # mask config in training
        mask_tgt = torch.ones((seq_length, seq_length))
        mask_tgt = torch.triu(mask_tgt == 1).transpose(0, 1)
        self.mask_tgt = mask_tgt.float().masked_fill(mask_tgt == 0, float("-inf"))
        # Build bart embedding
        self.build_bart_embedding(prefix_list_target, embedding_config_element)

    def build_bart_embedding(
        self,
        prefix_list_target: List,
        embedding_config_element: TextElementContextEmbeddingAttributeConfig,
    ) -> None:
        for prefix in prefix_list_target:
            target_embedding_config = getattr(embedding_config_element, prefix)
            kwargs = target_embedding_config.emb_layer_kwargs  # dict of args
            emb_layer = nn.Embedding(**kwargs)
            emb_layer = fn_ln_relu(emb_layer, self.d_model, self.dropout_element)
            setattr(self, f"{prefix}_emb", emb_layer)
            setattr(self, f"{prefix}_flag", target_embedding_config.flag)

    def get_features_via_fn(
        self, fn: Any, inputs: List, batch_num: int, text_num: Tensor
    ) -> Tensor:
        inputs_fn = []
        for b in range(batch_num):
            tn = int(text_num[b].item())
            for t in range(tn):
                inputs_fn.append(inputs[b][t].view(-1))
        outs = None
        if len(inputs_fn) > 0:
            inputs_fn = torch.stack(inputs_fn)
            outs = fn(inputs_fn)
            outs = outs.view(len(inputs_fn), self.d_model)
        feat = torch.zeros(self.seq_length, batch_num, self.d_model)
        feat = feat.to(text_num.device).float()
        cnt = 0
        for b in range(batch_num):
            tn = int(text_num[b].item())
            for t in range(tn):
                if outs is not None:
                    feat[t, b] = outs[cnt]
                cnt += 1
        return feat

    def get_style_context_embedding(
        self, model_inputs: ModelInput, batch_num: int, text_num: Tensor
    ) -> Tensor:
        feat = torch.zeros(self.seq_length, batch_num, self.d_model)
        feat = feat.to(text_num.device).float()
        for prefix in self.prefix_list_target:
            inp = getattr(model_inputs, prefix).long()
            layer = getattr(self, f"{prefix}_emb")
            f = self.get_features_via_fn(layer, inp, batch_num, text_num)
            feat = feat + f
        feat = feat / len(self.prefix_list_target)
        return feat

    def shift_context_feat(
        self, context_feat: Tensor, batch_num: int, text_num: Tensor
    ) -> Tensor:
        shifted_context_feat = torch.zeros(self.seq_length, batch_num, self.d_model)
        shifted_context_feat = shifted_context_feat.to(text_num.device).float()
        for b in range(batch_num):
            tn = int(text_num[b].item())
            for t in range(tn - 1):
                shifted_context_feat[t + 1, b] = context_feat[t, b]
        return shifted_context_feat

    def get_bart_embedding(self, src: Tensor, model_inputs: ModelInput) -> Tensor:
        batch_num, text_num = model_inputs.batch_num, model_inputs.canvas_text_num
        context_feat = self.get_style_context_embedding(
            model_inputs, batch_num, text_num
        )
        shifted_context_feat = self.shift_context_feat(
            context_feat, batch_num, text_num
        )
        position_feat = self.embedding(src)
        return shifted_context_feat + position_feat

    def forward(
        self,
        src: Tensor,
        z: Tensor,
        model_inputs: ModelInput,
    ) -> Tensor:
        if self.autoregressive_scheme is True:
            src = self.get_bart_embedding(src, model_inputs)
        else:
            src = self.embedding(src)
        mask_tgt = self.mask_tgt.to(src.device)
        out = self.decoder(src, z, mask_tgt, tgt_key_padding_mask=None)
        return out

    def get_transformer_weight(
        self,
        src: Tensor,
        z: Tensor,
        model_inputs: ModelInput,
    ) -> Tuple:
        if self.autoregressive_scheme is True:
            src = self.get_bart_embedding(src, model_inputs)
        else:
            src = self.embedding(src)
        mask_tgt = self.mask_tgt.to(src.device)
        out, weights = self.decoder(
            src, z, mask_tgt, tgt_key_padding_mask=None, get_weight=True
        )
        return out, weights


class FCN(nn.Module):
    def __init__(self, d_model: int, label_num: int, bypass: bool) -> None:
        super().__init__()
        self.bypass = bypass
        if bypass is True:
            self.fcn = Linearx2(d_model * 2, label_num)
        else:
            self.fcn = Linearx2(d_model, label_num)

    def forward(self, inp: Tensor, elm_agg_emb: Tensor = None) -> Tensor:
        if self.bypass is True:
            inp = torch.cat((inp, elm_agg_emb), 2)
        logits = inp.permute(1, 0, 2)
        logits = self.fcn(logits)  # Shape [G, N, 2]
        return logits


class MultiTask(nn.Module):
    def __init__(
        self,
        prefix_list_target: List,
        prediction_config: TextElementContextPredictionAttributeConfig,
        d_model: int = 256,
        bypass: bool = True,
    ):
        super().__init__()
        self.prefix_list_target = prefix_list_target
        for prefix in self.prefix_list_target:
            target_prediction_config = getattr(prediction_config, prefix)
            layer = FCN(d_model, target_prediction_config.out_dim, bypass)
            setattr(self, f"{prefix}_layer", layer)

    def forward(self, z: Tensor, elm_agg_emb: Tensor) -> dict:
        outputs = {}
        for prefix in self.prefix_list_target:
            layer = getattr(self, f"{prefix}_layer")
            out = layer(z, elm_agg_emb)
            outputs[prefix] = out
        return outputs
