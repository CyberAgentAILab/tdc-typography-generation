from typing import Tuple

import torch
from torch import Tensor, nn

from typography_generation.model.common import (
    MyTransformerEncoder,
    MyTransformerEncoderLayer,
)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_head: int = 8,
        dropout: float = 0.1,
        num_encoder_layers: int = 4,
    ):
        super(Encoder, self).__init__()
        encoder_norm = nn.LayerNorm(d_model)
        dim_feedforward = d_model * 2
        encoder_layer = MyTransformerEncoderLayer(
            d_model, n_head, dim_feedforward, dropout
        )
        self.encoder = MyTransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

    def forward(self, src: Tensor, text_mask: Tensor) -> Tuple:
        text_mask_fill = text_mask.masked_fill(
            text_mask == 0, float("-inf")
        ).masked_fill(text_mask == 1, float(0.0))
        z = self.encoder(src, mask=None, src_key_padding_mask=text_mask_fill.bool())
        if torch.sum(torch.isnan(z)) > 0:
            z = z.masked_fill(torch.isnan(z), 0)
        return z

    def get_transformer_weight(self, src: Tensor, text_mask: Tensor) -> Tuple:
        text_mask_fill = text_mask.masked_fill(
            text_mask == 0, float("-inf")
        ).masked_fill(text_mask == 1, float(0.0))
        z, weights = self.encoder(
            src, mask=None, src_key_padding_mask=text_mask_fill, get_weight=True
        )
        if torch.sum(torch.isnan(z)) > 0:
            z = z.masked_fill(torch.isnan(z), 0)
        return z, weights
