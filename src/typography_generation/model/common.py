import copy
from collections import OrderedDict
from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor, nn
from torch.functional import F


def _get_clones(module: Any, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: Any) -> Any:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _generate_square_subsequent_mask(
    sz: int,
    device: torch.device = torch.device(
        torch._C._get_default_device()
    ),  # torch.device('cpu'),
    dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype
        )

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


class MyTransformerDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        get_weight: bool = False,
    ) -> Tensor:
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)

        for mod in self.layers:
            output, w = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        if get_weight is True:
            return output, w
        else:
            return output


class MyTransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model: float = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state: Any) -> None:
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, weight = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, weight


class MyTransformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self, encoder_layer: nn.Module, num_layers: int, norm: Any = None
    ) -> None:
        super(MyTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        get_weight: bool = False,
    ) -> Tensor:
        output = src
        for mod in self.layers:
            output, w = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)
        if get_weight is True:
            return output, w
        else:
            return output


class MyTransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state: Any) -> None:
        if "activation" not in state:
            state["activation"] = F.relu
        super(MyTransformerEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            sa, w = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + sa
            x = x + self._ff_block(self.norm2(x))
        else:
            sa, w = self._sa_block(x, src_mask, src_key_padding_mask)
            x = self.norm1(x + sa)
            x = self.norm2(x + self._ff_block(x))

        return x, w

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x, w = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        return self.dropout1(x), w

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _conv3x3_bn_relu(
    in_channels: int,
    out_channels: int,
    dilation: int = 1,
    kernel_size: int = 3,
    stride: int = 1,
) -> nn.Sequential:
    if dilation == 0:
        dilation = 1
        padding = 0
    else:
        padding = dilation
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    ),
                ),
                ("bn", nn.BatchNorm2d(out_channels)),
                ("relu", nn.ReLU()),
            ]
        )
    )


class LinearView(nn.Module):
    def __init__(self, channel: int) -> None:
        super(LinearView, self).__init__()
        self.channel = channel

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            BN, CHARN, CN = x.shape
            if CHARN == 1:
                x = x.view(BN, self.channel)
        return x


def fn_ln_relu(fn: Any, out_channels: int, dp: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            [
                ("fn", fn),
                ("ln", nn.LayerNorm(out_channels)),
                ("view", LinearView(out_channels)),
                ("dp", nn.Dropout(dp)),
            ]
        )
    )


class Linearx2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.fcn = nn.Sequential(
            OrderedDict(
                [
                    ("l1", nn.Linear(in_ch, in_ch)),
                    ("r1", nn.LeakyReLU(0.2)),
                    ("l2", nn.Linear(in_ch, out_ch)),
                ]
            )
        )

    def forward(self, out: Tensor) -> Tensor:
        logits = self.fcn(out)  # Shape [G, N, 2]
        return logits


class Linearx3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.fcn = nn.Sequential(
            OrderedDict(
                [
                    ("l1", nn.Linear(in_ch, in_ch)),
                    ("r1", nn.LeakyReLU(0.2)),
                    ("l1", nn.Linear(in_ch, in_ch)),
                    ("r1", nn.LeakyReLU(0.2)),
                    ("l2", nn.Linear(in_ch, out_ch)),
                ]
            )
        )

    def forward(self, out: Tensor) -> Tensor:
        logits = self.fcn(out)  # Shape [G, N, 2]
        return logits


class ConstEmbedding(nn.Module):
    def __init__(
        self, d_model: int = 256, seq_len: int = 50, positional_encoding: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.PE = PositionalEncodingLUT(
            d_model, max_len=seq_len, positional_encoding=positional_encoding
        )

    def forward(self, z: Tensor) -> Tensor:
        if len(z.shape) == 2:
            N = z.size(0)
        elif len(z.shape) == 3:
            N = z.size(1)
        else:
            raise Exception
        pos = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return pos


class PositionalEncodingLUT(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.1,
        max_len: int = 50,
        positional_encoding: bool = True,
    ):
        super(PositionalEncodingLUT, self).__init__()
        self.PS = positional_encoding
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer("position", position)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x: Tensor, inp_ignore: bool = False) -> Tensor:
        pos = self.position[: x.size(0)]
        x = self.pos_embed(pos).repeat(1, x.size(1), 1)
        return self.dropout(x)
