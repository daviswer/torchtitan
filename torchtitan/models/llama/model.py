# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Llama 2 is licensed under the LLAMA 2 Community License,
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.norms import build_norm
from torchtitan.models.llama.utils import SMVecMatMul, UniversalAttention

# from causal_conv1d import causal_conv1d_fn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses 1 (was the total number of transformer blocks)
    depth_init: bool = True
    norm_type: str = "rmsnorm"

    # # muP values
    # #   - Comments are: Left, our formula, Right, target
    # #   - Values calculated based on TinyLlama (init=.02, d=1024, head_d=128, growf=8/3)
    # mup_head_scale: float = 32.0  # 1/sqrt(d) * f  =  1
    # mup_attn_temp: float = 11.314  # 1/d * f  =  1/sqrt(d)  (d is head dim here)
    # # mup_attn_gain: float = 0.4096  # f  =  (.02*sqrt(d))**2
    # # mup_ffn_gain: float = 0.3027  # f  =  (.02*sqrt(d)/sqrt(2)) * (.02*sqrt(d)) * (.02*sqrt(d*growf))
    # # residual_downscale = 0.35212  # sqrt(attn_gain * ffn_gain)
    # # attn_gain = downscale*sqrt(skew)
    # # ffn_gain = downscale/sqrt(skew)
    # mup_a_f_skew: float = 1.35316  # (attn_gain / ffn_gain)
    # # set residual_downscale to 1, adjust emb scale and dscale to compensate
    # mup_emb_scale: float = 0.0568  # f  =  .02 / residual_downscale
    # # 2d weights are scaled to .02 / residual_downscale. Adjust LR same (don't worry about LN LR)
    # mup_lr_dscale: float = 90.8781 # 1/sqrt(d) * f = 1 / residual_downscale

    # muP values
    #   - Comments are: Left, our formula, Right, target
    #   - Values calculated based on DebugModel (init=.02, d=256, head_d=16, growf=3)
    mup_head_scale: float = 16.0  # 1/sqrt(d) * f  =  1
    mup_attn_temp: float = 4.0  # 1/d * f  =  1/sqrt(d)  (d is head dim here)
    # mup_attn_gain: float = 0.1024  # f  =  (.02*sqrt(d))**2
    # mup_ffn_gain: float = 0.04013  # f  =  (.02*sqrt(d)/sqrt(2)) * (.02*sqrt(d)) * (.02*sqrt(d*growf))
    # residual_downscale = 0.0641  # sqrt(attn_gain * ffn_gain)
    # attn_gain = downscale*sqrt(skew)
    # ffn_gain = downscale/sqrt(skew)
    mup_a_f_skew: float = 2.5516  # (attn_gain / ffn_gain)
    # set residual_downscale to 1, adjust emb scale and dscale to compensate
    mup_emb_scale: float = 15.6  # f  =  1 / residual_downscale
    # 2d weights are scaled to .02 / residual_downscale. Adjust LR same (don't worry about LN LR)
    mup_lr_dscale: float = 249.61 # 1/sqrt(d) * f = 1 / residual_downscale


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.wstatic = nn.Linear(model_args.dim, 2*self.n_kv_heads, bias=True)
        self.wstatic.bias._no_weight_decay = True
        self.UA = UniversalAttention.apply
        self.SMVMM = SMVecMatMul.apply
        # self.gn = build_norm(
        #     model_args.norm_type, dim=self.head_dim, eps=model_args.norm_eps
        # )
        # self.sinks = nn.Parameter(torch.empty(2, self.n_kv_heads, self.head_dim, self.head_dim))

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv, self.wstatic):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        static_max = .1
        static_min = .001
        nn.init.uniform_(self.wstatic.bias)
        self.wstatic.bias.data = self.wstatic.bias.data * (math.log(static_max) - math.log(static_min)) + math.log(static_min)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Normalize k
        xk = xk/xk.pow(2).sum(-1, True).sqrt().add(1e-6)

        # apply half-rope
        xq_ = xq[...,self.head_dim//2:]
        xk_ = xk[...,self.head_dim//2:]
        xq, xk = apply_rotary_emb(xq[...,:self.head_dim//2], xk[...,:self.head_dim//2], freqs_cis=freqs_cis)
        xq = torch.cat([xq,xq_], dim=3)
        xk = torch.cat([xk,xk_], dim=3)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        
        # Split out q if n_kv_heads < n_heads
        xq = xq.view(bs, -1, self.n_rep, seqlen, self.head_dim)  # b h r l d

        # blockwise self-pruning attention
        c = 512
        l = seqlen
        n = seqlen//c
        b = bs
        s = [b, -1, n, c, self.head_dim]
        kc = xk.view(*s)
        vc = xv.view(*s)
        kt = xk.transpose(-2,-1)  # b h d l
        output = [] #torch.zeros_like(xv)  # b h l d
        denom = []
        mask = torch.ones(c,l,device=xq.device,dtype=torch.bool)
        static = self.wstatic(x).sigmoid().view(bs, seqlen, 2, self.n_kv_heads).permute(2,0,3,1)  # 2 b h l
        static_src = static[0].view(b, self.n_kv_heads, n, c)  # b h n c
        static_dest = static[1]  # b h l

        # Perform universal attention
        output, denom = self.UA(kc, vc, xq, static_src, static_dest)  # b h r l d n, b h r l n

        # Weighted avg for final softmax
        output = self.SMVMM(output, denom)  # b h r l d

        # Reshape, project out
        # output = self.gn(output)
        output = output.permute(
            0, 3, 1, 2, 4
        ).reshape(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # self.conv = nn.Parameter(torch.empty(dim, 4))

    def forward(self, x):
        z = x
        # s = z.size()
        # d2 = s[-1]//2
        # z = z.view(s[0], -1).roll(d2, 1)
        # z[:,:d2] = 0
        # z = z.view(*s)
        # z = causal_conv1d_fn(x.transpose(1,2), self.conv).transpose(1,2)
        return self.w2(F.silu(self.w1(x)) * self.w3(z))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        # nn.init.trunc_normal_(self.conv, mean=0, std=.25)


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * self.num_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(self):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        with torch.device(self.freqs_cis.device):
            self.freqs_cis = self._precompute_freqs_cis()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads // 2,
            # Need to compute until at least the max token limit for generation
            # (use 2x max sequence length to be safe)
            self.model_args.max_seq_len * 2,
            self.model_args.rope_theta,
        )

    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # h: b l d
        s = h.size()
        h = h.view(s[0],-1)
        d2 = s[2]//2
        h = h.roll(d2,1)
        h[:,:d2] = 0
        h = h.view(*s)

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h).float() if self.output else h
        return output

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
