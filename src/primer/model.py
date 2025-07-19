# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from typing import ClassVar, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchtitan.models.attention import init_attn_mask
from torchtitan.protocols.train_spec import ModelProtocol

from primer.config import ActFnType, ModelConfig, NormConfig, NormType, QKNormConfig

from .args import TransformerModelArgs

SDPBackendType = SDPBackend


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "swish": F.silu,
    "mish": F.mish,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}
SUPPORTED_FUNCTIONS = set(get_args(ActFnType))
assert set(ACT2FN.keys()) == SUPPORTED_FUNCTIONS, (
    f"ACT2FN keys {set(ACT2FN.keys())} don't match supported functions {SUPPORTED_FUNCTIONS} in config.py"
)


NORM2CLASS = {"layernorm": nn.LayerNorm, "rmsnorm": nn.RMSNorm}
SUPPORTED_NORM_TYPES = set(get_args(NormType))
assert set(NORM2CLASS.keys()) == SUPPORTED_NORM_TYPES, (
    f"NORM2CLASS keys {set(NORM2CLASS.keys())} don't match supported norm types {SUPPORTED_NORM_TYPES} in config.py"
)


# ==========================================================================================
# Attention Block
# ==========================================================================================
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float | None): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (Tensor): Frequency tensor to be reshaped.
        x (Tensor): Target tensor for broadcasting compatibility.

    Returns:
        Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (Tensor): Query tensor to apply rotary embeddings.
        xk (Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        tuple[Tensor, Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class ScaledDotProductAttention(nn.Module):
    backends: ClassVar[list[SDPBackendType]] = []

    def __init__(self, attn_mask_type: str, dropout_p: float) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        if attn_mask_type != "causal":
            raise ValueError("TorchTitan with SDPA currently only supports causal mask.")

        ScaledDotProductAttention._init_backend()

    @classmethod
    def _init_backend(cls) -> None:
        if not cls.backends:
            cls.backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

            # Add CuDNN on B200 w/ highest priority
            MAJ_MIN_CUDA_CAP = (10, 0)
            if torch.cuda.is_available() and torch.cuda.get_device_capability() >= MAJ_MIN_CUDA_CAP:
                cls.backends.insert(0, SDPBackend.CUDNN_ATTENTION)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        assert self.backends, "SDPA Backends should not be empty."
        with sdpa_kernel(self.backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout_p)


class Attention(nn.Module):
    """Multi-head attention module.

    Args:
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        n_kv_heads (int | None): Number of key-value heads. If None, set to n_heads.
        dropout_p (float): Dropout probability. Set to 0.0 for turning it off.
        bias (bool): Whether to use bias in linear layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout_p: float,
        bias: bool,
        use_rope: bool = True,
        qk_norm_config: QKNormConfig | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        
        # Allow for NoPe
        # NOTE: Using a boolean flag is compile-friendly as it allows the compiler to
        # optimize away conditional branches when the value is known at compile time.
        self.use_rope = use_rope

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)
        self.sdpa = ScaledDotProductAttention("causal", dropout_p=dropout_p)

        self.use_qk_norm = False
        self.norm_only_head_dim = False
        if qk_norm_config:
            self.use_qk_norm = True            
            self.norm_only_head_dim = qk_norm_config.only_head_dim
            
            qnorm_dim = self.head_dim if self.norm_only_head_dim else n_heads * self.head_dim
            knorm_dim = self.head_dim if self.norm_only_head_dim else n_kv_heads * self.head_dim
            self.qnorm = NORM2CLASS[qk_norm_config.type](qnorm_dim, eps=qk_norm_config.eps, **qk_norm_config.kwargs)
            self.knorm = NORM2CLASS[qk_norm_config.type](knorm_dim, eps=qk_norm_config.eps, **qk_norm_config.kwargs)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        bs, seqlen, _ = x.shape
        xq = self.wq(x)  # (bs, seqlen, n_local_heads * head_dim)
        xk = self.wk(x)  # (bs, seqlen, n_kv_heads * head_dim)
        xv = self.wv(x)  # (bs, seqlen, n_kv_heads * head_dim)

        if self.use_qk_norm and not self.norm_only_head_dim:
            # Normalize over the full concatenated dimension first
            xq = self.qnorm(xq)  # no shape change
            xk = self.knorm(xk)  # no shape change
    
        # NOTE: using -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual local heads
        # from sizes of xq, xk, and xv as TP may have sharded them after the above linear ops
        xq = xq.view(bs, seqlen, -1, self.head_dim)  # (bs, seqlen, n_local_heads, head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)  # (bs, seqlen, n_kv_heads, head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)  # (bs, seqlen, n_kv_heads, head_dim)

        if self.use_qk_norm and self.norm_only_head_dim:
            # Normalize each head independently after reshaping
            xq = self.qnorm(xq)  # no shape change
            xk = self.knorm(xk)  # no shape change

        # Apply rotary embeddings if provided (supports NoPe)
        if self.use_rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)  # no shape change

        # repeat k/v heads if n_kv_heads < n_heads
        # NOTE: in theory this could be handled by F.scaled_dot_product_attention, check whether we can remove this
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_kv_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)  # (bs, seqlen, n_local_heads * head_dim or d_model)
        return self.wo(output)  # (bs, seqlen, n_local_heads * head_dim or d_model)

    def init_weights(self, init_std: float) -> None:
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)


# ==========================================================================================
# FeedForward Block
# ==========================================================================================
class FeedForward(nn.Module):
    """Feedforward module with GLU activation.
    Args:
        d_model (int): Dimension of the model.
        intermediate_size (int): Size of the intermediate layer.
        act_fn (str): Activation function to use. Defaults to "silu".
        bias (bool): Whether to use bias in linear layers. Defaults to False.
        multiple_of (int): Multiple of which the intermediate size should be a multiple. Defaults to 64.
        size_multiplier (float | None): Multiplier for the intermediate size. Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        act_fn: str = "silu",
        bias: bool = False,
        multiple_of: int = 64,
        size_multiplier: float | None = None,
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.gated = gated

        # GLU variants scale down the intermediate_size by 2/3 to keep the number of parameters similar
        # to a standard MLP since the gating mechanism requires an additional linear layer
        _interm_size = int(2 * intermediate_size / 3) if self.gated else intermediate_size
        if size_multiplier is not None:
            _interm_size = int(size_multiplier * _interm_size)
        _interm_size = multiple_of * ((_interm_size + multiple_of - 1) // multiple_of)
        self.intermediate_size = _interm_size

        self.act_fn = ACT2FN[act_fn]
        self.wup = nn.Linear(d_model, self.intermediate_size, bias=bias)
        self.wdown = nn.Linear(self.intermediate_size, d_model, bias=bias)

        if self.gated:
            self.wgate = nn.Linear(d_model, self.intermediate_size, bias=bias)

    def _gated_forward(self, x: Tensor) -> Tensor:
        return self.wdown(self.act_fn(self.wgate(x)) * self.wup(x))

    def _standard_forward(self, x: Tensor) -> Tensor:
        return self.wdown(self.act_fn(self.wup(x)))

    def forward(self, x: Tensor) -> Tensor:
        return self._gated_forward(x) if self.gated else self._standard_forward(x)

    def init_weights(self, init_std: float) -> None:
        for linear in (self.wup, self.wdown):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        if self.gated:
            nn.init.trunc_normal_(self.wgate.weight, mean=0.0, std=0.02)


# ==========================================================================================
# Transformer Block
# ==========================================================================================
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig) -> None:
        super().__init__()
        # NOTE: using these boolean variables is torch.compile-friendly because:
        # 1. They are set once at initialization time (compile-time constants)
        # 2. torch.compile can optimize away the conditional branches when the boolean is known
        # 3. Using "if self.attn_norm_pre is not None" would be less efficient because:
        #    - It requires a runtime null check on every forward pass
        #    - torch.compile cannot optimize away the null check since it's a runtime condition
        #    - The boolean flags allow the compiler to eliminate dead code paths entirely
        self.parallel_layers = config.parallel_layers
        self.use_attn_pre = config.norm.attn_pre
        self.use_attn_post = config.norm.attn_post
        self.use_ffw_pre = config.norm.ffw_pre
        self.use_ffw_post = config.norm.ffw_post

        use_rope = config.nope.should_use_rope(layer_id)
        self.attn = Attention(
            config.d_model, 
            **config.attention.to_dict(), 
            use_rope=use_rope,
            qk_norm_config=config.norm.get_qk_norm_config(),
        )
        self.ffw = FeedForward(config.d_model, **config.ffw.to_dict())

        # norms
        def create_norm() -> nn.Module:
            return NORM2CLASS[config.norm.type](config.d_model, eps=config.norm.eps, **config.norm.kwargs)

        self.attn_norm_pre = create_norm() if self.use_attn_pre else None
        self.attn_norm_post = create_norm() if self.use_attn_post else None
        self.ffw_norm_pre = create_norm() if self.use_ffw_pre else None
        self.ffw_norm_post = create_norm() if self.use_ffw_post else None

        # Compute weight initialization standard deviation
        # 1. Per-layer depth scaling: deeper layers get smaller initialization
        # 2. Global depth scaling: all layers share the same initialization
        depth_factor = (layer_id + 1) if config.depth_init else config.n_layers
        self.weight_init_std = config.init_std / (2 * depth_factor) ** 0.5

    def _apply_attention(self, x: Tensor, freqs_cis: Tensor | None) -> Tensor:
        _x = x
        if self.use_attn_pre:
            _x = self.attn_norm_pre(_x)  # type: ignore
        _x = self.attn(_x, freqs_cis)
        if self.use_attn_post:
            _x = self.attn_norm_post(_x)  # type: ignore
        return _x

    def _apply_ffw(self, x: Tensor) -> Tensor:
        _x = x
        if self.use_ffw_pre:
            _x = self.ffw_norm_pre(_x)  # type: ignore
        _x = self.ffw(_x)
        if self.use_ffw_post:
            _x = self.ffw_norm_post(_x)  # type: ignore
        return _x

    def forward(self, x: Tensor, freqs_cis: Tensor | None = None) -> Tensor:
        if self.parallel_layers:
            attn_out = self._apply_attention(x, freqs_cis)
            ffw_out = self._apply_ffw(x)
            return x + attn_out + ffw_out

        attn_out = self._apply_attention(x, freqs_cis)
        h = x + attn_out
        ffw_out = self._apply_ffw(h)
        return h + ffw_out

    def init_weights(self) -> None:
        self.attn.init_weights(self.weight_init_std)
        self.ffw.init_weights(self.weight_init_std)
        for norm in (self.attn_norm_pre, self.attn_norm_post, self.ffw_norm_pre, self.ffw_norm_post):
            if norm:
                norm.reset_parameters()  # type: ignore -> the type checker is confused


class Transformer(nn.Module, ModelProtocol):
    """
    Transformer Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id

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
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None):
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
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
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

    def _precompute_freqs_cis(self) -> Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(self, tokens: Tensor, input_batch: Tensor | None = None):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            input_batch (Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).

        Returns:
            Tensor: Output logits after applying the Transformer model.

        """
        if self.model_args.use_flex_attn:
            init_attn_mask(input_batch if input_batch is not None else tokens, eos_id=self.eos_id)

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
