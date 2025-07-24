from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from primer.config import ACT2FN, NORM2CLASS, ModelConfig

SDPBackendType = SDPBackend


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
    """Reshape frequency tensor for broadcasting it with another tensor.

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
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # (bs, seqlen, n_heads, head_dim/2, 2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) # (bs, seqlen, n_kv_heads, head_dim/2, 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)  # (seqlen, head_dim/2, 2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape_as(xq)  # (bs, seqlen, n_heads, head_dim)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape_as(xk)  # (bs, seqlen, n_kv_heads, head_dim)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    if n_rep == 1:
        return x
    bs, seqlen, n_kv_heads, head_dim = x.shape
    return (
        torch.unsqueeze(x, dim=3)  # (bs, seqlen, n_kv_heads, 1, head_dim)
        .expand(bs, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)
    )


class ScaledDotProductAttention(nn.Module):
    backends: ClassVar[list[SDPBackendType]] = []

    def __init__(self, dropout_p: float) -> None:
        super().__init__()
        self.dropout_p = dropout_p
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
        head_dim: int | None = None,
        use_rope: bool = True,
        qknorm_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = d_model // n_heads if head_dim is None else head_dim
        self.n_rep = n_heads // n_kv_heads
        self.use_rope = use_rope

        self.use_qknorm = False
        self.qknorm_use_global = False
        if qknorm_kwargs is not None:
            assert isinstance(qknorm_kwargs, dict), "qknorm_kwargs must be a dictionary"
            fields = ["norm_type", "norm_eps", "use_global"]
            assert all(field in qknorm_kwargs for field in fields), f"qknorm_kwargs must contain '{fields}'"

            self.use_qknorm = True
            self.qknorm_use_global = qknorm_kwargs["use_global"]
            norm_type = qknorm_kwargs["norm_type"]
            norm_eps = qknorm_kwargs["norm_eps"]
            norm_kwargs = qknorm_kwargs.get("kwargs", {})
            qnorm_dim = n_heads * self.head_dim if self.qknorm_use_global else self.head_dim
            knorm_dim = n_kv_heads * self.head_dim if self.qknorm_use_global else self.head_dim
            self.qnorm = NORM2CLASS[norm_type](qnorm_dim, eps=norm_eps, **norm_kwargs)
            self.knorm = NORM2CLASS[norm_type](knorm_dim, eps=norm_eps, **norm_kwargs)

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)
        self.sdpa = ScaledDotProductAttention(dropout_p=dropout_p)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        bs, seqlen, _ = x.shape
        xq = self.wq(x)  # (bs, seqlen, n_local_heads * head_dim)
        xk = self.wk(x)  # (bs, seqlen, n_kv_heads * head_dim)
        xv = self.wv(x)  # (bs, seqlen, n_kv_heads * head_dim)

        # ==== QK Normalization if applicable
        if self.use_qknorm and self.qknorm_use_global:
            # (Option 1) Normalize over the full concatenated dimension first
            xq = self.qnorm(xq)  # no shape change
            xk = self.knorm(xk)  # no shape change

        # NOTE: using -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual local heads
        # from sizes of xq, xk, and xv as TP may have sharded them after the above linear ops
        xq = xq.view(bs, seqlen, -1, self.head_dim)  # (bs, seqlen, n_local_heads, head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)  # (bs, seqlen, n_kv_heads, head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)  # (bs, seqlen, n_kv_heads, head_dim)

        if self.use_qknorm and not self.qknorm_use_global:
            # (Option 2) Normalise each head independently after reshaping
            xq = self.qnorm(xq)  # no shape change
            xk = self.knorm(xk)  # no shape change

        # ==== Apply rotary embeddings if provided (supports NoPe)
        if self.use_rope:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)  # no shape change

        # ==== Attention
        # repeat k/v heads if n_kv_heads < n_heads
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_kv_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)  # (bs, seqlen, n_local_heads * head_dim or d_model)

        # ==== Final linear projection
        return self.wo(output)  # (bs, seqlen, n_local_heads * head_dim or d_model)

    def init_weights(self, init_std: float) -> None:
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.use_qknorm:
            self.qnorm.reset_parameters()
            self.knorm.reset_parameters()


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
        size_multiplier: float | None = None,
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.gated = gated

        # GLU variants scale down the intermediate_size by 2/3 to keep the number of parameters similar
        # to a standard MLP since the gating mechanism requires an additional linear layer
        if size_multiplier is not None:
            intermediate_size = int(size_multiplier * intermediate_size)
        self.intermediate_size = intermediate_size

        self.act_fn = ACT2FN[act_fn]()
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
        # NOTE: Creates norm modules conditionally based on config for clean model representation.
        # Static None/module assignments at init time are torch.compile friendly. The compiler
        # can optimize away unused conditional branches since module presence is determined once.
        self.parallel_layers = config.parallel_layers

        def create_norm(should_create: bool) -> nn.Module | None:
            if should_create:
                return NORM2CLASS[config.norm_type](config.d_model, eps=config.norm_eps, **config.norm_kwargs)

        # Attention Block
        self.attn_prenorm = create_norm(config.attn_prenorm)
        self.attn = Attention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,  # type: ignore -> the type checker is confused, this is dealt with in the config
            dropout_p=config.dropout_p,
            bias=config.attn_bias,
            head_dim=config.head_dim,
            use_rope=config.should_use_rope(layer_id),
            qknorm_kwargs={
                "norm_type": config.norm_type,
                "norm_eps": config.norm_eps,
                "use_global": config.qknorm_use_global,
                "kwargs": config.norm_kwargs,
            } if config.qknorm else None,
        )
        self.attn_postnorm = create_norm(config.attn_postnorm)
        
        # FeedForward Block
        self.ffw_prenorm = create_norm(config.ffw_prenorm)
        self.ffw = FeedForward(
            d_model=config.d_model,
            intermediate_size=config.intermediate_size,
            act_fn=config.act_fn,
            bias=config.ffw_bias,
            size_multiplier=config.size_multiplier,
            gated=config.gated,
        )
        self.ffw_postnorm = create_norm(config.ffw_postnorm)

        # Compute weight initialization standard deviation
        # 1. Per-layer depth scaling: deeper layers get smaller initialization
        # 2. Global depth scaling: all layers share the same initialization
        depth_factor = (layer_id + 1) if config.depth_init else config.n_layers
        self.weight_init_std = config.init_std / (2 * depth_factor) ** 0.5

    def _apply_attention(self, x: Tensor, freqs_cis: Tensor | None) -> Tensor:
        if self.attn_prenorm is not None:
            x = self.attn_prenorm(x)
        x = self.attn(x, freqs_cis)
        if self.attn_postnorm is not None:
            x = self.attn_postnorm(x)
        return x

    def _apply_ffw(self, x: Tensor) -> Tensor:
        if self.ffw_prenorm is not None:
            x = self.ffw_prenorm(x)
        x = self.ffw(x)
        if self.ffw_postnorm is not None:
            x = self.ffw_postnorm(x)
        return x

    def forward(self, x: Tensor, freqs_cis: Tensor | None = None) -> Tensor:
        # Parallel layers
        if self.parallel_layers:
            attn_out = self._apply_attention(x, freqs_cis)
            ffw_out = self._apply_ffw(x)
            return x + attn_out + ffw_out

        # Sequential layers
        x = x + self._apply_attention(x, freqs_cis)
        return x + self._apply_ffw(x)

    def init_weights(self) -> None:
        for norm in (self.attn_prenorm, self.attn_postnorm, self.ffw_prenorm, self.ffw_postnorm):
            if norm:
                norm.reset_parameters()  # type: ignore -> the type checker is confused
        self.attn.init_weights(self.weight_init_std)
        self.ffw.init_weights(self.weight_init_std)


# ==========================================================================================
# Transformer Model
# ==========================================================================================
class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # ==== RoPE frequency tensor
        # TODO persistent should be set to false, since this buffer can be recomputed.
        # however, we set it to true for 2 reasons.  (1) due to pytorch/pytorch#123411,
        # compile or pipeline-tracer will not correctly handle non-persistent buffers,
        # so we need to fix that.  (2) if we initialize pipeline-parallel models from
        # a seed checkpoint rather than calling init_weights, we need freqs_cis to be
        # initialized by the checkpoint, or we need to add a separate initializer for
        # just the non-persistent buffers that is called after loading checkpoints.
        self.register_buffer("freqs_cis", self._precompute_freqs_cis(), persistent=True)

        # ==== Embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # ==== Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(layer_id, config) for layer_id in range(config.n_layers)])

        # ==== Output layer
        self.norm = NORM2CLASS[config.norm_type](config.d_model, eps=config.norm_eps, **config.norm_kwargs)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        self.init_weights()

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
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
        for layer in self.layers:
            layer.init_weights()  # type: ignore -> the type checker is confused
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.config.d_model**-0.5
        cutoff_factor = 3
        if self.lm_head is not None:
            nn.init.trunc_normal_(
                self.lm_head.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_freqs_cis(self) -> Tensor:
        return precompute_freqs_cis(
            self.config.head_dim,  # type: ignore -> the type checker is confused, this is dealt with in the config
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.config.max_seqlen,
            self.config.rope_theta,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(input_ids) if self.tok_embeddings else input_ids

        for layer in self.layers:
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.lm_head(h) if self.lm_head else h
        return output
