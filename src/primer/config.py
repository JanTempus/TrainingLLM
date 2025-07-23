import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from os import cpu_count
from typing import Any, Literal

import torch

from primer.optim import TYPE_TO_OPTIMIZER_CLASS, TYPE_TO_SCHEDULER_FUNCTION
from primer.utilities import get_logger

logger = get_logger("config")


@dataclass
class DictConfig:
    """Dataclass which is subscriptable like a dict"""

    def to_dict(self) -> dict[str, Any]:
        out = copy.deepcopy(self.__dict__)
        return out

    def __getitem__(self, k: str) -> Any:
        return self.__dict__[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)


@dataclass
class OptimCofig(DictConfig):
    # Optimizer config
    optim_name: str
    lr: float
    weight_decay: float = 0.0
    weight_decay_embedding: bool = False  # If True, apply weight decay to embedding layers
    set_grad_to_none: bool = True  # If True, set gradients to None instead of zeroing them out
    optim_kwargs: dict = field(default_factory=dict)

    # Scheduler config
    scheduler_name: str | None = None
    num_warmup_steps: int | None = None
    scheduler_kwargs: dict = field(default_factory=dict)

    # Gradient accumulation config
    grad_acc_schedule: dict | None = None
    zloss_factor: float | None = None  # lambda for zloss, if used

    def __post_init__(self) -> None:
        assert self.optim_name in TYPE_TO_OPTIMIZER_CLASS
        if self.scheduler_name is not None:
            assert self.scheduler_name in TYPE_TO_SCHEDULER_FUNCTION


@dataclass
class DataloaderConfig(DictConfig):
    batch_size: int | None = None
    eval_batch_size: int | None = None
    shuffle_seed: int | None = None
    intra_doc_causal_mask: bool = False

    # kwargs
    num_workers: int | None = cpu_count()
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = False
    multiprocessing_context: str | None = None
    prefetch_factor: int | None = None

    def get_dataloader_kwargs(self) -> dict:
        # Remove batch_size, eval_batch_size, and shuffle_seed from the dataloader configuration
        kwargs = {
            k: v
            for k, v in self.to_dict().items()
            if k not in ["batch_size", "eval_batch_size", "shuffle_seed", "intra_doc_causal_mask"]
        }
        return kwargs


@dataclass
class AttentionConfig(DictConfig):
    """Configuration for the attention module."""

    n_heads: int = 12
    n_kv_heads: int | None = 3
    dropout_p: float = 0.0
    bias: bool = False
    head_dim: int | None = None

    def __post_init__(self) -> None:
        # Ensure n_heads is a positive integer
        assert self.n_heads > 0, "n_heads must be a positive integer"

        # If n_kv_heads is None, set it to n_heads (no GQA in this case)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # Ensure n_kv_heads is a positive integer
        assert self.n_kv_heads > 0, "n_kv_heads must be a positive integer"

        # Ensure dropout_p is a float between 0 and 1
        assert 0 <= self.dropout_p <= 1, "dropout_p must be between 0 and 1"

        # Ensure head_dim is set correctly
        if self.head_dim is None:
            # If head_dim is not provided, calculate it based on n_heads
            assert self.n_heads > 0, "n_heads must be greater than 0 to calculate head_dim"


# ==== FeedForward Configs ====
ACT2FN = {
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,
    "swish": torch.nn.SiLU,
    "mish": torch.nn.Mish,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}
ActFnType = Literal["relu", "gelu", "silu", "swish", "mish", "tanh", "sigmoid"]


@dataclass
class FeedForwardConfig(DictConfig):
    """Configuration for the feedforward module."""

    intermediate_size: int = 3072
    size_multiplier: float | None = None
    act_fn: ActFnType = "silu"
    bias: bool = True
    gated: bool = True

    def __post_init__(self) -> None:
        self.act_fn = self.act_fn.lower()  # type: ignore
        assert self.act_fn in ACT2FN, f"{self.act_fn} unsupported. Supported functions: {list(ACT2FN.keys())}"
        assert self.intermediate_size > 0, "intermediate_size must be a positive integer"
        if self.size_multiplier is not None:
            assert self.size_multiplier > 0, "size_multiplier must be a positive float"


# ==== Normalisation Configs ====
NORM2CLASS = {"layernorm": torch.nn.LayerNorm, "rmsnorm": torch.nn.RMSNorm}

NormType = Literal["layernorm", "rmsnorm"]


@dataclass
class QKNormConfig(DictConfig):
    """Configuration for QK normalization in attention layers."""

    type: NormType
    eps: float
    kwargs: dict = field(default_factory=dict)
    only_head_dim: bool = True

    def __post_init__(self) -> None:
        self.type = self.type.lower()  # type: ignore
        assert self.type in NORM2CLASS, f"Unsupported normalization type: {self.type}"
        assert self.eps > 0, "norm_eps must be a positive float"


@dataclass
class NormConfig(DictConfig):
    """Configuration for normalization layers."""

    type: NormType = "rmsnorm"
    eps: float = 1e-5
    kwargs: dict = field(default_factory=dict)
    attn_pre: bool = True
    attn_post: bool = False
    ffw_pre: bool = True
    ffw_post: bool = False
    attn_qk: bool = False
    attn_qk_only_head_dim: bool = True

    def __post_init__(self) -> None:
        self.type = self.type.lower()  # type: ignore
        assert self.type in NORM2CLASS, f"Unsupported {self.type}, supported norm types: {list(NORM2CLASS.keys())}"
        assert self.eps > 0, "eps must be a positive float"

        if not (self.attn_pre or self.attn_post):
            logger.warning("Neither attn_pre nor attn_post is True. This may lead to unexpected behavior.")
        if not (self.ffw_pre or self.ffw_post):
            logger.warning("Neither ffw_pre nor ffw_post is True. This may lead to unexpected behavior.")

    def get_qk_norm_config(self) -> QKNormConfig | None:
        """Get normalization kwargs for attention QK normalization."""
        if self.attn_qk:
            return QKNormConfig(
                type=self.type, eps=self.eps, kwargs=self.kwargs, only_head_dim=self.attn_qk_only_head_dim
            )


@dataclass
class RopeConfig:
    theta: float = 10000.0
    pattern: str = "all"  # e.g. "all", "none", "3:", ":3", "1,3,5", "1:2:10"
    _num_layers: int | None = None  # Number of layers, used for validation

    def __post_init__(self) -> None:
        assert self.theta > 0, "theta must be a positive float"

        # Validate the pattern string
        if self.pattern not in ["all", "none"] and not ("," in self.pattern or ":" in self.pattern):
            # Check if it's a valid slice or comma-separated indices
            try:
                if "," in self.pattern:
                    _ = [int(x) for x in self.pattern.split(",")]
                elif ":" in self.pattern:
                    parts = [int(x) if x else None for x in self.pattern.split(":")]
                    slice(*parts)
            except ValueError as e:
                raise ValueError(f"Invalid RoPE pattern: '{self.pattern}'") from e

    def _parse_pattern(self) -> list[bool]:
        """Parses the pattern string into a boolean mask of length `num_layers`."""
        assert self._num_layers
        if self.pattern == "all":
            return [True] * self._num_layers
        elif self.pattern == "none":
            return [False] * self._num_layers
        elif "," in self.pattern:
            indices = {int(x) for x in self.pattern.split(",")}
            return [i in indices for i in range(self._num_layers)]
        elif ":" in self.pattern:
            # Supports full slice syntax like "start:stop:step"
            parts = [int(x) if x else None for x in self.pattern.split(":")]
            indices = set(range(*slice(*parts).indices(self._num_layers)))
            return [i in indices for i in range(self._num_layers)]
        else:
            raise ValueError(f"Invalid RoPE pattern: '{self.pattern}'")

    def should_use_rope(self, layer_id: int) -> bool:
        """Determine whether to use RoPE at the given layer."""
        assert self._num_layers
        mask = self._parse_pattern()
        return mask[layer_id]


@dataclass
class ModelConfig(DictConfig):
    """Configuration for the language model.

    Attributes:
        d_model (int): Dimension of the model.
        n_layers (int): Number of layers in the model.
        max_seq_len (int): Maximum number of position embeddings.
        vocab_size (int): Size of the vocabulary.
        n_heads (int): Number of attention heads.
        n_kv_heads (int | None): Number of key-value heads. If None, it will be set to n_heads.
        att_bias (bool): Whether to use bias in attention layers.
        init_std (float | dict): Standard deviation for weight initialization.
    """

    d_model: int = 768
    n_layers: int = 6
    max_seq_len: int = 512
    vocab_size: int = 30522
    eos_id: int = 0  # End of sequence token ID
    tie_embeddings: bool = False

    # Layer configuration
    norm: NormConfig = field(default_factory=NormConfig)
    rope: RopeConfig = field(default_factory=RopeConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ffw: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    parallel_layers: bool = False

    # Initialisation
    depth_init: bool = False
    init_std: float = 0.02

    # This flag is so important. I fixed a weird bug in the past where
    # the loss was almost flat because I had 128001 vocab_size. I am now
    # defining it globally as use it everywhere it's needed.
    multiple_of: int = 128

    def __post_init__(self) -> None:
        # Convert dicts to dataclasses, they validate themselves
        if isinstance(self.norm, dict):
            self.norm = NormConfig(**self.norm)
        if isinstance(self.rope, dict):
            self.rope = RopeConfig(**self.rope)
        if isinstance(self.attention, dict):
            self.attention = AttentionConfig(**self.attention)
        if isinstance(self.ffw, dict):
            self.ffw = FeedForwardConfig(**self.ffw)

        # Validate the model configuration
        self._validate()

        # Ensure these aremultiple of `multiple_of`
        for attr_name in ["d_model", "vocab_size", "ffw.intermediate_size", "attention.head_dim"]:
            self._make_multiple_of(attr_name)

        # Add layer info for RoPE checks
        self.rope._num_layers = self.n_layers

    def _validate(self) -> None:
        # NOTE: Data type validation will be ensured by OmegaConf or similar
        # Here we only check for basic constraints
        assert self.d_model > 0, "d_model must be a positive integer"
        assert self.n_layers > 0, "n_layers must be a positive integer"
        assert self.max_seq_len > 0, "max_seq_len must be a positive integer"
        assert self.vocab_size > 0, "vocab_size must be a positive integer"

        # Ensure d_model is divisible by n_heads
        if not self.attention.head_dim:
            assert self.d_model % self.attention.n_heads == 0, "d_model must be divisible by n_heads"

    def _make_multiple_of(self, attr_name: str) -> None:
        """Rounds up an attribute to the nearest multiple of `multiple_of`."""

        # This loop handles nested attributes like "ffw.intermediate_size"
        obj = self
        parts = attr_name.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)

        final_attr = parts[-1]
        original_value = getattr(obj, final_attr)

        if original_value is None or original_value % self.multiple_of == 0:
            return

        new_value = self.multiple_of * ((original_value + self.multiple_of - 1) // self.multiple_of)
        logger.warning(f"`{attr_name}` rounded from {original_value} to {new_value} (multiple of {self.multiple_of})")
        setattr(obj, final_attr, new_value)
