import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from os import cpu_count
from typing import Any, Literal, get_args

from primer.optim import TYPE_TO_OPTIMIZER_CLASS, TYPE_TO_SCHEDULER_FUNCTION
from primer.utilities import get_logger

ActFnType = Literal["relu", "gelu", "silu", "swish", "mish", "tanh", "sigmoid"]
NormType = Literal["layernorm", "rmsnorm"]

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


@dataclass
class FeedForwardConfig(DictConfig):
    """Configuration for the feedforward module."""

    intermediate_size: int = 3072
    multiple_of: int = 64
    size_multiplier: float | None = None
    act_fn: ActFnType = "silu"
    bias: bool = True
    gated: bool = True

    def __post_init__(self) -> None:
        assert self.intermediate_size > 0, "intermediate_size must be a positive integer"
        assert self.multiple_of > 0, "multiple_of must be a positive integer"
        if self.size_multiplier is not None:
            assert self.size_multiplier > 0, "size_multiplier must be a positive float"
        assert self.act_fn in get_args(ActFnType), f"Unsupported activation function: {self.act_fn}"


@dataclass
class QKNormConfig(DictConfig):
    """Configuration for QK normalization in attention layers."""

    type: NormType
    eps: float
    kwargs: dict = field(default_factory=dict)
    only_head_dim: bool = True

    def __post_init__(self) -> None:
        assert self.type in get_args(NormType), f"Unsupported normalization type: {self.type}"
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
        assert self.type in get_args(NormType), f"Unsupported normalization type: {self.type}"
        assert self.eps > 0, "eps must be a positive float"

        if not (self.attn_pre or self.attn_post):
            logger.warning("Neither attn_pre nor attn_post is True. This may lead to unexpected behavior.")
        if not (self.ffw_pre or self.ffw_post):
            logger.warning("Neither ffw_pre nor ffw_post is True. This may lead to unexpected behavior.")

    def get_qk_norm_config(self) -> QKNormConfig | None:
        """Get normalization kwargs for attention QK normalization."""
        if self.attn_qk:
            return QKNormConfig(
                type=self.type,
                eps=self.eps,
                kwargs=self.kwargs,
                only_head_dim=self.attn_qk_only_head_dim,
            )

@dataclass
class NopeConfig:
    strategy: Literal["all", "first_n", "every_n", "custom", "none"] = "all"
    n: int | None = None         
    custom_pattern: list[bool] | None = None

    def __post_init__(self) -> None:
        assert self.strategy in ["all", "none", "first_n", "every_n", "skip_every_n", "custom"], f"Unknown NoPE strategy: {self.strategy}"
        assert not (self.strategy == "custom" and (self.custom_pattern is None or not isinstance(self.custom_pattern, list))), \
            "custom_pattern must be a list of booleans when strategy is 'custom'"
        
        if self.strategy in ["first_n", "every_n", "skip_every_n"]:
            assert self.n is not None and self.n >= 0, "n must be specified for first_n, every_n, or skip_every_n strategies"

    def should_use_rope(self, layer_id) -> bool:
        """Determine whether this layer should use RoPE based on NoPE configuration."""        
        if self.strategy == "all":
            return True
        elif self.strategy == "none":
            return False
        elif self.strategy == "first_n":
            # Use RoPE only for the first n layers
            assert self.n is not None  # make type checker happy
            return layer_id < self.n
        elif self.strategy == "every_n":
            # Use RoPE every n layers (0, n, 2n, 3n, ...)
            assert self.n is not None  # make type checker happy
            return self.n > 0 and (layer_id % self.n) == 0
        elif self.strategy == "skip_every_n":
            # Skip RoPE every n layers (disable at n-1, 2n-1, 3n-1, ...)
            assert self.n is not None  # make type checker happy
            return self.n == 0 or (layer_id % self.n) != (self.n - 1)
        elif self.strategy == "custom":
            # Use custom boolean pattern (cycles if layer_id exceeds pattern length)
            if self.custom_pattern:
                return self.custom_pattern[layer_id % len(self.custom_pattern)]
            return True
        raise ValueError(f"Unknown NoPE strategy: {self.strategy}")


@dataclass
class ModelConfig(DictConfig):
    """Configuration for the language model.

    Attributes:
        d_model (int): Dimension of the model.
        n_layers (int): Number of layers in the model.
        max_position_embeddings (int): Maximum number of position embeddings.
        vocab_size (int): Size of the vocabulary.
        n_heads (int): Number of attention heads.
        n_kv_heads (int | None): Number of key-value heads. If None, it will be set to n_heads.
        att_bias (bool): Whether to use bias in attention layers.
        init_std (float | dict): Standard deviation for weight initialization.
    """

    d_model: int = 768
    n_layers: int = 12
    max_position_embeddings: int = 512
    vocab_size: int = 30522
    eos_id: int = 0  # End of sequence token ID

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ffw: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    parallel_layers: bool = False

    norm: NormConfig = field(default_factory=NormConfig)

    depth_init: bool = False
    init_std: float = 0.02

    nope: NopeConfig = field(default_factory=NopeConfig)

    def __post_init__(self) -> None:
        # NOTE: Data type validation will be implemented later using OmegaConf or similar
        # Here we only check for basic constraints
        assert self.d_model > 0, "d_model must be a positive integer"
        assert self.n_layers > 0, "n_layers must be a positive integer"
        assert self.max_position_embeddings > 0, "max_position_embeddings must be a positive integer"
        assert self.vocab_size > 0, "vocab_size must be a positive integer"

        # Ensure d_model is divisible by n_heads
        assert self.d_model % self.attention.n_heads == 0, "d_model must be divisible by n_heads"
