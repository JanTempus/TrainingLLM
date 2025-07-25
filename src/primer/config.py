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
class PrimerConfig:
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
class OptimCofig(PrimerConfig):
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


# ==================================================================================================
# Data Configurations
# ==================================================================================================
@dataclass
class DataloaderConfig(PrimerConfig):
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


# ==================================================================================================
# Model Configurations
# ==================================================================================================
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
       
NORM2CLASS = {"layernorm": torch.nn.LayerNorm, "rmsnorm": torch.nn.RMSNorm}
NormType = Literal["layernorm", "rmsnorm"]


@dataclass
class ModelConfig(PrimerConfig):
    """Configuration for the language model.

    This configuration is used to define the architecture and hyperparameters of the model.

    General parameters:
        d_model (int): Dimension of the model.
        n_layers (int): Number of layers in the model.
        max_seqlen (int): Maximum sequence length the model can handle.
        vocab_size (int): Size of the vocabulary.
        eos_id (int): End of sequence token ID.
        tie_embeddings (bool): Whether to tie the input and output embeddings.
        parallel_layers (bool): Whether to use parallel layers.
        multiple_of (int): Ensure certain dimensions ("d_model", "vocab_size", "intermediate_size", "head_dim")
            are a multiple of this value. Especially important for performance and compatibility with certain kernels
            like Liger. Default is 128, which is a common choice for many transformer models.
        
    Attention parameters: 
        n_heads (int): Number of attention heads.
        n_kv_heads (int | None): Number of key-value heads, if None, it defaults to `n_heads`.
        head_dim (int | None): Dimension of each attention head. If None, it is calculated as `d_model // n_heads`.
        dropout_p (float): Dropout probability for attention layers.
        attn_bias (bool): Whether to use bias in attention layers.
        attn_prenorm (bool): Whether to apply prenormalization in attention layers.
        attn_postnorm (bool): Whether to apply postnormalization in attention layers.
        qknorm (bool): Whether to use QK normalization.
        qknorm_use_global (bool): Whether to use global normalization for QK.
    
    FeedForward parameters:
        intermediate_size (int): Size of the intermediate layer in feedforward networks.
        size_multiplier (float | None): Multiplier for the intermediate size.
        act_fn (ActFnType): Activation function type.
        gated (bool): Whether to use gated activation in feedforward networks.
        ffw_bias (bool): Whether to use bias in feedforward networks.
        ffw_prenorm (bool): Whether to apply prenormalization in feedforward networks.
        ffw_postnorm (bool): Whether to apply postnormalization in feedforward networks.
    
    Norms parameters:
        norm_type (NormType): Type of normalization to use.
        norm_eps (float): Epsilon value for normalization layers.
        norm_kwargs (dict): Additional keyword arguments for normalization layers.
    
    RoPE parameters:
        rope_theta (float): Theta value for RoPE (Rotary Position Embedding).
        rope_pattern (str): Pattern for applying RoPE, e.g., "all", "none", "3:", ":3", "1,3,5", "1:2:10".
    
    Initialization parameters:
        depth_init (bool): Whether to use depth initialization.
        init_std (float): Standard deviation for weight initialization.
    """

    d_model: int = 768
    n_layers: int = 6
    max_seqlen: int = 512
    vocab_size: int = 30522
    eos_id: int = 0  # End of sequence token ID
    tie_embeddings: bool = False
    parallel_layers: bool = False

    # This flag is so important. I fixed a weird bug in the past where
    # the loss was almost flat because I had 128001 vocab_size. I am now
    # defining it globally as use it everywhere it's needed.
    multiple_of: int = 128

    # Attention configuration
    n_heads: int = 12
    n_kv_heads: int | None = 3
    head_dim: int | None = None
    dropout_p: float = 0.0
    attn_bias: bool = False
    attn_prenorm: bool = True
    attn_postnorm: bool = False
    qknorm: bool = False
    qknorm_use_global: bool = True

    # FeedForward configuration
    intermediate_size: int = 3072
    size_multiplier: float | None = None
    act_fn: ActFnType = "silu"
    gated: bool = True
    ffw_bias: bool = True
    ffw_prenorm: bool = True
    ffw_postnorm: bool = False

    # Norms configuration
    norm_type: NormType = "rmsnorm"
    norm_eps: float = 1e-5
    norm_kwargs: dict = field(default_factory=dict)

    # Rope configuration
    rope_theta: float = 10000.0
    rope_pattern: str = "all"  # e.g. "all", "none", "3:", ":3", "1,3,5", "1:2:10"
    
    # Initialisation
    depth_init: bool = False
    init_std: float = 0.02

    def __post_init__(self) -> None:
        """Validate the model configuration parameters.

        NOTE: Data type validation will be ensured by OmegaConf or similar
        Here we only check for basic constraints and sanity checks.
        """
        assert self.d_model > 0, "d_model must be a positive integer"
        assert self.n_layers > 0, "n_layers must be a positive integer"
        assert self.max_seqlen > 0, "max_seqlen must be a positive integer"
        assert self.vocab_size > 0, "vocab_size must be a positive integer"
        assert self.eos_id >= 0, "eos_id must be a non-negative integer"
        assert self.multiple_of > 0, "multiple_of must be a positive integer"

        # ==== Attention configuration checks ====
        assert self.n_heads > 0, "n_heads must be a positive integer"
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
            logger.info("n_kv_heads is None, setting it to n_heads, so no GQA will be used.")
        assert self.n_kv_heads > 0, "n_kv_heads must be a positive integer"
        assert 0 <= self.dropout_p <= 1, "dropout_p must be between 0 and 1"
        if self.head_dim is None:
            assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads, or pass head_dim"
            self.head_dim = self.d_model // self.n_heads
            logger.info(f"head_dim is None, setting it to {self.head_dim} (d_model // n_heads)")
        assert self.head_dim > 0, "head_dim must be a positive integer"

        # ==== FeedForward configuration checks ====
        assert self.intermediate_size > 0, "intermediate_size must be a positive integer"
        if self.size_multiplier is not None:
            assert self.size_multiplier > 0, "size_multiplier must be a positive float"
        self.act_fn = self.act_fn.lower()  # type: ignore
        assert self.act_fn in ACT2FN, f"{self.act_fn} unsupported. Supported functions: {list(ACT2FN.keys())}"

        # ==== Norm configuration checks ====
        self.norm_type = self.norm_type.lower()  # type: ignore
        assert self.norm_type in NORM2CLASS, f"Unsupported {self.norm_type}, supported norm types: {list(NORM2CLASS.keys())}"
        assert self.norm_eps > 0, "eps must be a positive float"
        if not (self.attn_prenorm or self.attn_postnorm):
            logger.warning("Neither attn_prenorm nor attn_postnorm is True. This may lead to unexpected behavior.")
        if not (self.ffw_prenorm or self.ffw_postnorm):
            logger.warning("Neither ffw_prenorm nor ffw_postnorm is True. This may lead to unexpected behavior.")
        
        # ==== RoPE configuration checks ====
        assert self.rope_theta > 0, "rope_theta must be a positive float"
        if self.rope_pattern not in ["all", "none"] and not ("," in self.rope_pattern or ":" in self.rope_pattern):
            # Check if it's a valid slice or comma-separated indices
            try:
                if "," in self.rope_pattern:
                    _ = [int(x) for x in self.rope_pattern.split(",")]
                elif ":" in self.rope_pattern:
                    parts = [int(x) if x else None for x in self.rope_pattern.split(":")]
                    slice(*parts)
            except ValueError as e:
                raise ValueError(f"Invalid RoPE pattern: '{self.rope_pattern}'") from e
            
        self._rope_mask = self._parse_rope_pattern()
            
        # ==== General checks ====
        # Ensure these values are multiple of `multiple_of`
        for attr_name in ["d_model", "vocab_size", "intermediate_size", "head_dim"]:
            self._make_multiple_of(attr_name)

    def _make_multiple_of(self, attr_name: str) -> None:
        """Rounds up an attribute to the nearest multiple of `multiple_of`."""
        original_value = getattr(self, attr_name)
        if original_value is None or original_value % self.multiple_of == 0:
            return

        new_value = self.multiple_of * ((original_value + self.multiple_of - 1) // self.multiple_of)
        logger.warning(f"`{attr_name}` rounded from {original_value} to {new_value} (multiple of {self.multiple_of})")
        setattr(self, attr_name, new_value)

    def _parse_rope_pattern(self) -> list[bool]:
        """Parses the pattern string into a boolean mask of length `n_layers`."""
        if self.rope_pattern == "all":
            logger.info("Using RoPE for all layers")
            mask = [True] * self.n_layers
        elif self.rope_pattern == "none":
            logger.info("Not using RoPE for any layer")
            mask = [False] * self.n_layers
        elif "," in self.rope_pattern:
            indices = {int(x) for x in self.rope_pattern.split(",")}
            logger.info(f"Using RoPE for layers: {indices}")
            mask = [i in indices for i in range(self.n_layers)]
        elif ":" in self.rope_pattern:
            # Supports full slice syntax like "start:stop:step"
            parts = [int(x) if x else None for x in self.rope_pattern.split(":")]
            indices = set(range(*slice(*parts).indices(self.n_layers)))
            logger.info(f"Using RoPE for layers: {sorted(indices)}")
            mask = [i in indices for i in range(self.n_layers)]
        else:
            raise ValueError(f"Invalid RoPE pattern: '{self.rope_pattern}'")

        return mask

    def should_use_rope(self, layer_id: int) -> bool:
        """Determine whether to use RoPE at the given layer."""
        return self._rope_mask[layer_id]