from torch.optim import AdamW
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

TYPE_TO_OPTIMIZER_CLASS = {"adamw": AdamW}
TYPE_TO_SCHEDULER_FUNCTION = TYPE_TO_SCHEDULER_FUNCTION.copy()  # for clarity, prevents ruff removing the unused import
