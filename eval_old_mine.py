import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from primer.model import load_hf_from_pl
from tqdm import tqdm
from pathlib import Path
from datasets import load_from_disk, DatasetDict, load_dataset,Dataset
from transformers import AutoTokenizer
import numpy as np


# --- Config ---
dataset_url="pietrolesci/finewebedu-20B"
tokenizer_path = "/local/home/jtempus/tokenisation_lp/bpe_tokenizer/bpe_tokenizers/bpe_8192_finewebedu"
batch_size = 10000
SEQ_LEN = 2049    
num_proc = 16            # parallel workers for Dataset.map
val_frac = 0.1

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


@torch.no_grad()
def cross_entropy_per_byte_from_dataset(
    model_path: str | Path,
    tokenizer_path: str | Path,
    dataset,
    text_column: str = "text",
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Compute cross-entropy per byte (nats/byte) on a pre-tokenized Hugging Face dataset.

    Args:
        model_path: Path to PyTorch Lightning checkpoint (.ckpt)
        tokenizer_path: Path to HuggingFace tokenizer directory
        dataset_path: Path to preprocessed HF dataset (load_from_disk)
        text_column: Column in dataset containing raw text (for byte count)
        batch_size: Batch size for evaluation
        device: 'cuda' or 'cpu'

    Returns:
        Cross entropy per byte (in nats/byte)
    """
    # --- Load model + tokenizer ---
    print(f"Loading model from {model_path}")
    model = load_hf_from_pl(model_path).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")

    # --- Load dataset ---
 
    ds = dataset

    # Ensure columns exist
    required_cols = {"input_ids", text_column}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # --- DataLoader ---
    def collate_fn(batch):
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        texts = [b[text_column] for b in batch]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "texts": texts}

    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

    # --- Accumulators ---
    total_loss = 0.0
    total_bytes = 0

    for batch in tqdm(dl, desc="Evaluating", dynamic_ncols=True):
        input_ids = batch["input_ids"].to(device)
        att_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=att_mask, labels=labels)
        loss = outputs.loss * input_ids.numel()  # HF loss is avg per token
        total_loss += loss.item()

        total_bytes += sum(len(txt.encode("utf-8")) for txt in batch["texts"])

    ce_per_byte = total_loss / total_bytes
    bits_per_byte = ce_per_byte / torch.log(torch.tensor(2.0))

    print(f"Cross-entropy per byte: {ce_per_byte:.6f} nats/byte")
    print(f"Bits per byte: {bits_per_byte:.6f}")
    return ce_per_byte

def process(batch: dict) -> dict:
    tokens = tokenizer(batch["text"])["input_ids"]
    return {"input_ids": tokens,
        "len": [len(x) for x in tokens]
    }

def concat_docs(batch):
    all_tokens = []
    for ids in batch["input_ids"]:
        all_tokens.extend(ids + [EOS_TOKEN_ID])
    # import ipdb; ipdb.set_trace()
    return {"input_ids": all_tokens}

def chunk_docs(batch):
    all_tokens = []
    chunk = []
    for ids in batch["input_ids"]:
        chunk += [ids]
        if len(chunk) == SEQ_LEN:
            all_tokens.extend([chunk])
            chunk = []
    return {"input_ids": all_tokens}


def pack_data(dataset,tokenizer_path):    

    dataset = dataset.shuffle(seed=42)
    dataset=dataset.flatten_indices()
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    
    concatenated = dataset.map(concat_docs, 
                            batched=True, 
                            batch_size=batch_size,
                            remove_columns=dataset.column_names,
                            desc="Concatenating Things",
                            num_proc=num_proc
                            )

    print("Finished Concatinating")
    chunks = concatenated.map(chunk_docs, 
                            batched=True, 
                            batch_size=batch_size * SEQ_LEN,
                            remove_columns=concatenated.column_names,
                            desc="Chunking Things",
                            num_proc=num_proc
                            )
    packed = chunks.shuffle(seed=42)
    # Step 5: Save packed datasetdatasets-cli cache clear
    return packed



if __name__ == "__main__":
  
    dataset = load_dataset(dataset_url)  # let HF do the caching for you
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]  # flatten

    dataset = dataset.select_columns(["id", "text"])  # keep only the required columns


    # --- Train/val/test split ---
    # Read the docs: https://huggingface.co/docs/datasets/en/process#split
    # NOTE: here we want a DatasetDict because it's more convenient
    ds_dict = dataset.train_test_split(test_size=2_000_000, seed=42)

    # NOTE: However, we want to rename "test" to "validation"
    
    dataset_raw=ds_dict.pop("test")

    # --- Tokenize ---
    # NOTE: read the docs https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.DatasetDict.map
    dataset = dataset_raw.map(
        process,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Tokenizing"
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    EOS_TOKEN_ID = tokenizer.eos_token_id

    # dataset_packed=pack_data(dataset,tokenizer_path)


    ce = cross_entropy_per_byte_from_dataset(
        model_path="/local/home/jtempus/TrainingLLM/outputs/training/run_2025-10-22_bpe_8k_57M/.checkpoints/last.ckpt",
        tokenizer_path=tokenizer_path,
        dataset=dataset,
        text_column="text",
        batch_size=64,
    )

