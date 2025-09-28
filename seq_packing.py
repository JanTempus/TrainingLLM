from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
import numpy as np
import os
from tqdm import tqdm

def pack_and_save_dataset(
    dataset_path: str,
    tokenizer_path: str,
    save_path: str,
    seq_len: int = 2049,
    batch_size: int = 1000,
):
    """
    Pack a Hugging Face dataset into fixed-length sequences and save to disk.

    Args:
        dataset_path (str): Path to the tokenized dataset on disk (must contain "input_ids").
        tokenizer_path (str): Path to the tokenizer (for EOS token).
        save_path (str): Where to save the packed dataset.
        seq_len (int): Length of each packed sequence (default 2049).
        batch_size (int): How many documents to process at once in memory.
    """
    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    if EOS_TOKEN_ID is None:
        raise ValueError("Tokenizer must define eos_token_id.")
    
    dataset = load_from_disk(dataset_path)
    
    # Collect chunks incrementally
    buffer = []
    packed_chunks = []

    def flush_chunks(tokens):
        """Split a list of tokens into fixed-length chunks."""
        n_full_chunks = len(tokens) // seq_len
        return [
            tokens[i * seq_len : (i + 1) * seq_len]
            for i in range(n_full_chunks)
        ], tokens[n_full_chunks * seq_len :]

    for i in tqdm(range(0, len(dataset), batch_size), desc="Packing dataset"):
        batch = dataset[i : i + batch_size]["input_ids"]

        for ids in batch:
            buffer.extend(ids + [EOS_TOKEN_ID])

            if len(buffer) >= seq_len:
                full_chunks, buffer = flush_chunks(buffer)
                packed_chunks.extend(full_chunks)

                # To keep RAM usage small, periodically flush to disk
                if len(packed_chunks) > 100_000:  
                    partial = Dataset.from_dict({"input_ids": packed_chunks})
                    if os.path.exists(save_path):
                        partial.save_to_disk(save_path, append=True)
                    else:
                        partial.save_to_disk(save_path)
                    packed_chunks = []

    # Final flush
    if buffer:
        full_chunks, _ = flush_chunks(buffer)
        packed_chunks.extend(full_chunks)

    if packed_chunks:
        partial = Dataset.from_dict({"input_ids": packed_chunks})
        if os.path.exists(save_path):
            partial.save_to_disk(save_path, append=True)
        else:
            partial.save_to_disk(save_path)

    print(f"✅ Packed dataset saved at: {save_path}")


pack_and_save_dataset(
    dataset_path="tokenized_dataset_uv/train",
    tokenizer_path="tokenizers_lp_uv/lp_32768_finewebedu_data",
    save_path="dataset_packed/train",
    seq_len=2049,
    batch_size=1000
)