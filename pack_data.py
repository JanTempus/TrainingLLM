from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
DATASET_PATH = "tokenized_dataset_uv/train"   # where your tokenized data is stored
PACKED_PATH = "dataset_packed/train"            # where you want to save the packed version
tokenizer_path = "tokenizers_lp_uv/lp_32768_finewebedu_data"
SEQ_LEN = 2049      
NUM_PROC = 8                              # sequence length
batch_size=1000

# --- Load tokenizer (to get EOS token id) ---
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
EOS_TOKEN_ID = tokenizer.eos_token_id
print(f"EOS token id = {EOS_TOKEN_ID}")

# --- Load dataset ---


# Assumes each example has {"input_ids": [list of token IDs]}
# Step 1: Shuffle documents


def concat_docs(batch):
    all_tokens = []
    for ids in batch["input_ids"]:
        all_tokens.extend(ids + [EOS_TOKEN_ID])
    return {"all_tokens": all_tokens}

# Step 3: Chunk into fixed-size sequences


# Example usage

if __name__ == "__main__":
    dataset = load_from_disk(DATASET_PATH).select(range(1000000))


    dataset = dataset.shuffle(seed=42)

    concatenated = dataset.map(concat_docs, 
                            batched=True, 
                            batch_size=batch_size,
                            remove_columns=dataset.column_names,
                            desc="Concatenating Things",
                            num_proc=NUM_PROC
                            )
    

    print("Finished Concatinating")
    
    arr=concatenated["all_tokens"]
    n = len(arr)
    chunks = []
    for i in tqdm(range(0, n, SEQ_LEN), desc="chunking", total=(n + SEQ_LEN - 1)//SEQ_LEN):
        chunks.append(arr[i:i+SEQ_LEN])  # Hugging Face wants lists
    # Wrap into Hugging Face Dataset

    chunked= Dataset.from_dict({"input_ids": chunks})

    # Step 4: Shuffle final sequences
    packed = chunked.shuffle(seed=42)

    # Step 5: Save packed dataset
    packed.save_to_disk(PACKED_PATH,max_shard_size="3GB")

    print("Packed dataset saved at:", PACKED_PATH)