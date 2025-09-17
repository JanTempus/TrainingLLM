from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np

# --- CONFIG ---
DATASET_PATH = "tokenized_dataset_uv/train"   # where your tokenized data is stored
PACKED_PATH = "dataset_packed/train"            # where you want to save the packed version
tokenizer_path = "tokenizers_lp_uv/lp_32768_finewebedu_data"
SEQ_LEN = 2049      
NUM_PROC = 8                               # sequence length
batch_size=1000

# --- Load tokenizer (to get EOS token id) ---
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
EOS_TOKEN_ID = tokenizer.eos_token_id
print(f"EOS token id = {EOS_TOKEN_ID}")

# --- Load dataset ---
dataset = load_from_disk(DATASET_PATH)

# Assumes each example has {"input_ids": [list of token IDs]}
# Step 1: Shuffle documents
dataset = dataset.shuffle(seed=42)

def concat_docs(batch):
    all_tokens = []
    for ids in batch["input_ids"]:
        all_tokens.extend(ids + [EOS_TOKEN_ID])
    return {"all_tokens": all_tokens}

# Step 3: Chunk into fixed-size sequences
def chunk(batch):
    # Ensure all elements are 1D arrays
    token_seqs = []
    for x in batch["all_tokens"]:
        arr = np.atleast_1d(x)   # converts scalars → 1D arrays
        token_seqs.append(arr)

    tokens = np.concatenate(token_seqs)

    # Now do your chunking logic
    chunk_size = 2049
    chunks = [
        tokens[i : i + chunk_size]
        for i in range(0, len(tokens) - chunk_size + 1, chunk_size)
    ]

    return {"input_ids": chunks}



if __name__ == "__main__":
    concatenated = dataset.map(concat_docs, 
                            batched=True, 
                            batch_size=batch_size,
                            remove_columns=dataset.column_names,
                            desc="Concatenating Things",
                            num_proc=NUM_PROC
                            )


    chunked = concatenated.map(chunk, 
                            batched=True, 
                            batch_size=batch_size,
                            remove_columns=["all_tokens"],
                            desc="Splitting into chunks",
                            num_proc=NUM_PROC
                            )

    # Step 4: Shuffle final sequences
    packed = chunked.shuffle(seed=42)

    # Step 5: Save packed dataset
    packed.save_to_disk(PACKED_PATH,max_shard_size="3GB")

    print("Packed dataset saved at:", PACKED_PATH)
    print("Example sequence:", packed[0]["input_ids"][:50])  # preview first 50 tokens
