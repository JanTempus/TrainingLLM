from datasets import load_from_disk, Dataset, disable_caching
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import os
disable_caching()

# --- CONFIG ---
DATASET_PATH = "tokenized_dataset_uv/validation"   # where your tokenized data is stored
PACKED_PATH = "dataset_packed_lp/validation"            # where you want to save the packed version
#DATASET_PATH = "tokenized_dataset_uv/train"   # where your tokenized data is stored
#PACKED_PATH = "dataset_packed_lp/train_3"            # where you want to save the packed version
tokenizer_path = "/local/home/jtempus/tokenisation_lp/lp_tokenizer/tokenizers_lp/lp_32768_finewebedu_data"
SEQ_LEN = 2049      
NUM_PROC = 8                              # sequence length
batch_size=10000


# --- Load tokenizer (to get EOS token id) ---
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
EOS_TOKEN_ID = tokenizer.eos_token_id
print(f"EOS token id = {EOS_TOKEN_ID}")

if EOS_TOKEN_ID is None:
    raise ValueError("EOS_Token not defined")

# Assumes each example has {"input_ids": [list of token IDs]}
# Step 1: Shuffle documents


def concat_docs(batch):
    all_tokens = []
    for ids in batch["input_ids"]:
        all_tokens.extend(ids + [EOS_TOKEN_ID])
    # import ipdb; ipdb.set_trace()
    return {"input_ids": all_tokens}

# Step 3: Chunk into fixed-size sequences


def chunk_docs(batch):
    all_tokens = []
    chunk = []
    for ids in batch["input_ids"]:
        chunk += [ids]
        if len(chunk) == SEQ_LEN:
            all_tokens.extend([chunk])
            chunk = []
    return {"input_ids": all_tokens}

# Example usage

if __name__ == "__main__":

    
    dataset = load_from_disk(DATASET_PATH)
    #dataset_len= int(len(dataset)/4)
    #dataset=dataset.select(range(2*dataset_len,3*dataset_len))


    dataset = dataset.shuffle(seed=42)
    dataset=dataset.flatten_indices()

    
    concatenated = dataset.map(concat_docs, 
                            batched=True, 
                            batch_size=batch_size,
                            remove_columns=dataset.column_names,
                            desc="Concatenating Things",
                            num_proc=NUM_PROC
                            )

    print("Finished Concatinating")
    chunks = concatenated.map(chunk_docs, 
                            batched=True, 
                            batch_size=batch_size * SEQ_LEN,
                            remove_columns=concatenated.column_names,
                            desc="Chunking Things",
                            num_proc=NUM_PROC
                            )
    packed = chunks.shuffle(seed=42)
    packed.flatten_indices()
    # Step 5: Save packed datasetdatasets-cli cache clear
    packed.save_to_disk(PACKED_PATH,max_shard_size="3GB")

    print("Packed dataset saved at:", PACKED_PATH)
    # import ipdb; ipdb.set_trace()

    