from datasets import load_from_disk, Dataset, disable_caching,concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import os

# --- CONFIG ---
SEQ_LEN = 2049      
NUM_PROC = 16                              # sequence length
batch_size=10000

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

def pack_data(dataset,PACKED_PATH,tokenizer_path):    

    dataset = dataset.shuffle(seed=42)
    dataset=dataset.flatten_indices()
    EOS_TOKEN_ID
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
    # Step 5: Save packed datasetdatasets-cli cache clear
    packed.save_to_disk(PACKED_PATH,max_shard_size="3GB")

    print("Packed dataset saved at:", PACKED_PATH)



def merge_datasets(input_paths, output_path, max_shard_size="3GB"):
    """
    Load multiple Hugging Face datasets from disk, concatenate them,
    and save the merged dataset back to disk in shards.

    Args:
        input_paths (list[str]): List of dataset directories to load.
        output_path (str): Path to save the merged dataset.
        max_shard_size (str): Maximum size per shard, e.g. "3GB".
    """
    datasets = [load_from_disk(p) for p in input_paths]
    merged = concatenate_datasets(datasets)
    merged.save_to_disk(output_path, max_shard_size=max_shard_size)
    print(f"✅ Merged {len(input_paths)} datasets and saved to {output_path}")


if __name__ == "__main__":

    base_path="dataset_packed_bpe_65536/"
    tokenizer_path = "/local/home/jtempus/tokenisation_lp/bpe_tokenizer/bpe_tokenizers/bpe_65536_finewebedu"

    TOKENIZED_PATH_val = "tokenized_dataset_uv/validation"   # where your tokenized data is stored
    PACKED_PATH_val = base_path+"validation"            # where you want to save the packed version




    TOKENIZED_PATH_train = "tokenized_dataset_uv/train"            # where you want to save the packed version
   

    # --- Load tokenizer (to get EOS token id) ---
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    print(f"EOS token id = {EOS_TOKEN_ID}")

    if EOS_TOKEN_ID is None:
        raise ValueError("EOS_Token not defined")

    datasets=[]
    for i in range(3):
        PACKED_PATH_train = base_path+f"train_{i}"
        datasets.append(PACKED_PATH_train)
        
        dataset = load_from_disk(TOKENIZED_PATH_train)
        dataset_len= int(len(dataset)/4)
        print(i*dataset_len)
        print((i+1)*dataset_len)
        dataset=dataset.select(range(i*dataset_len,(i+1)*dataset_len))
        pack_data(dataset,PACKED_PATH_train,tokenizer_path)
    
    merge_datasets(datasets,base_path+"train")

    dataset=load_from_disk(TOKENIZED_PATH_val)
    pack_data(dataset,PACKED_PATH_val,tokenizer_path)