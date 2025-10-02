from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
DATASET_PATH = "tokenized_dataset_uv/validation"   # where your tokenized data is stored
PACKED_PATH = "dataset_packed/validation"            # where you want to save the packed version
tokenizer_path = "tokenizers_lp_uv/lp_32768_finewebedu_data"
SEQ_LEN = 2049      
NUM_PROC = 8                              # sequence length
batch_size=10000

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
    # import ipdb; ipdb.set_trace()
    return {"input_ids": all_tokens}

# Step 3: Chunk into fixed-size sequences


def chunk_docs(batch):
    # assert (len(batch["input_ids"]) % SEQ_LEN) == 0
    all_tokens = []
    chunk = []
    for ids in batch["input_ids"]:
        chunk += [ids]
        if len(chunk) == SEQ_LEN:
            all_tokens.extend([chunk])
            chunk = []
    # import ipdb; ipdb.set_trace()
    # chunks=[x["input_ids"] for x in batch.iter(batch_size=SEQ_LEN)]
    return {"input_ids": all_tokens}

# Example usage

if __name__ == "__main__":
    dataset = load_from_disk(DATASET_PATH)


    dataset = dataset.shuffle(seed=42)

    concatenated = dataset.map(concat_docs, 
                            batched=True, 
                            batch_size=batch_size,
                            remove_columns=dataset.column_names,
                            desc="Concatenating Things",
                            num_proc=NUM_PROC
                            )
    chunks = concatenated.map(chunk_docs, 
                            batched=True, 
                            batch_size=batch_size * SEQ_LEN,
                            remove_columns=concatenated.column_names,
                            desc="ConcatenatChunkiing Things",
                            num_proc=NUM_PROC
                            )
    print(concatenated)

    print("Finished Concatinating")
    
    # # arr=concatenated["all_tokens"]
    # # print(1)
    # print(1)
    # chunks = []
    # print(1)
    
    # chunks=list([x["input_ids"] for x in concatenated.iter(batch_size=SEQ_LEN)])
    # import ipdb; ipdb.set_trace()

        # for temp in concatenated.iter(batch_size=SEQ_LEN):
        #     # ipdb.set_trace()
            
        #     chunks.append(temp["all_tokens"])
        #     #print(temp)
        # # # Wrap into Hugging Face Dataset
    # print(1)
    # chunked= Dataset.from_dict({"input_ids": chunks})

    # Step 4: Shuffle final sequences
    packed = chunks.shuffle(seed=42)

    # Step 5: Save packed dataset
    packed.save_to_disk(PACKED_PATH,max_shard_size="3GB")

    print("Packed dataset saved at:", PACKED_PATH)
    # import ipdb; ipdb.set_trace()

    