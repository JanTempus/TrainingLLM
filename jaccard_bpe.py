from transformers import AutoTokenizer
from itertools import combinations

def jaccard_distance(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return  intersection / union

def compare_tokenizer_vocabs(tokenizer_paths):
    """
    Compute pairwise Jaccard distances between vocabularies
    of multiple Hugging Face tokenizers.
    
    Args:
        tokenizer_paths: list of str
            Paths or model names for your saved tokenizers.
            
    Returns:
        dict of tuple -> float
            Dictionary mapping (tokenizer_i, tokenizer_j) -> Jaccard distance.
    """
    vocabs = {}
    for path in tokenizer_paths:
        tok = AutoTokenizer.from_pretrained(path)
        vocab = set(tok.get_vocab().keys())
        vocabs[path] = vocab
    
    results = {}
    for a, b in combinations(tokenizer_paths, 2):
        dist = jaccard_distance(vocabs[a], vocabs[b])
        results[(a, b)] = dist
    
    return results


tokenizer_paths = [
    "/local/home/jtempus/tokenisation_lp/sampled_bpe_tokenizer/bpe_32768_0",
    "/local/home/jtempus/tokenisation_lp/sampled_bpe_tokenizer/bpe_32768_1",
    "/local/home/jtempus/tokenisation_lp/sampled_bpe_tokenizer/bpe_32768_2",
    "/local/home/jtempus/tokenisation_lp/sampled_bpe_tokenizer/bpe_32768_3",
    "/local/home/jtempus/tokenisation_lp/sampled_bpe_tokenizer/bpe_32768_4",
]

results = compare_tokenizer_vocabs(tokenizer_paths)

for pair, dist in results.items():
    print(f"{pair[0]} vs {pair[1]}: Jaccard distance = {dist:.4f}")