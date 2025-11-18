# Debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=3

#uv run train.py  data.batch_size=64 tok_path="/local/home/jtempus/tokenisation_lp/bpe_tokenizer/bpe_tokenizers/bpe_8192_finewebedu" #For 8k vocab models
#uv run train.py  data.batch_size=32 tok_path="/local/home/jtempus/tokenisation_lp/bpe_tokenizer/bpe_tokenizers/bpe_131072_finewebedu" #For 131072k vocab models
uv run train.py  data.batch_size=64 tok_path="/local/home/jtempus/tokenisation_lp/lp_tokenizer/tokenizers_lp/lp_32768_finewebedu_data" #For 32k vocab models

#uv run train.py data.batch_size=32 optim.lr=0.001 