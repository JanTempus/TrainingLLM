"""
Benchmark Evaluation Script

Evaluates trained language models on standard benchmarks using lm-evaluation-harness:
- Hellaswag
- ARC-E (arc_easy)
- ARC-C (arc_challenge)  
- Winogrande
- PIQA

Usage:
    python scripts/run_benchmarks.py run_path=outputs/run_2025-10-13_bpe_32k_57M checkpoint=step50000
    python scripts/run_benchmarks.py run_path=outputs/run_2025-10-13_bpe_32k_57M checkpoint=last benchmarks=[hellaswag,piqa]
"""

import json
import logging
from pathlib import Path

import hydra
import srsly
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from src.primer.model import load_hf_from_pl
from src.primer.utilities import get_logger

# Register resolver for folder name extraction
OmegaConf.register_new_resolver(
    "get_folder_name", lambda x, y: Path(x).name if x is not None else Path(y.split("/")[1]).name
)

SEP_LINE = "=" * 80
logger = get_logger("benchmark_eval")


def run_evaluation(
    model,
    tokenizer,
    benchmarks: list[str],
    batch_size: str | int,
    num_fewshot: int,
    limit: int | None,
    device: str,
) -> dict:
    """Run lm-evaluation-harness on the specified benchmarks."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    # Wrap model for lm-eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=benchmarks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
    )

    return results


@hydra.main(version_base=None, config_path="../conf", config_name="benchmark_conf")
def main(cfg: DictConfig) -> None:
    # =============================
    # Step 1. Prepare configuration
    # =============================
    OmegaConf.resolve(cfg)
    OmegaConf.save(cfg, "./hparams.yaml")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}\n{SEP_LINE}")

    torch.manual_seed(cfg.seed)

    # ===========================
    # Step 2. Load model
    # ===========================
    if cfg.run_path:
        # Get metadata from training run
        run_path = Path(cfg.run_path)
        hparams = srsly.read_yaml(run_path / "hparams.yaml")
        tok_path = Path(hparams["tok_path"])  # type: ignore

        # Load model from checkpoint
        ckpt_path = run_path / ".checkpoints" / f"{cfg.checkpoint}.ckpt"
        logger.info(f"Loading model from {ckpt_path}")
        model = load_hf_from_pl(ckpt_path)

    elif all([cfg.tok_path, cfg.repo_id, cfg.checkpoint]):
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading model from {cfg.repo_id=}, {cfg.checkpoint=}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.repo_id, revision=cfg.checkpoint, cache_dir=".model_cache"
        )
        tok_path = Path(cfg.tok_path)

    else:
        raise ValueError(
            "Either `run_path` or (`tok_path`, `repo_id`, and `checkpoint`) must be provided"
        )

    # Load tokenizer
    logger.info(f"Loading tokenizer from {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Move model to device and set to eval mode
    model = model.to(cfg.device)
    model.eval()

    # ===========================
    # Step 3. Run benchmarks
    # ===========================
    benchmarks = list(cfg.benchmarks)
    logger.info(f"Running benchmarks: {benchmarks}")
    logger.info(f"Batch size: {cfg.batch_size}, Num fewshot: {cfg.num_fewshot}")

    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        benchmarks=benchmarks,
        batch_size=cfg.batch_size,
        num_fewshot=cfg.num_fewshot,
        limit=cfg.limit,
        device=cfg.device,
    )

    # ===========================
    # Step 4. Save and display results
    # ===========================
    # Save full results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Full results saved to benchmark_results.json")

    # Display summary
    logger.info(f"\n{SEP_LINE}")
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info(SEP_LINE)

    if "results" in results:
        for task_name, task_results in results["results"].items():
            # Get the main accuracy metric
            acc_key = None
            for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
                if key in task_results:
                    acc_key = key
                    break

            if acc_key:
                accuracy = task_results[acc_key]
                if isinstance(accuracy, float):
                    logger.info(f"  {task_name}: {accuracy * 100:.2f}%")
                else:
                    logger.info(f"  {task_name}: {accuracy}")
            else:
                logger.info(f"  {task_name}: {task_results}")

    logger.info(SEP_LINE)

    # Save summary CSV for easy comparison
    summary = []
    if "results" in results:
        for task_name, task_results in results["results"].items():
            row = {"task": task_name}
            for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
                if key in task_results:
                    row["accuracy"] = task_results[key]
                    break
            summary.append(row)

    srsly.write_jsonl("benchmark_summary.jsonl", summary)
    logger.info("Summary saved to benchmark_summary.jsonl")


if __name__ == "__main__":
    main()
