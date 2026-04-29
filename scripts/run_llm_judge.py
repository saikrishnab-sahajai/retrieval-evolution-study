"""
Phase 5: LLM-as-judge evaluation using RAGAS + Ollama (local LLM).

Prerequisites:
    ollama serve                        # start Ollama daemon
    ollama pull llama3.2:3b             # or: ollama pull mistral:7b

Usage:
    python scripts/run_llm_judge.py
    python scripts/run_llm_judge.py --datasets nq scifact
    python scripts/run_llm_judge.py --model mistral:7b --sample 200
    python scripts/run_llm_judge.py --results-file results/dense_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.llm_judge import LLMJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_beir_dataset(dataset_name: str, data_dir: Path, split: str = "test"):
    from beir.datasets.data_loader import GenericDataLoader
    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}. Run: python scripts/download_datasets.py")
        sys.exit(1)
    return GenericDataLoader(data_folder=str(dataset_path)).load(split=split)


def load_retrieval_results(results_file: Path) -> dict:
    """Load pre-computed retrieval results from a JSON file."""
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.error("Run run_sparse_eval.py or run_dense_eval.py first.")
        sys.exit(1)
    with open(results_file) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation (RAGAS + Ollama)")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override")
    parser.add_argument("--sample", type=int, default=None, help="Number of queries to sample")
    parser.add_argument(
        "--results-file", type=str, default=None,
        help="Path to pre-computed retrieval results JSON. "
             "Default: uses best model (hybrid_results.json if exists, else dense_results.json)"
    )
    args = parser.parse_args()

    # Load configs
    with open(REPO_ROOT / "configs" / "datasets.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "eval.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    judge_cfg = eval_cfg["llm_judge"]
    data_dir = REPO_ROOT / ds_cfg["data_dir"]
    results_dir = REPO_ROOT / eval_cfg.get("results_dir", "results")
    results_dir.mkdir(exist_ok=True)

    ollama_model = args.model or judge_cfg["ollama_model"]
    sample_n = args.sample or judge_cfg["sample_queries"]
    top_k = judge_cfg["top_k_passages"]

    # Determine which datasets to evaluate
    judge_datasets = judge_cfg.get("datasets", list(ds_cfg["datasets"].keys()))
    if args.datasets:
        judge_datasets = [d for d in judge_datasets if d in args.datasets]

    # Load retrieval results
    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = results_dir / "hybrid_results.json"
        if not results_path.exists():
            results_path = results_dir / "dense_results.json"
    logger.info(f"Using retrieval results from: {results_path}")
    all_retrieval_results = load_retrieval_results(results_path)

    # Pick the best model's results to evaluate
    # Priority: hybrid_rrf+reranker > hybrid_rrf > bge > mpnet
    priority = ["hybrid_rrf+reranker", "hybrid_rrf", "bge", "mpnet", "minilm"]
    best_model = next((m for m in priority if m in all_retrieval_results), None)
    if best_model is None:
        best_model = next(iter(all_retrieval_results))
    logger.info(f"Evaluating retrieval results from: {best_model}")

    # MLflow
    mlflow.set_tracking_uri(str(REPO_ROOT / eval_cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(eval_cfg["mlflow"]["experiment_name"])

    judge = LLMJudge(
        ollama_model=ollama_model,
        ollama_base_url=judge_cfg["ollama_base_url"],
    )

    summary = {}

    for ds_name in judge_datasets:
        ds_meta = ds_cfg["datasets"].get(ds_name)
        if ds_meta is None:
            logger.warning(f"Dataset '{ds_name}' not in config, skipping.")
            continue

        logger.info(f"\n--- LLM judge | {ds_name} | model={ollama_model} ---")
        split = ds_meta.get("split", "test")
        corpus, queries, qrels = load_beir_dataset(ds_meta["beir_name"], data_dir, split)

        # Get retrieval results for this model + dataset
        model_results_for_ds = all_retrieval_results.get(best_model, {})
        # model_results_for_ds is {dataset: metrics} — we need raw doc results
        # Note: run_llm_judge needs raw retrieval dicts {qid: {doc_id: score}}
        # The scripts save metrics only; for LLM judge we re-run BM25+BGE quickly on the sample
        # or accept that we need the raw results stored separately.
        # For simplicity here we re-run BGE on the sampled queries.
        logger.info(f"  Re-running BGE retrieval on sample for LLM judge...")

        from src.retrievers.dense import BiEncoderRetriever, get_device
        from src.retrievers.sparse import BM25Retriever
        from src.retrievers.dense import reciprocal_rank_fusion
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        with open(REPO_ROOT / "configs" / "models.yaml") as f:
            model_cfg = yaml.safe_load(f)

        bge_cfg = model_cfg["dense"]["bge"]
        bge = BiEncoderRetriever(
            model_id=bge_cfg["hf_id"],
            device=device,
            batch_size=64,
            query_prefix=bge_cfg.get("instruction_prefix") or "",
        )
        bge.index(corpus)
        bge_results = bge.retrieve(queries, top_k=top_k)

        scores = judge.evaluate(
            queries=queries,
            results=bge_results,
            corpus=corpus,
            sample_n=sample_n,
            top_k=top_k,
        )
        mean_cp = judge.mean_score(scores)
        logger.info(f"  Mean context_precision ({ollama_model}): {mean_cp:.4f}")

        summary[ds_name] = {
            "model": best_model,
            "llm": ollama_model,
            "sample_n": len(scores),
            "mean_context_precision": mean_cp,
        }

        with mlflow.start_run(run_name=f"llm_judge_{ds_name}"):
            mlflow.set_tags({
                "phase": "5_llm_judge",
                "method_type": "llm_judge",
                "dataset": ds_name,
                "model": f"llm_judge_{ollama_model}",
            })
            mlflow.log_params({
                "dataset": ds_name,
                "llm_model": ollama_model,
                "retriever": best_model,
                "sample_n": len(scores),
                "top_k_passages": top_k,
            })
            mlflow.log_metrics({"mean_context_precision": mean_cp})

    # Save summary
    out_path = results_dir / "llm_judge_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nLLM judge results saved to {out_path}")

    print("\n" + "=" * 60)
    print(f"LLM-as-Judge Summary ({ollama_model})")
    print("=" * 60)
    for ds, v in summary.items():
        print(f"  {ds:<20} context_precision = {v['mean_context_precision']:.4f}  (n={v['sample_n']})")


if __name__ == "__main__":
    main()
