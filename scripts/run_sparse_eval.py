"""
Phase 1: Evaluate sparse retrievers (BOW, TF-IDF, BM25) across BEIR datasets.

Usage:
    python scripts/run_sparse_eval.py
    python scripts/run_sparse_eval.py --datasets scifact fiqa
    python scripts/run_sparse_eval.py --models bm25 tfidf
    python scripts/run_sparse_eval.py --datasets trec-covid --k1 1.2 --b 0.8
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import mlflow
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrievers.sparse import BOWRetriever, BM25Retriever, TFIDFRetriever
from src.evaluation.metrics import compute_metrics, format_results_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_beir_dataset(dataset_name: str, data_dir: Path, split: str = "test"):
    from beir.datasets.data_loader import GenericDataLoader
    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}. Run: python scripts/download_datasets.py")
        sys.exit(1)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split=split)
    return corpus, queries, qrels


def run_one(retriever_cls, retriever_kwargs, dataset_name, corpus, queries, qrels, eval_cfg, run_name):
    retriever = retriever_cls(**retriever_kwargs)

    t_index = time.time()
    retriever.index(corpus)
    index_time = time.time() - t_index

    t_query = time.time()
    results = retriever.retrieve(queries, top_k=eval_cfg["retrieval_top_k"])
    query_time = time.time() - t_query

    metrics = compute_metrics(
        qrels, results,
        ndcg_at=eval_cfg["metrics"]["ndcg_at"],
        mrr_at=eval_cfg["metrics"]["mrr_at"],
        map_at=eval_cfg["metrics"]["map_at"],
        recall_at=eval_cfg["metrics"]["recall_at"],
    )
    metrics["index_time_s"] = round(index_time, 2)
    metrics["query_time_s"] = round(query_time, 2)

    logger.info(f"  {run_name} | {dataset_name}: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Sparse retrieval evaluation")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=["bow", "tfidf", "bm25"],
                        choices=["bow", "tfidf", "bm25"])
    parser.add_argument("--k1", type=float, default=None, help="BM25 k1 override")
    parser.add_argument("--b", type=float, default=None, help="BM25 b override")
    args = parser.parse_args()

    # Load configs
    with open(REPO_ROOT / "configs" / "datasets.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "eval.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    data_dir = REPO_ROOT / ds_cfg["data_dir"]
    results_dir = REPO_ROOT / eval_cfg.get("results_dir", "results")
    results_dir.mkdir(exist_ok=True)

    # MLflow
    mlflow.set_tracking_uri(str(REPO_ROOT / eval_cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(eval_cfg["mlflow"]["experiment_name"])

    # Determine datasets
    datasets_to_run = {}
    for name, meta in ds_cfg["datasets"].items():
        if not meta["enabled"]:
            continue
        if args.datasets and name not in args.datasets:
            continue
        datasets_to_run[name] = meta

    # Retriever configs
    sparse_cfgs = model_cfg["sparse"]
    bm25_params = sparse_cfgs["bm25"]["params"].copy()
    if args.k1 is not None:
        bm25_params["k1"] = args.k1
    if args.b is not None:
        bm25_params["b"] = args.b

    retriever_map = {
        "bow":   (BOWRetriever,   {"max_features": sparse_cfgs["bow"]["params"]["max_features"]}),
        "tfidf": (TFIDFRetriever, {"max_features": sparse_cfgs["tfidf"]["params"]["max_features"],
                                   "sublinear_tf": sparse_cfgs["tfidf"]["params"]["sublinear_tf"]}),
        "bm25":  (BM25Retriever,  bm25_params),
    }

    all_results = {}  # {model_name: {dataset_name: metrics}}

    for model_name in args.models:
        cls, kwargs = retriever_map[model_name]
        all_results[model_name] = {}

        for ds_name, ds_meta in datasets_to_run.items():
            split = ds_meta.get("split", "test")
            logger.info(f"\n--- {model_name.upper()} | {ds_name} (split={split}) ---")

            corpus, queries, qrels = load_beir_dataset(ds_meta["beir_name"], data_dir, split)
            logger.info(f"  corpus={len(corpus):,}  queries={len(queries):,}")

            with mlflow.start_run(run_name=f"{model_name}_{ds_name}"):
                mlflow.set_tags({
                    "phase": "1_sparse",
                    "method_type": "sparse",
                    "dataset": ds_name,
                    "model": model_name,
                })
                mlflow.log_params({
                    "model": model_name,
                    "dataset": ds_name,
                    "corpus_size": len(corpus),
                    "query_count": len(queries),
                    **kwargs,
                })
                metrics = run_one(cls, kwargs, ds_name, corpus, queries, qrels,
                                  eval_cfg, run_name=model_name)
                mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})

            all_results[model_name][ds_name] = metrics

    # Save results JSON
    out_path = results_dir / "sparse_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("NDCG@10 Summary — Sparse Retrieval")
    print("=" * 60)
    print(format_results_table(all_results, primary_metric="ndcg@10"))


if __name__ == "__main__":
    main()
