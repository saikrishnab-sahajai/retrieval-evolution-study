"""
Phase 2 & 3: Evaluate dense retrievers (Word2Vec, BiEncoder models, DPR).

Usage:
    python scripts/run_dense_eval.py
    python scripts/run_dense_eval.py --datasets scifact fiqa
    python scripts/run_dense_eval.py --models minilm mpnet bge
    python scripts/run_dense_eval.py --models word2vec_mean word2vec_idf
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

# Prevent SIGABRT from duplicate OpenMP runtimes (FAISS + PyTorch on macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mlflow
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrievers.dense import (
    BiEncoderRetriever,
    Doc2VecRetriever,
    DPRRetriever,
    Word2VecRetriever,
    get_device,
)
from src.evaluation.metrics import compute_metrics, format_results_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_beir_dataset(dataset_name: str, data_dir: Path, split: str = "test"):
    from beir.datasets.data_loader import GenericDataLoader
    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}. Run: python scripts/download_datasets.py")
        sys.exit(1)
    return GenericDataLoader(data_folder=str(dataset_path)).load(split=split)


def build_retriever(model_name: str, model_cfg: dict, device: str, batch_size: int):
    """Instantiate the right retriever class from model config."""

    if model_name in ("word2vec_mean", "word2vec_idf"):
        pooling = "mean" if model_name == "word2vec_mean" else "idf_weighted"
        return Word2VecRetriever(
            model_key=model_cfg["early_neural"]["word2vec_mean"]["model_path"],
            pooling=pooling,
        )

    if model_name == "doc2vec_dbow":
        cfg = model_cfg["early_neural"]["doc2vec_dbow"]
        return Doc2VecRetriever(
            vector_size=cfg.get("vector_size", 300),
            epochs=cfg.get("epochs", 40),
            dm=cfg.get("dm", 0),
            min_count=cfg.get("min_count", 2),
        )

    if model_name == "dpr":
        return DPRRetriever(
            ctx_encoder_id=model_cfg["dense"]["dpr"]["hf_id"],
            q_encoder_id="facebook/dpr-question_encoder-single-nq-base",
            device=device,
            batch_size=batch_size,
        )

    # BiEncoder models
    dm = model_cfg["dense"][model_name]
    query_prefix = dm.get("instruction_prefix_query") or dm.get("instruction_prefix") or ""
    passage_prefix = dm.get("instruction_prefix_passage") or ""
    return BiEncoderRetriever(
        model_id=dm["hf_id"],
        device=device,
        batch_size=batch_size,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval evaluation")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--models", nargs="*",
        default=["word2vec_mean", "word2vec_idf", "doc2vec_dbow", "minilm", "dpr", "mpnet", "bge", "e5"],
    )
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

    device = model_cfg.get("device", get_device())
    batch_size = model_cfg.get("batch_size", 128)
    logger.info(f"Device: {device}  |  Batch size: {batch_size}")
    if device == "mps":
        logger.info(f"MPS available: {torch.backends.mps.is_available()}")

    # MLflow
    mlflow.set_tracking_uri(str(REPO_ROOT / eval_cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(eval_cfg["mlflow"]["experiment_name"])

    # Datasets
    datasets_to_run = {
        name: meta
        for name, meta in ds_cfg["datasets"].items()
        if meta["enabled"] and (args.datasets is None or name in args.datasets)
    }

    # Resume: load existing results to skip already-completed (model, dataset) pairs
    out_path = results_dir / "dense_results.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
        completed = [(m, d) for m, ds in all_results.items() for d in ds]
        logger.info(f"Resuming — {len(completed)} (model, dataset) pairs already done: {completed}")
    else:
        all_results = {}

    for model_name in args.models:
        if model_name not in all_results:
            all_results[model_name] = {}
        retriever = None  # build fresh per model (not per dataset)

        for ds_name, ds_meta in datasets_to_run.items():
            if ds_name in all_results[model_name]:
                logger.info(f"Skipping {model_name} | {ds_name} — already done")
                continue

            split = ds_meta.get("split", "test")
            logger.info(f"\n--- {model_name} | {ds_name} ---")

            corpus, queries, qrels = load_beir_dataset(ds_meta["beir_name"], data_dir, split)
            logger.info(f"  corpus={len(corpus):,}  queries={len(queries):,}")

            if retriever is None:
                retriever = build_retriever(model_name, model_cfg, device, batch_size)

            t0 = time.time()
            retriever.index(corpus)
            index_time = time.time() - t0

            t1 = time.time()
            results = retriever.retrieve(queries, top_k=eval_cfg["retrieval_top_k"])
            query_time = time.time() - t1

            metrics = compute_metrics(
                qrels, results,
                ndcg_at=eval_cfg["metrics"]["ndcg_at"],
                mrr_at=eval_cfg["metrics"]["mrr_at"],
                map_at=eval_cfg["metrics"]["map_at"],
                recall_at=eval_cfg["metrics"]["recall_at"],
            )
            metrics["index_time_s"] = round(index_time, 2)
            metrics["query_time_s"] = round(query_time, 2)

            with mlflow.start_run(run_name=f"{model_name}_{ds_name}"):
                _early = model_name in ("word2vec_mean", "word2vec_idf", "doc2vec_dbow")
                hf_id = model_name if _early else model_cfg["dense"].get(model_name, {}).get("hf_id", model_name)
                phase = "2_early_neural" if _early else "3_dense"
                method_type = "early_neural" if _early else "dense"
                mlflow.set_tags({
                    "phase": phase,
                    "method_type": method_type,
                    "dataset": ds_name,
                    "model": model_name,
                })
                mlflow.log_params({
                    "model": model_name,
                    "hf_id": hf_id,
                    "dataset": ds_name,
                    "device": device,
                    "batch_size": batch_size,
                    "corpus_size": len(corpus),
                })
                mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items()})

            all_results[model_name][ds_name] = metrics
            logger.info(f"  {metrics}")

            # Save incrementally after every (model, dataset) pair
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"  Saved to {out_path}")

            # Explicit cleanup — FAISS index + model weights + MPS cache
            retriever = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    out_path = results_dir / "dense_results.json"
    logger.info(f"\nFinal results saved to {out_path}")

    print("\n" + "=" * 60)
    print("NDCG@10 Summary — Dense Retrieval")
    print("=" * 60)
    print(format_results_table(all_results, primary_metric="ndcg@10"))


if __name__ == "__main__":
    main()
