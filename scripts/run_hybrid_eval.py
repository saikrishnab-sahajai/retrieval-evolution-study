"""
Phase 4: Hybrid retrieval (BM25 + BGE RRF) + cross-encoder reranker.

Usage:
    python scripts/run_hybrid_eval.py
    python scripts/run_hybrid_eval.py --datasets nq trec-covid
    python scripts/run_hybrid_eval.py --skip-reranker
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mlflow
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrievers.sparse import BM25Retriever
from src.retrievers.dense import BiEncoderRetriever, get_device, reciprocal_rank_fusion
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


def rerank_with_cross_encoder(
    queries: dict,
    fused_results: dict,
    corpus: dict,
    model_id: str,
    device: str,
    first_stage_k: int = 100,
    final_k: int = 10,
) -> dict:
    """
    Apply cross-encoder reranker on top of fused first-stage results.
    Scores every (query, passage) pair in the top-K; returns top final_k.
    """
    from sentence_transformers import CrossEncoder

    logger.info(f"Loading cross-encoder: {model_id} on {device}")
    cross_encoder = CrossEncoder(model_id, device=device)

    reranked = {}
    qids = list(queries.keys())

    for qid in qids:
        q_text = queries[qid]
        top_docs = sorted(
            fused_results.get(qid, {}).items(), key=lambda x: x[1], reverse=True
        )[:first_stage_k]

        if not top_docs:
            reranked[qid] = {}
            continue

        doc_ids = [d for d, _ in top_docs]
        passages = [
            (corpus[did].get("title", "") + " " + corpus[did].get("text", "")).strip()
            for did in doc_ids
            if did in corpus
        ]
        # Align doc_ids with passages (skip missing)
        valid_doc_ids = [did for did in doc_ids if did in corpus]

        pairs = [[q_text, p] for p in passages]
        scores = cross_encoder.predict(pairs, show_progress_bar=False)

        # Return ALL reranked docs (not just top final_k) so recall@50/100 stay valid.
        # NDCG@10 and MRR@10 naturally use only the top-10 of the reranked order.
        scored = sorted(zip(valid_doc_ids, scores), key=lambda x: x[1], reverse=True)
        reranked[qid] = {did: float(score) for did, score in scored}

    return reranked


def main():
    parser = argparse.ArgumentParser(description="Hybrid retrieval + reranking evaluation")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--skip-reranker", action="store_true",
                        help="Skip cross-encoder reranking step")
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
    rrf_k = eval_cfg["rrf"]["k"]
    retrieval_top_k = eval_cfg["retrieval_top_k"]

    reranker_cfg = model_cfg["reranking"]["cross_encoder"]
    first_stage_k = reranker_cfg["first_stage_top_k"]
    final_k = reranker_cfg["rerank_top_k"]
    reranker_model = reranker_cfg["hf_id"]

    # BGE config
    bge_cfg = model_cfg["dense"]["bge"]
    bge_query_prefix = bge_cfg.get("instruction_prefix") or bge_cfg.get("instruction_prefix_query") or ""
    bge_passage_prefix = bge_cfg.get("instruction_prefix_passage") or ""

    # MLflow
    mlflow.set_tracking_uri(str(REPO_ROOT / eval_cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(eval_cfg["mlflow"]["experiment_name"])

    datasets_to_run = {
        name: meta
        for name, meta in ds_cfg["datasets"].items()
        if meta["enabled"] and (args.datasets is None or name in args.datasets)
    }

    # Resume: load existing results to skip already-completed datasets
    out_path = results_dir / "hybrid_results.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
        done = set()
        for variant in all_results.values():
            done.update(variant.keys())
        logger.info(f"Resuming — datasets already done: {done}")
    else:
        all_results = {}

    for ds_name, ds_meta in datasets_to_run.items():
        # Skip if both hybrid and reranked results already saved for this dataset
        hybrid_done = ds_name in all_results.get("hybrid_rrf", {})
        reranked_done = ds_name in all_results.get("hybrid_rrf+reranker", {})
        if hybrid_done and (args.skip_reranker or reranked_done):
            logger.info(f"Skipping {ds_name} — already done")
            continue

        split = ds_meta.get("split", "test")
        logger.info(f"\n{'='*60}\n  Dataset: {ds_name} (split={split})\n{'='*60}")

        corpus, queries, qrels = load_beir_dataset(ds_meta["beir_name"], data_dir, split)
        logger.info(f"  corpus={len(corpus):,}  queries={len(queries):,}")

        # ---- BM25 first-stage ----
        logger.info("  [1/4] BM25 indexing + retrieval...")
        bm25_params = model_cfg["sparse"]["bm25"]["params"]
        bm25 = BM25Retriever(**bm25_params)
        t0 = time.time()
        bm25.index(corpus)
        bm25_index_time = time.time() - t0
        t0 = time.time()
        bm25_results = bm25.retrieve(queries, top_k=retrieval_top_k)
        bm25_query_time = time.time() - t0

        # ---- BGE first-stage ----
        logger.info("  [2/4] BGE indexing + retrieval...")
        bge = BiEncoderRetriever(
            model_id=bge_cfg["hf_id"],
            device=device,
            batch_size=batch_size,
            query_prefix=bge_query_prefix,
            passage_prefix=bge_passage_prefix,
        )
        t0 = time.time()
        bge.index(corpus)
        bge_index_time = time.time() - t0
        t0 = time.time()
        bge_results = bge.retrieve(queries, top_k=retrieval_top_k)
        bge_query_time = time.time() - t0

        # ---- RRF fusion ----
        logger.info(f"  [3/4] RRF fusion (k={rrf_k})...")
        t0 = time.time()
        fused_results = reciprocal_rank_fusion([bm25_results, bge_results], k=rrf_k)
        rrf_time = time.time() - t0

        # Evaluate hybrid (before reranking)
        hybrid_metrics = compute_metrics(
            qrels, fused_results,
            ndcg_at=eval_cfg["metrics"]["ndcg_at"],
            mrr_at=eval_cfg["metrics"]["mrr_at"],
            map_at=eval_cfg["metrics"]["map_at"],
            recall_at=eval_cfg["metrics"]["recall_at"],
        )
        hybrid_metrics["index_time_s"] = round(bm25_index_time + bge_index_time, 2)
        hybrid_metrics["query_time_s"] = round(bm25_query_time + bge_query_time + rrf_time, 2)

        with mlflow.start_run(run_name=f"hybrid_rrf_{ds_name}"):
            mlflow.set_tags({
                "phase": "4_hybrid",
                "method_type": "hybrid",
                "dataset": ds_name,
                "model": "hybrid_rrf",
            })
            mlflow.log_params({
                "model": "BM25+BGE (RRF)",
                "dataset": ds_name,
                "rrf_k": rrf_k,
                "device": device,
            })
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in hybrid_metrics.items()})

        all_results.setdefault("hybrid_rrf", {})[ds_name] = hybrid_metrics
        logger.info(f"  Hybrid RRF: {hybrid_metrics}")

        # ---- Cross-encoder reranker (optional) ----
        if not args.skip_reranker:
            logger.info(f"  [4/4] Cross-encoder reranking (top-{first_stage_k} → top-{final_k})...")
            t0 = time.time()
            reranked_results = rerank_with_cross_encoder(
                queries, fused_results, corpus,
                model_id=reranker_model,
                device=device,
                first_stage_k=first_stage_k,
                final_k=final_k,
            )
            rerank_time = time.time() - t0

            reranked_metrics = compute_metrics(
                qrels, reranked_results,
                ndcg_at=eval_cfg["metrics"]["ndcg_at"],
                mrr_at=eval_cfg["metrics"]["mrr_at"],
                map_at=eval_cfg["metrics"]["map_at"],
                recall_at=eval_cfg["metrics"]["recall_at"],
            )
            reranked_metrics["rerank_time_s"] = round(rerank_time, 2)

            with mlflow.start_run(run_name=f"hybrid_rrf_reranked_{ds_name}"):
                mlflow.set_tags({
                    "phase": "4_hybrid",
                    "method_type": "hybrid_reranked",
                    "dataset": ds_name,
                    "model": "hybrid_rrf+reranker",
                })
                mlflow.log_params({
                    "model": "BM25+BGE+CrossEncoder",
                    "dataset": ds_name,
                    "rrf_k": rrf_k,
                    "reranker": reranker_model,
                    "first_stage_k": first_stage_k,
                    "final_k": final_k,
                })
                mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in reranked_metrics.items()})

            all_results.setdefault("hybrid_rrf+reranker", {})[ds_name] = reranked_metrics
            logger.info(f"  Reranked: {reranked_metrics}")

        # Incremental save + cleanup after each dataset
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Saved to {out_path}")
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    logger.info(f"\nFinal results saved to {out_path}")

    print("\n" + "=" * 60)
    print("NDCG@10 Summary — Hybrid Retrieval")
    print("=" * 60)
    print(format_results_table(all_results, primary_metric="ndcg@10"))


if __name__ == "__main__":
    main()
