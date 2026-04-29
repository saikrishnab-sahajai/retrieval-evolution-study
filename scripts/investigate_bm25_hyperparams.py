"""
BM25 Hyperparameter Investigation — Phase 1 open question.

Why does TF-IDF beat BM25 on SciFact and FIQA but not TREC-COVID?

This script runs:
  1. Passage length stats per dataset
  2. BM25 b sweep (0.0 → 1.0) — isolates length normalisation effect
  3. BM25 k₁ sweep (0.5 → 2.5) — isolates term frequency saturation effect

Results saved to: results/bm25_investigation.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from beir.datasets.data_loader import GenericDataLoader
from src.retrievers.sparse import BM25Retriever, TFIDFRetriever
from src.evaluation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "datasets"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "scifact":   "test",
    "fiqa":      "test",
    "trec-covid": "test",
}

B_VALUES  = [0.0, 0.25, 0.5, 0.75, 1.0]
K1_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]


def load_dataset(name, split):
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(DATA_DIR / name)
    ).load(split=split)
    return corpus, queries, qrels


def passage_length_stats(corpus):
    lengths = [
        len((d.get("title", "") + " " + d.get("text", "")).split())
        for d in corpus.values()
    ]
    return {
        "count":  len(lengths),
        "mean":   round(float(np.mean(lengths)), 1),
        "median": round(float(np.median(lengths)), 1),
        "std":    round(float(np.std(lengths)), 1),
        "min":    int(np.min(lengths)),
        "max":    int(np.max(lengths)),
        "p25":    round(float(np.percentile(lengths, 25)), 1),
        "p75":    round(float(np.percentile(lengths, 75)), 1),
    }


def run_bm25(corpus, queries, qrels, k1, b):
    r = BM25Retriever(k1=k1, b=b)
    r.index(corpus)
    results = r.retrieve(queries, top_k=100)
    metrics = compute_metrics(qrels, results)
    return round(metrics["ndcg@10"], 4)


def run_tfidf(corpus, queries, qrels):
    r = TFIDFRetriever(max_features=100_000, sublinear_tf=True)
    r.index(corpus)
    results = r.retrieve(queries, top_k=100)
    metrics = compute_metrics(qrels, results)
    return round(metrics["ndcg@10"], 4)


def main():
    investigation = {
        "passage_length_stats": {},
        "tfidf_baseline": {},
        "b_sweep": {},   # {dataset: {b_value: ndcg@10}}
        "k1_sweep": {},  # {dataset: {k1_value: ndcg@10}}
    }

    for ds_name, split in DATASETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'='*60}")

        corpus, queries, qrels = load_dataset(ds_name, split)

        # 1. Passage length stats
        logger.info("  Computing passage length stats...")
        stats = passage_length_stats(corpus)
        investigation["passage_length_stats"][ds_name] = stats
        logger.info(f"  Length stats: mean={stats['mean']} median={stats['median']} "
                    f"std={stats['std']} min={stats['min']} max={stats['max']}")

        # 2. TF-IDF baseline
        logger.info("  Running TF-IDF baseline...")
        tfidf_score = run_tfidf(corpus, queries, qrels)
        investigation["tfidf_baseline"][ds_name] = tfidf_score
        logger.info(f"  TF-IDF NDCG@10: {tfidf_score}")

        # 3. b sweep (k₁ fixed at 1.5)
        logger.info(f"  Running b sweep {B_VALUES} (k₁=1.5)...")
        investigation["b_sweep"][ds_name] = {}
        for b in B_VALUES:
            score = run_bm25(corpus, queries, qrels, k1=1.5, b=b)
            investigation["b_sweep"][ds_name][str(b)] = score
            logger.info(f"    b={b:.2f} → NDCG@10={score:.4f}")

        # 4. k₁ sweep (b fixed at 0.75)
        logger.info(f"  Running k₁ sweep {K1_VALUES} (b=0.75)...")
        investigation["k1_sweep"][ds_name] = {}
        for k1 in K1_VALUES:
            score = run_bm25(corpus, queries, qrels, k1=k1, b=0.75)
            investigation["k1_sweep"][ds_name][str(k1)] = score
            logger.info(f"    k₁={k1:.1f} → NDCG@10={score:.4f}")

    # Save
    out_path = RESULTS_DIR / "bm25_investigation.json"
    with open(out_path, "w") as f:
        json.dump(investigation, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # Print summary tables
    print("\n" + "="*65)
    print("PASSAGE LENGTH STATS")
    print("="*65)
    print(f"{'Dataset':<15} {'Count':>8} {'Mean':>7} {'Median':>8} {'Std':>7} {'Max':>7}")
    print("-"*65)
    for ds, s in investigation["passage_length_stats"].items():
        print(f"{ds:<15} {s['count']:>8,} {s['mean']:>7.1f} {s['median']:>8.1f} "
              f"{s['std']:>7.1f} {s['max']:>7}")

    print("\n" + "="*65)
    print("b SWEEP — NDCG@10 (k₁=1.5 fixed)  |  TF-IDF baseline shown")
    print("="*65)
    header = f"{'Dataset':<15}" + "".join(f"  b={b}" for b in B_VALUES) + "  TF-IDF"
    print(header)
    print("-"*65)
    for ds in DATASETS:
        row = f"{ds:<15}"
        for b in B_VALUES:
            row += f"  {investigation['b_sweep'][ds][str(b)]:.3f}"
        row += f"  {investigation['tfidf_baseline'][ds]:.3f}"
        print(row)

    print("\n" + "="*65)
    print("k₁ SWEEP — NDCG@10 (b=0.75 fixed)")
    print("="*65)
    header = f"{'Dataset':<15}" + "".join(f"  k₁={k}" for k in K1_VALUES)
    print(header)
    print("-"*65)
    for ds in DATASETS:
        row = f"{ds:<15}"
        for k1 in K1_VALUES:
            row += f"   {investigation['k1_sweep'][ds][str(k1)]:.3f}"
        print(row)


if __name__ == "__main__":
    main()
