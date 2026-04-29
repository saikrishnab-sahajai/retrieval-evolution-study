"""
IR evaluation metrics via pytrec_eval.

compute_metrics(qrels, results, k_values) → dict of metric → score
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pytrec_eval

logger = logging.getLogger(__name__)

# Metric keys understood by pytrec_eval
_METRIC_MAP = {
    "ndcg": "ndcg_cut",
    "mrr": "recip_rank",
    "map": "map_cut",
    "recall": "recall",
}


def compute_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    ndcg_at: List[int] = [10],
    mrr_at: List[int] = [10],
    map_at: List[int] = [100],
    recall_at: List[int] = [10, 50, 100],
) -> Dict[str, float]:
    """
    Compute standard IR metrics.

    Args:
        qrels:   {query_id: {doc_id: relevance_score}}  (0/1 or 0/1/2)
        results: {query_id: {doc_id: retrieval_score}}
        *_at:    list of k values for each metric

    Returns:
        Flat dict: {"ndcg@10": 0.65, "mrr@10": 0.71, ...}
    """
    # Build pytrec_eval measure set
    measures = set()
    for k in ndcg_at:
        measures.add(f"ndcg_cut_{k}")
    for k in mrr_at:
        measures.add(f"recip_rank")       # pytrec_eval computes over all k; we cut below
    for k in map_at:
        measures.add(f"map_cut_{k}")
    for k in recall_at:
        measures.add(f"recall_{k}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(results)

    # Aggregate: mean over queries
    agg: Dict[str, float] = {}

    for k in ndcg_at:
        key = f"ndcg_cut_{k}"
        scores = [per_query[qid][key] for qid in per_query if key in per_query[qid]]
        agg[f"ndcg@{k}"] = float(sum(scores) / len(scores)) if scores else 0.0

    # MRR@k — truncate results to top-k so rank stops there, then compute recip_rank.
    # pytrec_eval's recip_rank has no cutoff; we enforce it by truncating the result dict.
    for k in mrr_at:
        trunc = {
            qid: dict(sorted(v.items(), key=lambda x: x[1], reverse=True)[:k])
            for qid, v in results.items()
        }
        mrr_evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
        mrr_per_query = mrr_evaluator.evaluate(trunc)
        scores = [mrr_per_query[qid]["recip_rank"] for qid in mrr_per_query]
        agg[f"mrr@{k}"] = float(sum(scores) / len(scores)) if scores else 0.0

    for k in map_at:
        key = f"map_cut_{k}"
        scores = [per_query[qid][key] for qid in per_query if key in per_query[qid]]
        agg[f"map@{k}"] = float(sum(scores) / len(scores)) if scores else 0.0

    for k in recall_at:
        key = f"recall_{k}"
        scores = [per_query[qid][key] for qid in per_query if key in per_query[qid]]
        agg[f"recall@{k}"] = float(sum(scores) / len(scores)) if scores else 0.0

    return agg


def format_results_table(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    primary_metric: str = "ndcg@10",
) -> str:
    """
    Pretty-print a method × dataset results table.

    Args:
        all_results: {method_name: {dataset_name: {metric: score}}}
        primary_metric: metric to show in the table

    Returns:
        Formatted string table.
    """
    methods = list(all_results.keys())
    if not methods:
        return "No results."

    datasets = list(next(iter(all_results.values())).keys())
    col_w = max(len(d) for d in datasets) + 2
    row_label_w = max(len(m) for m in methods) + 2

    header = f"{'Method':<{row_label_w}}" + "".join(f"{d:>{col_w}}" for d in datasets)
    lines = [header, "-" * len(header)]

    for method in methods:
        row = f"{method:<{row_label_w}}"
        for dataset in datasets:
            score = all_results[method].get(dataset, {}).get(primary_metric, float("nan"))
            row += f"{score:>{col_w}.4f}"
        lines.append(row)

    return "\n".join(lines)
