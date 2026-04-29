"""
Phase 6: Fine-tune a bi-encoder (MiniLM) on BEIR training splits.

Loss: MultipleNegativesRankingLoss (InfoNCE with in-batch negatives).
      Each batch of (query, positive_passage) pairs treats all other positives
      in the batch as negatives — simple, effective, no hard-negative mining needed.

Usage:
    python scripts/run_finetune.py                         # defaults: fiqa, MiniLM
    python scripts/run_finetune.py --dataset scifact
    python scripts/run_finetune.py --dataset fiqa scifact  # combined
    python scripts/run_finetune.py --base-model sentence-transformers/all-mpnet-base-v2
"""

import argparse
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
from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_metrics, format_results_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_beir_dataset(dataset_name: str, data_dir: Path, split: str = "test"):
    from beir.datasets.data_loader import GenericDataLoader
    dataset_path = data_dir / dataset_name
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    return GenericDataLoader(data_folder=str(dataset_path)).load(split=split)


def build_training_pairs(
    dataset_name: str, data_dir: Path
) -> list[dict]:
    """
    Load BEIR training qrels and build (anchor, positive) pairs.
    Filters to relevance >= 1 (binary or graded).
    """
    from beir.datasets.data_loader import GenericDataLoader

    dataset_path = data_dir / dataset_name
    train_qrels_path = dataset_path / "qrels" / "train.tsv"
    if not train_qrels_path.exists():
        logger.warning(f"No training qrels for {dataset_name} — skipping")
        return []

    # Load corpus and queries
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(dataset_path)
    ).load(split="train")

    pairs = []
    for qid, doc_rels in qrels.items():
        q_text = queries.get(qid, "")
        if not q_text:
            continue
        for doc_id, rel in doc_rels.items():
            if rel < 1:
                continue
            doc = corpus.get(doc_id, {})
            passage = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            if passage:
                pairs.append({"anchor": q_text, "positive": passage})

    logger.info(f"  {dataset_name} train: {len(pairs):,} (query, positive) pairs")
    return pairs


def evaluate_model(model, datasets_to_eval: dict, eval_cfg: dict) -> dict:
    """Run retrieval eval on all test sets and return metrics dict."""
    import numpy as np

    results = {}
    for ds_name, (corpus, queries, qrels) in datasets_to_eval.items():
        logger.info(f"  Evaluating on {ds_name} ({len(corpus):,} docs)...")

        # Encode corpus
        doc_ids = list(corpus.keys())
        texts = [
            (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
            for d in doc_ids
        ]
        doc_matrix = model.encode(
            texts, batch_size=128, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)

        # Encode queries
        q_ids = list(queries.keys())
        q_texts = list(queries.values())
        q_embs = model.encode(
            q_texts, batch_size=128, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)

        # Exact search
        top_k = eval_cfg["retrieval_top_k"]
        score_matrix = q_embs @ doc_matrix.T
        retrieval_results = {}
        for i, qid in enumerate(q_ids):
            row = score_matrix[i]
            idx = np.argpartition(row, -top_k)[-top_k:]
            idx = idx[np.argsort(row[idx])[::-1]]
            retrieval_results[qid] = {doc_ids[j]: float(row[j]) for j in idx}

        metrics = compute_metrics(
            qrels, retrieval_results,
            ndcg_at=eval_cfg["metrics"]["ndcg_at"],
            mrr_at=eval_cfg["metrics"]["mrr_at"],
            map_at=eval_cfg["metrics"]["map_at"],
            recall_at=eval_cfg["metrics"]["recall_at"],
        )
        results[ds_name] = metrics
        logger.info(f"    {ds_name}: NDCG@10={metrics['ndcg@10']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune bi-encoder on BEIR training splits")
    parser.add_argument("--dataset", nargs="+", default=["fiqa"],
                        help="Training dataset(s): fiqa, scifact, or both")
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

    # Load configs
    with open(REPO_ROOT / "configs" / "datasets.yaml") as f:
        ds_cfg = yaml.safe_load(f)
    with open(REPO_ROOT / "configs" / "eval.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    data_dir = REPO_ROOT / ds_cfg["data_dir"]
    results_dir = REPO_ROOT / eval_cfg.get("results_dir", "results")
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # MLflow
    mlflow.set_tracking_uri(str(REPO_ROOT / eval_cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(eval_cfg["mlflow"]["experiment_name"])

    # -------------------------------------------------------------------------
    # Build training data
    # -------------------------------------------------------------------------
    all_pairs = []
    for ds_name in args.dataset:
        pairs = build_training_pairs(ds_name, data_dir)
        all_pairs.extend(pairs)

    if not all_pairs:
        logger.error("No training pairs found. Check dataset names and training splits.")
        sys.exit(1)

    logger.info(f"Total training pairs: {len(all_pairs):,}")
    train_dataset = Dataset.from_list(all_pairs)

    # -------------------------------------------------------------------------
    # Load datasets for evaluation
    # -------------------------------------------------------------------------
    logger.info("Loading test datasets for evaluation...")
    datasets_to_eval = {}
    for ds_name in ["scifact", "fiqa", "trec-covid"]:
        ds_meta = ds_cfg["datasets"].get(ds_name)
        if ds_meta and (data_dir / ds_name).exists():
            corpus, queries, qrels = load_beir_dataset(
                ds_meta["beir_name"], data_dir, ds_meta.get("split", "test")
            )
            datasets_to_eval[ds_name] = (corpus, queries, qrels)

    # -------------------------------------------------------------------------
    # Evaluate base model (before fine-tuning)
    # -------------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer

    logger.info(f"\nLoading base model: {args.base_model}")
    base_model = SentenceTransformer(args.base_model, device=device)

    logger.info("Evaluating base model...")
    base_results = evaluate_model(base_model, datasets_to_eval, eval_cfg)

    # -------------------------------------------------------------------------
    # Fine-tune
    # -------------------------------------------------------------------------
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    train_name = "_".join(sorted(args.dataset))
    model_save_name = f"finetuned_{Path(args.base_model).name}_{train_name}"
    output_dir = models_dir / model_save_name

    loss = MultipleNegativesRankingLoss(base_model)

    steps_per_epoch = len(all_pairs) // args.batch_size
    warmup_steps = int(steps_per_epoch * args.epochs * args.warmup_ratio)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        fp16=False,         # MPS doesn't support fp16
        bf16=False,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",   # we log to MLflow manually
    )

    trainer = SentenceTransformerTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    logger.info(f"\nFine-tuning {args.base_model} on {train_name}...")
    logger.info(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
    logger.info(f"  Steps/epoch: {steps_per_epoch}  Warmup: {warmup_steps}")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    logger.info(f"Training complete in {train_time / 60:.1f} min")

    # Save final model
    base_model.save(str(output_dir / "final"))
    logger.info(f"Model saved to {output_dir / 'final'}")

    # -------------------------------------------------------------------------
    # Evaluate fine-tuned model
    # -------------------------------------------------------------------------
    logger.info("\nEvaluating fine-tuned model...")
    ft_results = evaluate_model(base_model, datasets_to_eval, eval_cfg)

    # -------------------------------------------------------------------------
    # Log to MLflow + save results
    # -------------------------------------------------------------------------
    run_name = f"finetune_{model_save_name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "phase": "6_finetune",
            "method_type": "finetuned_dense",
            "model": model_save_name,
            "train_datasets": train_name,
        })
        mlflow.log_params({
            "base_model": args.base_model,
            "train_datasets": train_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_pairs": len(all_pairs),
            "train_time_min": round(train_time / 60, 1),
        })
        # Log base model metrics with "base_" prefix for side-by-side comparison
        for ds_name, metrics in base_results.items():
            mlflow.log_metrics(
                {f"base_{ds_name}_{k.replace('@', '_at_')}": v for k, v in metrics.items()}
            )
        for ds_name, metrics in ft_results.items():
            mlflow.log_metrics(
                {f"ft_{ds_name}_{k.replace('@', '_at_')}": v for k, v in metrics.items()}
            )
        # Log deltas directly
        for ds_name in ft_results:
            base_ndcg = base_results.get(ds_name, {}).get("ndcg@10", 0.0)
            ft_ndcg = ft_results.get(ds_name, {}).get("ndcg@10", 0.0)
            mlflow.log_metric(f"delta_{ds_name}_ndcg_at_10", round(ft_ndcg - base_ndcg, 4))

    # Comparison table
    comparison = {
        f"base_{Path(args.base_model).name}": base_results,
        model_save_name: ft_results,
    }
    out_path = results_dir / f"finetune_{model_save_name}.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    print("\n" + "=" * 65)
    print(f"Fine-tuning Results — {model_save_name}")
    print("=" * 65)
    print(format_results_table(comparison, primary_metric="ndcg@10"))

    print("\nNDCG@10 delta (fine-tuned vs base):")
    for ds in ft_results:
        base_ndcg = base_results.get(ds, {}).get("ndcg@10", 0)
        ft_ndcg = ft_results.get(ds, {}).get("ndcg@10", 0)
        arrow = "+" if ft_ndcg >= base_ndcg else ""
        print(f"  {ds:<20} {arrow}{ft_ndcg - base_ndcg:+.4f}  "
              f"({base_ndcg:.4f} → {ft_ndcg:.4f})")


if __name__ == "__main__":
    main()
