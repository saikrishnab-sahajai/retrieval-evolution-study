# Retrieval Evolution Study

## Abstract

This report traces the full evolution of information retrieval — from Bag-of-Words to modern large embedding models — through a single guiding thread: **each generation directly addresses a failure mode of the previous one**. Starting with classical sparse methods (BOW, TF-IDF, BM25), moving through early distributional semantics (Word2Vec averaging), into transformer-based dense retrieval (SBERT, DPR, MiniLM, MPNet), and onto large-scale modern encoders (BGE, E5), the study evaluates each approach on five open-source benchmarks (MS MARCO, NQ, TREC-COVID, FIQA, SciFact) using standard IR metrics (NDCG@10, MRR@10, MAP@100, Recall@100). Hybrid retrieval (BM25 + dense + Reciprocal Rank Fusion) is evaluated as the current production best-practice; a lightweight cross-encoder reranker is added on top of the best hybrid to illustrate the two-stage production pattern. A final section uses LLM-as-judge (RAGAS + local LLM via Ollama) to validate retrieval quality beyond annotation gaps in qrels. All experiments run locally on Apple M5 (MPS backend) with no cloud or API dependency.

---

## Narrative Arc

| Method | Challenge | What the next step fixes |
|--------|-----------|--------------------------|
| BOW | No term weighting — "the" = "concrete" | TF-IDF down-weights common terms |
| TF-IDF | No saturation, no length normalisation | BM25 adds k₁ saturation + length penalty b |
| BM25 | Synonyms score 0 — purely lexical | Dense vectors capture semantic similarity |
| Word2Vec avg | Context-free; averaging loses word order | Transformer attention is contextual |
| SBERT / DPR | Single-vector bottleneck; expensive fine-tuning | Distillation (MiniLM), better objectives (MPNet) |
| MiniLM / MPNet | Fixed vector loses token-level signals | Hybrid RRF: lexical precision + semantic recall |
| Hybrid | No labels for new domains | LLM-as-judge for annotation-gap validation |

Cross-encoder reranker sits at the end of the hybrid phase as the two-stage production pattern: first-stage (BM25 + BGE) → top-100 → reranker → top-10.

---

## Repository Structure

```
retrieval_evolution_study/
├── configs/            ← datasets.yaml, models.yaml, eval.yaml
├── data/datasets/      ← auto-downloaded by scripts/download_datasets.py
├── docs/               ← WORK_TRACKER.md
├── notebooks/          ← 01–06 covering each phase
├── scripts/            ← CLI experiment runners (emit MLflow runs)
└── src/
    ├── retrievers/     ← sparse.py, dense.py
    └── evaluation/     ← metrics.py, llm_judge.py
```

---

## Quick Start

```bash
# 1. Activate environment
source ~/LearningAndDevelopment/virtualenvs/retrieval_exp_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets (writes to data/datasets/)
python scripts/download_datasets.py

# 4. Run experiments
python scripts/run_sparse_eval.py      # Phase 1 — BOW, TF-IDF, BM25
python scripts/run_dense_eval.py       # Phase 3 — SBERT, DPR, MPNet, BGE, E5
python scripts/run_hybrid_eval.py      # Phase 4 — RRF fusion + cross-encoder reranker
python scripts/run_llm_judge.py        # Phase 5 — RAGAS + Ollama

# 5. View experiment results
mlflow ui
# Open http://localhost:5000
```

---

## Results Summary

> Populated after experiments run. See MLflow dashboard for full run logs.

| Method | MS MARCO | NQ | TREC-COVID | FIQA | SciFact |
|--------|----------|----|-----------|------|---------|
| BOW | — | — | — | — | — |
| TF-IDF | — | — | — | — | — |
| BM25 | — | — | — | — | — |
| Word2Vec avg | — | — | — | — | — |
| all-MiniLM-L6-v2 | — | — | — | — | — |
| DPR | — | — | — | — | — |
| all-mpnet-base-v2 | — | — | — | — | — |
| BGE-base-en-v1.5 | — | — | — | — | — |
| E5-base-v2 | — | — | — | — | — |
| BM25 + BGE (RRF) | — | — | — | — | — |
| + Cross-encoder rerank | — | — | — | — | — |

Metric shown: NDCG@10. Full table (MRR@10, MAP@100, Recall@100, latency) in `notebooks/05`.

---

## Hardware & Environment

| Item | Value |
|------|-------|
| Device | Apple M5 — MPS backend (`torch.backends.mps`) |
| Python | 3.12 |
| PyTorch | 2.11 |
| sentence-transformers | 5.4 |
| FAISS | faiss-cpu (MPS not supported; embedding runs on MPS) |
| Virtual env | `retrieval_exp_env` |

---

## Datasets

| Dataset | Passages | Queries | Relevance | Role |
|---------|----------|---------|-----------|------|
| MS MARCO v1 (dev) | 8.8M | 6,980 | Binary | In-domain baseline (most dense models trained here) |
| NQ (Natural Questions) | 2.7M | 3,452 | Binary | Open-domain QA; paraphrase queries stress lexical methods |
| TREC-COVID | 171K | 50 | Graded 0–2 | Domain-shift test — biomedical vocabulary |
| FIQA | 57K | 648 | Graded | Financial Q&A; non-factoid long queries |
| SciFact | 5K | 300 | Binary | Claim verification; precision-focused |

All loaded via the `beir` library. Qrels (relevance labels) provided by dataset authors.

---

## Experiment Tracking

MLflow (local, no cloud):

```bash
mlflow ui      # http://localhost:5000
```

All scripts log parameters + metrics automatically. Runs stored in `mlruns/` (git-ignored).

---

## Architecture & Metrics Reference

Detailed documentation in [`docs/architectures_and_metrics.md`](docs/architectures_and_metrics.md):

- **Architecture diagrams** for every retrieval method: BOW → TF-IDF → BM25 → Word2Vec → Bi-encoder → DPR → BGE/E5 → RRF fusion → Cross-encoder reranker → LLM-as-judge pipeline
- **Metric formulas**: NDCG@k, MRR@k, MAP@k, Recall@k, RAGAS Context Precision — with worked examples
- **Model comparison table**: parameter counts, embedding dimensions, MPS inference times
