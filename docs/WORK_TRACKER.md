# Work Tracker — Retrieval Evolution Study

**Project:** Evolution of retrieval mechanisms — sparse to dense to hybrid  
**Started:** 2026-04-16  
**Virtual environment:** `retrieval_exp_env`  
**Hardware:** Apple M5 (MPS backend)

---

## Project Goal

Trace the evolution of information retrieval through a challenge → solution narrative. Evaluate each generation on five open-source benchmarks with standard IR metrics and validate findings with LLM-as-judge (RAGAS + local Ollama LLM).

---

## Phase Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Repo setup — structure, configs, scripts skeleton | Complete |
| **Phase 1** | Sparse retrieval — BOW, TF-IDF, BM25 | Complete |
| **Phase 2** | Early neural — Word2Vec averaging | Complete |
| **Phase 3** | Dense retrieval — SBERT, DPR, MiniLM, MPNet, BGE, E5 | Complete |
| **Phase 4** | Hybrid + reranking — RRF fusion + cross-encoder reranker | Complete |
| **Phase 5** | LLM-as-judge — RAGAS + Ollama (context_precision) | Complete |
| **Phase 6** | Fine-tuning encoders (triplet loss / contrastive loss) | Complete |

---

## Phase 0 — Repository Setup

### Tasks
- [x] Create repo structure (configs, scripts, src, notebooks, data)
- [x] Write README.md with abstract
- [x] Write WORK_TRACKER.md
- [x] Create requirements.txt and install in retrieval_exp_env
- [x] Create configs/ (datasets.yaml, models.yaml, eval.yaml)
- [x] Create src/ skeleton (sparse.py, dense.py, metrics.py, llm_judge.py)
- [x] Create scripts/ (download_datasets.py, run_*.py)
- [x] Create notebook skeletons (01–06)
- [x] Verify: `python scripts/download_datasets.py --list` runs cleanly
- [ ] Download datasets: `python scripts/download_datasets.py`
- [ ] Verify: `mlflow ui` starts at localhost:5000
- [x] **Create `notebooks/00_eda_datasets.ipynb`** — EDA across all 3 downloaded datasets (SciFact, FIQA, TREC-COVID). Should cover: corpus size + passage length distribution, query length distribution, qrels relevance label distribution (binary vs graded), train/test split sizes, vocabulary overlap between datasets, sample passages + queries per dataset. Prerequisite for understanding dataset-specific behaviour observed in experiments (e.g. BM25 vs TF-IDF inversion).

---

## Phase 1 — Sparse Retrieval

### Methods
- BOW (CountVectorizer + cosine similarity)
- TF-IDF (TfidfVectorizer + cosine similarity)
- BM25 (BM25Okapi via rank_bm25)

### Challenge callouts
- **BOW → TF-IDF**: Show score distribution where common words dominate BOW ranking
- **TF-IDF → BM25**: Show effect of k₁ (saturation) and b (length normalisation) on per-query score variance
- **BM25 ceiling**: Show query where synonym paraphrase gets 0 score (motivates dense retrieval)

### Tasks
- [x] Run `python scripts/run_sparse_eval.py --datasets scifact fiqa trec-covid`
- [ ] Run on MS MARCO + NQ (large datasets — not yet downloaded)
- [ ] Complete notebook `01_sparse_retrieval.ipynb`
- [x] Log all runs to MLflow

### Expected Results (approximate, from BEIR leaderboard)
| Method | TREC-COVID | FIQA | SciFact | NQ |
|--------|-----------|------|---------|-----|
| BM25 | ~0.656 | ~0.236 | ~0.665 | ~0.305 |
| TF-IDF | ~0.50 | ~0.18 | ~0.58 | ~0.26 |

### Open Questions

**Q: Why does TF-IDF outperform BM25 on SciFact and FIQA?**

Observed across the Phase 1 run (3 datasets). Results:

| Method | TREC-COVID | FIQA | SciFact |
|--------|-----------|------|---------|
| TF-IDF | 0.287 | **0.179** | **0.629** |
| BM25 | **0.447** | 0.159 | 0.560 |

**Finding:** TF-IDF > BM25 on SciFact AND FIQA, but BM25 > TF-IDF on TREC-COVID. Not SciFact-specific.

**Working hypothesis:** BM25's length normalisation (b=0.75) is calibrated for longer, more variable passages (MS MARCO style). SciFact and FIQA have shorter, more uniform passages where length norm doesn't help and may penalise borderline-relevant longer docs. TREC-COVID biomedical abstracts are longer and more variable — length norm kicks in usefully there.

**Investigation results** (`scripts/investigate_bm25_hyperparams.py`):

Passage length stats:
| Dataset | Mean | Median | Std | CV (std/mean) |
|---------|------|--------|-----|--------------|
| SciFact | 214.6 | 204.0 | 87.3 | 0.41 |
| FIQA | 132.9 | 90.0 | 128.7 | 0.97 |
| TREC-COVID | 161.2 | 168.0 | 135.0 | 0.84 |

b sweep (k₁=1.5 fixed) — NDCG@10:
| Dataset | b=0.0 | b=0.25 | b=0.5 | b=0.75 | b=1.0 | TF-IDF |
|---------|-------|--------|-------|--------|-------|--------|
| scifact | 0.547 | 0.558 | 0.556 | 0.560 | 0.563 | **0.629** |
| fiqa | 0.099 | 0.147 | 0.162 | 0.159 | 0.039 | **0.179** |
| trec-covid | 0.337 | 0.460 | **0.479** | 0.447 | 0.317 | 0.287 |

k₁ sweep (b=0.75 fixed) — NDCG@10:
| Dataset | k₁=0.5 | k₁=1.0 | k₁=1.5 | k₁=2.0 | k₁=2.5 | TF-IDF |
|---------|--------|--------|--------|--------|--------|--------|
| scifact | 0.559 | 0.563 | 0.560 | 0.562 | 0.567 | **0.629** |
| fiqa | 0.162 | 0.163 | 0.159 | 0.156 | 0.151 | **0.179** |
| trec-covid | 0.424 | 0.437 | **0.447** | 0.438 | 0.441 | 0.287 |

**Conclusion:** The TF-IDF advantage on SciFact and FIQA is **structural, not a hyperparameter problem**. Even b=0.0 (zero length normalisation) doesn't close the gap. The root cause: TF-IDF uses cosine similarity, which L2-normalises document vectors implicitly — a soft, global length normalisation that outperforms BM25's explicit term-level formula on these shorter, denser corpora. BM25's explicit length norm (b) is tuned for longer, more variable documents (MS MARCO style). Also note: best BM25 on TREC-COVID is at b=0.5, not the default 0.75.

---

## Phase 2 — Early Neural Retrieval

### Methods
- Word2Vec (Google News 300d) with mean pooling
- Word2Vec with IDF-weighted pooling (marginal improvement)

### Shortlisted Extensions (Phase 2+)
| Model | Description | Status |
|-------|-------------|--------|
| **Doc2Vec PV-DBOW** | Trains dedicated document-level vector (Le & Mikolov, 2014). No averaging — avoids long-passage centroid collapse. `gensim.models.Doc2Vec(dm=0)` | **Shortlisted** — skeleton in `02_early_neural_retrieval.ipynb` Section 8; implement in `scripts/run_doc2vec_eval.py` |
| **FastText** | Subword n-gram embeddings — handles OOV biomedical terms (`PCNA`, `cSMAC`) that Word2Vec silently drops | Shortlisted |
| **SIF pooling** | Smooth Inverse Frequency + subtract top principal component (Arora et al. 2017) — better than IDF-weighted avg | Shortlisted |

### Challenge callouts
- Show polysemy collapse: "bank" (river) ≡ "bank" (finance) — same vector
- Show that averaging dilutes meaning on longer passages
- Show where it beats BM25 (synonym queries) and where it fails (long passages)

### Results (NDCG@10)
| Method | TREC-COVID | FIQA | SciFact |
|--------|-----------|------|---------|
| Word2Vec (mean) | 0.339 | 0.060 | 0.269 |
| Word2Vec (IDF) | 0.437 | 0.089 | 0.310 |

**Finding:** Word2Vec IDF-weighting helps consistently but remains below BM25 on TREC-COVID and SciFact. Both variants collapse on FIQA (conversational queries with long passages). Context-free averaging loses polysemy signal — "bank" in a finance query matches river-bank passages equally.

### Tasks
- [x] Download `word2vec-google-news-300` via gensim downloader (auto via gensim API)
- [x] Run `python scripts/run_dense_eval.py --models word2vec_mean word2vec_idf`
- [ ] Complete notebook `02_early_neural_retrieval.ipynb`

---

## Phase 3 — Dense Retrieval

### Models
| Model | Params | HF ID |
|-------|--------|-------|
| all-MiniLM-L6-v2 | 22M | `sentence-transformers/all-MiniLM-L6-v2` |
| DPR (context) | 110M | `facebook/dpr-ctx_encoder-single-nq-base` |
| all-mpnet-base-v2 | 109M | `sentence-transformers/all-mpnet-base-v2` |
| BGE-base-en-v1.5 | 109M | `BAAI/bge-base-en-v1.5` |
| E5-base-v2 | 109M | `intfloat/e5-base-v2` |

### Challenge callouts (within phase)
- DPR vs MiniLM: IR-specific fine-tuning (DPR) vs general-purpose distillation (MiniLM)
- MPNet: PLM + MLM training objective vs MLM-only (BERT/MiniLM)
- BGE / E5: Instruction prefix effect (`"query: "` / `"Represent this sentence for searching relevant passages: "`)
- Domain shift: BM25 beats dense models on TREC-COVID zero-shot (motivates hybrid)

### Results (NDCG@10)
| Model | TREC-COVID | FIQA | SciFact |
|-------|-----------|------|---------|
| MiniLM (22M) | 0.473 | 0.369 | 0.645 |
| MPNet (109M) | 0.513 | **0.500** | 0.656 |
| E5 (109M) | 0.696 | 0.399 | 0.719 |
| BGE (109M) | **0.781** | 0.406 | **0.740** |

**Key findings:**
- BGE is the strongest overall dense model
- MPNet is best on FIQA despite being smaller/older than BGE and E5 — MPNet's permuted LM objective may suit financial conversational queries better
- DPR skipped — thermal throttled on M5 (BERT encoder, no SentenceTransformer optimisation); low-priority revisit with a modern dual-encoder replacement
- All dense models comfortably beat Word2Vec; BGE beats BM25 on all 3 datasets

### Tasks
- [x] Run `python scripts/run_dense_eval.py --datasets trec-covid fiqa scifact --models minilm mpnet bge e5`
- [x] Run Word2Vec models (included in dense eval script)
- [ ] DPR: revisit with modern dual-encoder (low priority — see Decisions Log)
- [ ] Complete notebook `03_dense_retrieval.ipynb`
- [ ] Complete notebook `04_modern_embeddings.ipynb`

---

## Phase 4 — Hybrid + Reranking

### Methods
- BM25 + BGE (best dense) → Reciprocal Rank Fusion
- Two-stage: BM25+BGE (RRF) → top-100 → cross-encoder reranker → top-10
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Key finding to verify
Hybrid should consistently outperform either BM25 or BGE alone. Reranker should add ~2–3 NDCG points on NQ and TREC-COVID.

### Results (NDCG@10) — corrected 2026-04-19
| Method | TREC-COVID | FIQA | SciFact |
|--------|-----------|------|---------|
| BM25 (best sparse) | 0.447 | 0.159 | 0.560 |
| BGE (best dense) | 0.781 | 0.406 | 0.740 |
| Hybrid RRF (BM25+BGE) | 0.710 | 0.292 | 0.667 |
| Hybrid + Reranker | **0.763** | **0.374** | **0.689** |

**Corrected recall/MAP (post-reranker fix — all docs returned, not just top-10):**
| Method | MAP@100 | Recall@10 | Recall@50 | Recall@100 |
|--------|---------|-----------|-----------|------------|
| Hybrid RRF FIQA | 0.2423 | 0.3690 | 0.6255 | 0.6964 |
| Hybrid+Reranker FIQA | 0.3119 | 0.4574 | 0.6334 | 0.6965 |
| Hybrid RRF SciFact | 0.6296 | 0.7977 | 0.9343 | 0.9527 |
| Hybrid+Reranker SciFact | **0.6520** | **0.8089** | 0.9303 | 0.9527 |
| Hybrid RRF TREC-COVID | 0.0830 | 0.0189 | 0.0723 | 0.1197 |
| Hybrid+Reranker TREC-COVID | 0.0932 | 0.0209 | 0.0837 | 0.1195 |

**Key findings:**
- Hybrid RRF *underperforms* BGE alone on TREC-COVID (0.710 vs 0.781) — BM25's weakness on biomedical vocabulary drags down RRF fusion
- Hybrid RRF also underperforms BGE on FIQA (0.292 vs 0.406) — same cause: BM25 is especially weak on FIQA conversational queries
- Cross-encoder reranker consistently helps (+5 NDCG on TREC-COVID, +8 on FIQA, +2 on SciFact)
- Recall@50/100 now correctly measures recall before/after reranking — the reranker's recall@100 matches first-stage (it only reorders, not filters)
- **Conclusion:** hybrid+reranker is the production best-practice, but strong domain-specific dense retrieval (BGE on biomedical) can beat hybrid when BM25 is a weak first stage

### Tasks
- [x] Run `python scripts/run_hybrid_eval.py --datasets fiqa scifact` (corrected recall/MAP/timing)
- [x] Run `python scripts/run_hybrid_eval.py --datasets trec-covid` (5h run, batch_size=64, completed 2026-04-19 18:30)
- [ ] Run on msmarco/nq once those datasets are downloaded
- [ ] Complete notebook `05_hybrid_and_reranking.ipynb`

---

## Phase 5 — LLM-as-Judge

### Setup
- Ollama must be running: `ollama serve`
- Pull model: `ollama pull llama3.2:3b` (fastest on M5) or `ollama pull mistral:7b`
- RAGAS metric: `context_precision` (no reference answer needed)
- Sample: 100–200 (query, top-5 passages) pairs per dataset

### Key analysis
- Correlation: RAGAS context_precision vs NDCG@10 across methods
- Where they agree: factoid datasets (NQ, SciFact)
- Where they diverge: domain-shift datasets (TREC-COVID, FIQA) — annotation gaps in qrels

### Results (context_precision@5 — hybrid_rrf+reranker, llama3.2:3b judge)
| Dataset | LLM context_precision | NDCG@10 (IR metric) | Agreement |
|---------|----------------------|---------------------|-----------|
| SciFact | 0.257 (n=150) | 0.689 | **Diverges** |
| TREC-COVID | 0.628 (n=50) | 0.763 | Agrees |

**Key finding:** LLM judge aligns with IR metrics on TREC-COVID (factoid, natural-language queries) but diverges sharply on SciFact (claim-verification framing). Two causes: (1) llama3.2:3b doesn't naturally handle scientific claim verification as a relevance task; (2) SciFact qrels count tangential supporting evidence as relevant — the LLM applies stricter topical relevance. This confirms the Abstract's hypothesis: annotation gaps in qrels are visible on domain-specific, non-factoid datasets.

**Implementation note:** RAGAS `LLMContextPrecisionWithoutReference` requires a `response` column (not documented) — incompatible with pure retrieval evaluation. Replaced with a direct Ollama API prompt: for each (query, passage) pair, ask LLM "is this passage relevant? yes/no" with temperature=0. Context precision = fraction of top-5 judged relevant.

### Tasks
- [x] Verify Ollama running and model pulled (`llama3.2:3b` already present)
- [x] Run `python scripts/run_llm_judge.py --datasets scifact trec-covid --sample 150`
- [ ] Complete notebook `06_llm_as_judge.ipynb`

> **Note:** The Abstract explicitly promises LLM-as-judge validation — *"A final section uses LLM-as-judge (RAGAS + local LLM via Ollama) to validate retrieval quality beyond annotation gaps in qrels."* This phase is required to fulfil the Abstract, not optional. Keep Ollama installed and llama3.2:3b pulled on the M5.

---

## Phase 6 — Fine-tuning Encoders

### Setup
- Base model: `sentence-transformers/all-MiniLM-L6-v2` (22M params)
- Training data: FIQA train split — 14,131 (query, positive passage) pairs
- Loss: `MultipleNegativesRankingLoss` (InfoNCE with in-batch negatives)
- Trainer: `SentenceTransformerTrainer` (sentence-transformers 5.4.1)
- Hardware: Apple M5 MPS
- Training: 3 epochs, batch size 64, LR 2e-5, warmup 10%
- Duration: **6.3 minutes** (663 steps)

### Results — NDCG@10 across all fine-tuning variants

| Model | TREC-COVID | FIQA | SciFact | Train pairs | Time |
|-------|-----------|------|---------|-------------|------|
| MiniLM base | 0.4725 | 0.3687 | 0.6451 | — | — |
| MiniLM ft-FIQA | **0.4867** | **0.3964** | 0.6451 | 14,131 | 6.3 min |
| MiniLM ft-SciFact | 0.4606 | 0.3593 | **0.6949** | 919 | 1.1 min |
| MiniLM ft-Combined | **0.5027** | **0.3997** | 0.6596 | 15,050 | 6.7 min |

**Delta from base (NDCG@10):**

| Fine-tuned on | TREC-COVID | FIQA | SciFact |
|---------------|-----------|------|---------|
| FIQA | **+0.0142** | **+0.0277** | ±0.0000 |
| SciFact | -0.0119 | -0.0094 | **+0.0498** |
| Combined | **+0.0302** | **+0.0310** | +0.0145 |

**Key findings:**
- **In-domain specialisation is strong**: FIQA ft improves FIQA by +7.5%; SciFact ft improves SciFact by +7.7%
- **Transfer is domain-dependent**: FIQA ft transfers positively to TREC-COVID (open-domain QA similarity); SciFact ft does not
- **Negative transfer exists but is small**: SciFact ft loses ~1 NDCG point on FIQA and TREC-COVID — domain specialisation trades breadth for depth
- **Training efficiency**: 919 SciFact pairs in 1.1 min gives +7.7% in-domain NDCG — data efficiency far exceeds dataset size
- **Recall@10 jump on SciFact**: 0.783 → 0.865 (+8.2 points) — fine-tuning especially improves early-recall
- **Combined training generalises well**: ft-Combined improves all three datasets simultaneously (+3.0% TREC-COVID, +3.1% FIQA, +1.5% SciFact) — no negative transfer; trades peak in-domain gain for breadth
- Models saved: `models/finetuned_all-MiniLM-L6-v2_fiqa/final/`, `models/finetuned_all-MiniLM-L6-v2_scifact/final/`, `models/finetuned_all-MiniLM-L6-v2_fiqa_scifact/final/`

### Tasks
- [x] Write `scripts/run_finetune.py` with `MultipleNegativesRankingLoss` + `SentenceTransformerTrainer`
- [x] Run fine-tuning on FIQA train split (14,131 pairs, 3 epochs)
- [x] Run fine-tuning on SciFact train split (919 pairs, 5 epochs)
- [x] Evaluate base vs fine-tuned on all 3 test sets
- [x] Log results to MLflow + save comparison JSONs
- [x] Run fine-tuning on combined FIQA+SciFact (15,050 pairs, 3 epochs, 6.7 min)
- [x] Complete notebook `07_finetuning.ipynb`

---

## Repo Audit & Bug Fixes (2026-04-19)

Full audit surfaced 4 bugs fixed and 1 MLflow gap closed.

### Bugs Fixed

| Bug | Severity | File | Fix Applied |
|-----|----------|------|-------------|
| Recall@50/100 corrupted after reranking — `rerank_with_cross_encoder()` returned only top-10, making recall@k meaningless | **Critical** | `run_hybrid_eval.py` | Return all `first_stage_k` docs (not `[:final_k]`); NDCG/MRR still use top-10 naturally |
| MRR@k hardcoded as `"mrr@10"` regardless of `mrr_at` param; also computed over full top-100 instead of truncating to k | **Medium** | `metrics.py` | Truncate result dict to top-k per query before passing to pytrec_eval; iterate over `mrr_at` list |
| Hybrid `query_time_s` was only the RRF fusion step (~0.05s), not BM25+BGE retrieval time | **Medium** | `run_hybrid_eval.py` | Split BM25 and BGE index vs query timers separately; `query_time_s = bm25_query + bge_query + rrf` |
| Word2Vec OOV fallback vector hardcoded to 300 dims | **Low** | `dense.py` | `np.zeros(self.wv.vector_size)` |

### MLflow Tags Added (all scripts)

Each MLflow run now carries structured tags for dataset-level filtering:
```
phase:        "1_sparse" | "2_early_neural" | "3_dense" | "4_hybrid" | "5_llm_judge" | "6_finetune"
method_type:  "sparse" | "early_neural" | "dense" | "hybrid" | "hybrid_reranked" | "llm_judge" | "finetuned_dense"
dataset:      "scifact" | "fiqa" | "trec-covid"
model:        e.g. "bge", "hybrid_rrf", "hybrid_rrf+reranker"
```

**Usage in MLflow UI:** Filter runs by `tags.dataset = 'trec-covid'` to compare all methods on a single dataset across all phases.  
**API query:** `mlflow.search_runs(filter_string="tags.dataset = 'trec-covid'")`

### Re-runs Status (2026-04-19)
- `run_hybrid_eval.py` — **complete** (all 3 datasets; TREC-COVID completed 2026-04-19 18:30, 5h run)
- `run_sparse_eval.py` — **complete** (MRR@10 corrected; BM25 TREC-COVID mrr@10=0.7197, SciFact mrr@10=0.5242)
- `run_dense_eval.py` — **running** (PID 41044; corrected MRR@10 for all 6 dense models)

### Additional Experiments Status (2026-04-19)
- Fine-tune MiniLM on SciFact (919 pairs) — **complete** (ndcg@10: 0.6451→0.6949 on SciFact)
- Fine-tune MiniLM on FIQA (14,131 pairs) — **complete** (ndcg@10: 0.3687→0.3964 on FIQA)
- Fine-tune MiniLM on FIQA+SciFact combined — **complete** (ndcg@10: fiqa 0.3687→0.3997, scifact 0.6451→0.6596, trec-covid 0.4725→0.5027)

---

## Technical Report

- [x] `docs/report.tex` — **22 pages, compiles cleanly** (last updated 2026-04-21)

### Completed in report
| Section | Content |
|---------|---------|
| Author | Sai Krishna B \& Ravindra Babu T |
| Abstract | 4 phases + LLM-as-judge/fine-tuning sentence; corrected BGE>hybrid claim |
| Sec 2.1 Benchmarks | Dataset table (small font, 7 cols visible) + graded TC example (grade 0/1/2) |
| Sec 2.2 Metrics | DCG, NDCG, MRR, Recall formulas with variable definitions |
| Sec 2.4 Loss Functions | Contrastive, Triplet, MNR with comparison table |
| Sec 3 Setup | 15-core CPU; full software list; MLflow TODO |
| Sec 4.1 Sparse | BoW + TF-IDF + BM25 with full formulas; tf(t,d) and f(t,d) defined |
| Sec 4.3 BM25 Sweep | b-sweep and k₁-sweep tables from bm25_investigation.json |
| Sec 5.1 Word2Vec | Skip-gram + CBOW architecture (original paper figure); GloVe comparison table |
| Sec 5.2 Pooling | Mean and IDF-weighted pooling formulas |
| Sec 5.3 Findings | 5-bullet itemize list |
| Sec 6.1 Bi-encoder | Architecture equations + SBERT figure |
| Sec 6.2 Model Archs | MiniLM, MPNet, E5, BGE with figures + paragraph explanations |
| Sec 7 Hybrid | Lexical+dense motivation; RRF k=60; cross-encoder equation |
| Extension A | Fine-tuning setup + results |
| Extension B | LLM-as-judge (moved after fine-tuning) |
| Appendix Table 12 | Recall@10 + MRR@10 filled from dense_results.json |

### Open TODOs in report (tracked with `\todo{}` markers)
- MLflow dashboard screenshots (Sec 3)
- Hybrid pipeline diagram → `figures/fig_hybrid_pipeline.pdf` (Sec 7)
- Metric validity on graded TREC-COVID relevance (Sec 2.1) — MRR/Recall binarise at grade≥1

### Figures directory: `docs/figures/`
| File | Source | Used in |
|------|--------|---------|
| `fig_word2vec_orig_paper_rgb.png` | Mikolov et al. 2013 via ar5iv | Sec 5.1 |
| `fig_sbert_biencoder_rgb.png` | Reimers & Gurevych 2019 via ar5iv | Sec 6.1 |
| `fig_minilm_arch_rgb.png` | Wang et al. 2020 via ar5iv | Sec 6.2 |
| `fig_mpnet_arch.jpg` | Song et al. 2020 via Microsoft Research blog | Sec 6.2 |
| `fig_e5_arch_rgb.png` | Wang et al. 2022 via ar5iv | Sec 6.2 |
| `fig_bge_arch_rgb.png` | Xiao et al. 2023 via ar5iv | Sec 6.2 |
| `fig_word2vec_skipgram.pdf` | Wikimedia Commons (backup) | — |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-16 | New standalone repo | Separate from slm_construction_ie — different research focus (general IR vs construction IE) |
| 2026-04-16 | Use retrieval_exp_env | Dedicated env for this study — isolated from other projects |
| 2026-04-16 | BEIR datasets via beir library | Standardised loading, qrels included, no manual download |
| 2026-04-16 | MLflow for experiment tracking | Local-only, no cloud dependency, Python-native, good web UI |
| 2026-04-16 | Cross-encoder scoped to Phase 4 add-on | Completes two-stage production pattern without multiplying experiment count |
| 2026-04-16 | RAGAS context_precision only | No reference answer available in BEIR retrieval task — pointwise LLM judgment without answer text |
| 2026-04-16 | Ollama + llama3.2:3b as LLM judge | Fastest local model on M5; no API cost; RAGAS supports Ollama backend |
| 2026-04-17 | Replaced FAISS with numpy exact search | FAISS + PyTorch both ship libomp.dylib on macOS → OMP Error #15 SIGABRT; numpy dot-product is exact-equivalent and has no external threading library |
| 2026-04-17 | DPR skipped in dense eval | BERT encoder without SentenceTransformer optimisation thermal-throttles on M5 (175s/batch on TREC-COVID 171K); revisit with modern dual-encoder (low priority) |
| 2026-04-17 | Dense eval resume capability added | Incremental JSON saves after every (model, dataset) pair; script skips already-completed pairs on restart |
| 2026-04-17 | Replaced RAGAS with direct Ollama API for LLM judge | RAGAS LLMContextPrecisionWithoutReference requires a `response` column — incompatible with pure retrieval (no generated answer). Direct Ollama prompt (yes/no relevance per passage) is simpler, transparent, and version-stable |
| 2026-04-18 | retrieval_exp_env venv recreated | venv had broken shebang pointing to `retrieval_exp` (missing `_env` suffix); recreated with `python3.12 -m venv --clear`; requirements.txt cleaned of unused deps (faiss-cpu, ragas, langchain) |
| 2026-04-19 | MRR@k fix: truncate before pytrec_eval | pytrec_eval's `recip_rank` has no cutoff; previously computed MRR over all 100 retrieved docs (labelled "mrr@10"). Fix: truncate result dict to top-k per query per mrr_at value. Old MRR@10 values slightly overestimated for weak models. |
| 2026-04-19 | Reranker eval: return all reranked docs | Reranker previously returned only top-10, making recall@50/100 and MAP@100 meaningless. Changed to return all first_stage_k=100 docs in reranked order so full recall metrics are valid. |
| 2026-04-19 | MLflow tags added across all scripts | Added `phase`, `method_type`, `dataset`, `model` tags. Enables `tags.dataset = 'trec-covid'` filter in MLflow UI to compare all methods on one dataset. Previously required manual param filtering. |
| 2026-04-19 | TREC-COVID hybrid eval run separately | BGE indexing 171K docs at batch_size=64 takes ~100 min; running after FIQA+SciFact avoids thermal throttling. NDCG@10 from old run (0.710/0.763) is still valid — bug only affected recall/MAP. |
| 2026-04-19 | Dense eval re-run queued after TREC-COVID hybrid | Both use MPS; sequential ordering prevents contention. MRR@10 correction is minor (~0.01) but needed for complete correctness. |
