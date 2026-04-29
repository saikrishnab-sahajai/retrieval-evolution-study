# Encoder Architectures & Evaluation Metrics

Reference document for all retrieval methods, encoder architectures, and IR evaluation metrics used in this study.

---

## Table of Contents

1. [Sparse Retrieval Architectures](#1-sparse-retrieval-architectures)
   - [Bag-of-Words (BOW)](#bag-of-words-bow)
   - [TF-IDF](#tf-idf)
   - [BM25](#bm25-okapi-bm25)
2. [Early Neural Architectures](#2-early-neural-architectures)
   - [Word2Vec Averaging](#word2vec-averaging)
3. [Dense Retrieval Architectures](#3-dense-retrieval-architectures)
   - [Bi-Encoder (SBERT / MiniLM / MPNet)](#bi-encoder-sbert--minilm--mpnet)
   - [DPR — Dual-Encoder](#dpr--dual-encoder)
   - [BGE & E5 (Instruction-tuned Encoders)](#bge--e5-instruction-tuned-encoders)
4. [Hybrid & Reranking Architectures](#4-hybrid--reranking-architectures)
   - [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf)
   - [Cross-Encoder Reranker](#cross-encoder-reranker)
5. [LLM-as-Judge Architecture](#5-llm-as-judge-architecture)
6. [Evaluation Metrics](#6-evaluation-metrics)
   - [NDCG@k](#ndcgk--normalised-discounted-cumulative-gain)
   - [MRR@k](#mrrk--mean-reciprocal-rank)
   - [MAP@k](#mapk--mean-average-precision)
   - [Recall@k](#recallk)
   - [RAGAS Context Precision](#ragas-context-precision)

---

## 1. Sparse Retrieval Architectures

### Bag-of-Words (BOW)

**Core idea**: represent each document as a count vector over the vocabulary. No term weighting — every word is equally important.

```
Query: "steel reinforcement bar"
                    ↓
            Tokenise + lower-case
                    ↓
    Vocabulary:  [bar, concrete, reinforcement, steel, the, ...]
                    ↓
    Query vec:   [1,   0,        1,             1,     0,   ...]
    Doc A vec:   [2,   0,        3,             2,     5,   ...]
    Doc B vec:   [0,   4,        0,             0,     3,   ...]
                    ↓
            cosine_similarity(query_vec, doc_vec)
                    ↓
               Ranked list
```

**Formula**: score(q, d) = cos(v_q, v_d)  where v_i ∈ ℕ^|V|

**Failure mode**: stopwords ("the", "is") get the same weight as domain terms ("reinforcement"). High-frequency generic terms dominate ranking.

---

### TF-IDF

**Core idea**: down-weight common terms (IDF) and up-weight terms that appear frequently in the document but rarely in the corpus (TF × IDF).

```
Query: "steel reinforcement bar"
                    ↓
            Tokenise + lower-case
                    ↓
    For each term t in document d:
    ┌─────────────────────────────────────────────────────┐
    │  TF(t, d)  = log(1 + count(t, d))  [sublinear TF]  │
    │  IDF(t)    = log(N / df(t)) + 1    [smooth IDF]    │
    │  TF-IDF(t) = TF(t,d) × IDF(t)                      │
    └─────────────────────────────────────────────────────┘
                    ↓
    TF-IDF vectors for query and each document
                    ↓
            cosine_similarity(q_tfidf, d_tfidf)
                    ↓
               Ranked list
```

**Formula**:

```
TF(t, d)    = log(1 + f(t,d))
IDF(t)      = log(N / df(t)) + 1
score(q, d) = Σ_{t ∈ q∩d}  TF(t,d) × IDF(t)  [cosine-normalised]
```

where:
- `f(t, d)` = raw frequency of term t in document d
- `N` = total number of documents in corpus
- `df(t)` = number of documents containing term t

**Fixes BOW's problem**: "the" has near-zero IDF (appears in all docs) → near-zero TF-IDF weight.

**New failure mode**: TF grows without bound (no saturation). A document with 100 mentions of "steel" scores 10× higher than one with 10 mentions — likely not 10× more relevant.

---

### BM25 (Okapi BM25)

**Core idea**: add term frequency **saturation** (k₁) and document **length normalisation** (b) to the scoring function.

```
Query: "steel reinforcement bar"
                    ↓
    For each term t in query q:
    ┌────────────────────────────────────────────────────────────────────┐
    │                           f(t,d) · (k₁ + 1)                      │
    │  score(t,d) = IDF(t) × ─────────────────────────────────────────  │
    │                          f(t,d) + k₁·(1 - b + b·(|d|/avgdl))    │
    └────────────────────────────────────────────────────────────────────┘
                    ↓
    score(q, d) = Σ_{t ∈ q∩d}  score(t, d)
                    ↓
               Ranked list
```

**Formula parameters**:
- `k₁ ∈ [1.2, 2.0]` — TF saturation constant (default 1.5). Higher k₁ = slower saturation.
- `b ∈ [0, 1]` — length normalisation strength (default 0.75). b=0 = no length normalisation; b=1 = full normalisation.
- `|d|` — document length in tokens
- `avgdl` — average document length in the corpus

**Effect of k₁ saturation**: as f(t,d) → ∞, BM25 term score → IDF(t)·(k₁+1). TF-IDF score grows unboundedly.

**Effect of b normalisation**: long documents are penalised proportionally. Prevents a 10-page document from dominating purely because it mentions more terms.

**Fundamental ceiling**: BM25 is purely lexical. A query "reinforced concrete column" gets score 0 against a passage that says "RC column" — the abbreviation shares no tokens with the query. This motivates dense retrieval.

---

## 2. Early Neural Architectures

### Word2Vec Averaging

**Core idea**: map each token to a dense 300-dimensional vector (trained via skip-gram/CBOW on Google News), then average token vectors to get a document embedding.

```
Query: "steel reinforcement bar"
              ↓
    Tokenise: ["steel", "reinforcement", "bar"]
              ↓
    ┌──────────────────────────────────────────────┐
    │  Lookup word vectors:                        │
    │    e("steel")         = [0.32, -0.11, ...]  │
    │    e("reinforcement") = [0.28,  0.04, ...]  │
    │    e("bar")           = [0.15, -0.22, ...]  │
    └──────────────────────────────────────────────┘
              ↓
    Mean pooling:  q_vec = (1/3) Σ e(token)
              ↓
    Same for each document passage
              ↓
    cosine_similarity(q_vec, d_vec)   [or dot product]
              ↓
           Ranked list

Variant — IDF-weighted pooling:
    q_vec = Σ  IDF(token) · e(token)  /  Σ IDF(token)
```

**What it fixes over BM25**: "RC" and "reinforced concrete" map to nearby vectors in embedding space → non-zero similarity even with zero token overlap.

**Failure modes**:
1. **Polysemy collapse**: "bank" (river bank) and "bank" (financial institution) share a single vector — their contexts are merged.
2. **Averaging dilutes meaning**: longer passages produce "average" embeddings that lose the key distinction between mentions.
3. **No word-order awareness**: "dog bites man" ≡ "man bites dog" after averaging.
4. **Out-of-vocabulary**: words not in the Google News corpus are silently dropped.

---

## 3. Dense Retrieval Architectures

### Bi-Encoder (SBERT / MiniLM / MPNet)

**Core idea**: encode query and document **independently** into a single dense vector using a shared or separate transformer. Similarity at retrieval time is a single dot product or cosine.

```
          QUERY                         DOCUMENT
             │                               │
    ┌────────┴────────┐           ┌──────────┴──────────┐
    │  Transformer    │           │   Transformer        │
    │  (BERT / MPNet) │           │   (same weights)     │
    └────────┬────────┘           └──────────┬──────────┘
             │                               │
    [CLS, t1, t2, ...]              [CLS, t1, t2, ...]
             │                               │
    ┌────────┴────────┐           ┌──────────┴──────────┐
    │  Pooling        │           │   Pooling            │
    │  (mean / [CLS]) │           │   (mean / [CLS])     │
    └────────┬────────┘           └──────────┬──────────┘
             │                               │
          q_vec ∈ ℝ^d                   d_vec ∈ ℝ^d
             │                               │
             └──────────────┬────────────────┘
                            │
                    sim = q_vec · d_vec
                    (after L2 normalisation → cosine)
                            │
                     Score for ranking
```

**Training objective** (contrastive / multiple negatives ranking):
```
L = -log [ exp(sim(q, p+) / τ) / Σ_{j} exp(sim(q, pj) / τ) ]
```
where p+ is the positive passage and pj are in-batch negatives.

**Key models in this study**:

| Model | Params | Dim | Training Objective | Notes |
|-------|--------|-----|--------------------|-------|
| `all-MiniLM-L6-v2` | 22M | 384 | Distillation from larger model | 6-layer, fast inference |
| `all-mpnet-base-v2` | 109M | 768 | PLM + MLM combined objectives | Permuted language modelling |
| `BAAI/bge-base-en-v1.5` | 109M | 768 | Large-scale contrastive + instruction | Instruction prefix for queries |
| `intfloat/e5-base-v2` | 109M | 768 | Weak supervision (1B+ text pairs) | "query:" / "passage:" prefixes |

**Indexing at scale**: embed all corpus documents offline → store in FAISS flat index → at query time, embed query, compute ANN search.

```
Offline (index):                     Online (query):
┌────────────┐                       ┌────────────┐
│  Corpus    │                       │  Query     │
│  (N docs)  │                       │            │
└─────┬──────┘                       └─────┬──────┘
      │  Bi-encoder                        │  Bi-encoder
      ↓                                    ↓
 [d₁, d₂, ... dN]                      q_vec
      │                                    │
      ↓  FAISS IndexFlatIP                 │
 Build index                               │
      │                                    │
      └───────────── search(q_vec, top_k) ─┘
                           │
                    Top-K doc IDs + scores
```

---

### DPR — Dual-Encoder

**Core idea**: same bi-encoder pattern, but uses **separate** BERT encoders for questions and passages, fine-tuned end-to-end on QA pairs (NQ training data).

```
     QUESTION                      PASSAGE (Context)
          │                               │
  ┌───────┴───────┐               ┌───────┴───────┐
  │  Q-encoder    │               │  P-encoder    │
  │  (BERT-base)  │               │  (BERT-base)  │
  │  fine-tuned   │               │  fine-tuned   │
  └───────┬───────┘               └───────┬───────┘
          │                               │
       [CLS] vec                       [CLS] vec
       q_enc ∈ ℝ^768                p_enc ∈ ℝ^768
          │                               │
          └──────────── dot product ───────┘
                              │
                         sim(q, p)
```

HuggingFace IDs used:
- Question encoder: `facebook/dpr-question_encoder-single-nq-base`
- Context encoder: `facebook/dpr-ctx_encoder-single-nq-base`

**vs. MiniLM**: DPR was trained specifically on NQ open-domain QA data (factoid retrieval). MiniLM was distilled from a larger general-purpose sentence encoder. On in-domain QA (NQ) DPR wins; on domain-shifted data (TREC-COVID, SciFact) the general-purpose distillation often generalises better.

---

### BGE & E5 (Instruction-tuned Encoders)

**Core idea**: use an **instruction prefix** prepended to the query at inference time to tell the encoder what kind of search is being performed. The encoder was trained with these prefixes, so it adjusts its representation accordingly.

```
BGE query encoding:
   "Represent this sentence for searching relevant passages: <query text>"
                    ↓
          BERT-base transformer
                    ↓
           Mean pool → L2 normalise
                    ↓
              q_vec ∈ ℝ^768

BGE passage encoding:
   "<passage text>"        ← no prefix for documents
                    ↓
          BERT-base transformer (same weights)
                    ↓
           Mean pool → L2 normalise
                    ↓
              p_vec ∈ ℝ^768


E5 query encoding:
   "query: <query text>"
                    ↓
          BERT-base transformer
                    ↓
             CLS pool → L2 normalise

E5 passage encoding:
   "passage: <passage text>"
                    ↓
          BERT-base transformer (same weights)
                    ↓
             CLS pool → L2 normalise
```

**Training scale difference** (BGE vs MiniLM):
- MiniLM: distilled from mpnet on ~1M sentence pairs
- BGE: mined from large-scale web corpora, hard-negative mining + instruction-following fine-tuning on curated datasets
- E5: trained on ≥1B weakly supervised text pairs from Common Crawl

---

## 4. Hybrid & Reranking Architectures

### Reciprocal Rank Fusion (RRF)

**Core idea**: merge multiple ranked lists without needing to normalise their incompatible score scales. Each document's final score is the sum of its reciprocal ranks across all input rankers.

```
BM25 ranked list          BGE ranked list
────────────────          ───────────────
1. doc_A (0.823)          1. doc_C (0.912)
2. doc_B (0.712)          2. doc_A (0.891)
3. doc_C (0.544)          3. doc_E (0.843)
4. doc_D (0.321)          4. doc_B (0.801)
5. doc_E (0.188)          5. doc_D (0.754)
        ↓                         ↓
        └──────────────────────────┘
                   │
         RRF(k=60) score:
         ┌─────────────────────────────────────────────────────┐
         │  RRF_score(doc) = Σᵢ  1 / (k + rankᵢ(doc))        │
         │                                                     │
         │  doc_A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613   │
         │       = 0.03252                                     │
         │  doc_C: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639   │
         │       = 0.03226                                     │
         └─────────────────────────────────────────────────────┘
                   │
         RRF fused ranked list:
         1. doc_A (0.03252)   ← top in BM25, 2nd in BGE
         2. doc_C (0.03226)   ← 3rd in BM25, 1st in BGE
         3. doc_B (0.03205)   ...
         ...
```

**Formula**:
```
RRF_score(d) = Σ_{r ∈ rankers}  1 / (k + rank_r(d))
```

- `k = 60` (standard; prevents top-ranked docs from having extreme scores)
- Documents not in a ranker's top-N are treated as rank → ∞ (score contribution ≈ 0)

**Why it works**: BM25 is precise but misses synonyms; dense retrieval covers semantic similarity but may miss exact-match queries. RRF rewards documents that rank well in *both* — they are likely genuinely relevant.

---

### Cross-Encoder Reranker

**Core idea**: unlike a bi-encoder (encodes query and document independently), a cross-encoder receives the concatenated `[CLS] query [SEP] passage [SEP]` and applies **full self-attention across both**. This produces a relevance score, not an embedding.

```
FIRST STAGE (recall):   BM25 + BGE → RRF → top-100 candidates
                                 │
                         ┌───────┴───────┐
                         │  Cross-Encoder │
                         │  reranking    │
                         └───────┬───────┘

Cross-encoder scoring for each (query, doc) pair:
                                 │
    [CLS] query tokens [SEP] passage tokens [SEP]
                  │
     ┌────────────┴────────────────────────────────┐
     │        Full Self-Attention (12 layers)      │
     │    Every query token attends to every       │
     │    passage token and vice versa             │
     └────────────┬────────────────────────────────┘
                  │
               [CLS] hidden state
                  │
             Linear → sigmoid
                  │
            Relevance score ∈ [0, 1]

Applied to top-100 candidates → re-sort → return top-10
```

Model used: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Why cross-encoder > bi-encoder for precision**: full cross-attention allows the model to compare query and passage tokens directly. Bi-encoder only sees the query *or* the passage — the two representations meet only at the dot-product step, losing fine-grained interactions.

**Why cross-encoder can't be the first stage**: scoring N×M query-document pairs at query time is O(N × corpus_size) — impractical at millions of documents. That's why the pipeline is two-stage:
1. First stage: fast bi-encoder + FAISS → top-100 candidates (milliseconds)
2. Second stage: cross-encoder scores only top-100 → top-10 (seconds per query)

---

## 5. LLM-as-Judge Architecture

```
     Query                   Top-5 retrieved passages
       │                              │
       └──────────────────────────────┘
                      │
              Prompt template:
       ┌───────────────────────────────────────────────┐
       │  "Given the question: {query}                 │
       │   and the following context passages:         │
       │   [1] {passage_1}                             │
       │   [2] {passage_2}  ...                        │
       │                                               │
       │   For each passage, indicate whether it       │
       │   contains information relevant to answering  │
       │   the question. Answer: [Yes/No, Yes/No, ...]"│
       └───────────────────────────────────────────────┘
                      │
            Local LLM (llama3.2:3b via Ollama)
            ─────────────────────────────────
            Running at http://localhost:11434
                      │
            [Yes, No, Yes, Yes, No]   ← per-passage verdicts
                      │
         RAGAS Context Precision =
         ┌───────────────────────────────────────────────────────────────┐
         │  mean over positions of: (cumulative precision up to pos k)   │
         │  where only relevant passages contribute to the numerator     │
         └───────────────────────────────────────────────────────────────┘
                      │
              scalar ∈ [0, 1]
```

**RAGAS `LLMContextPrecisionWithoutReference`**: computes a ranked precision signal — relevant passages appearing at the top of the retrieved list contribute more to the score than relevant passages appearing later.

---

## 6. Evaluation Metrics

### NDCG@k — Normalised Discounted Cumulative Gain

Measures ranking quality with **graded relevance** (0, 1, 2). Positions are discounted logarithmically — a relevant document at rank 3 contributes less than the same document at rank 1.

**Step 1 — DCG@k** (Discounted Cumulative Gain):
```
        k     rel(i)
DCG@k = Σ  ──────────
       i=1  log₂(i+1)

where rel(i) = relevance score of the document at rank i
               (0 = not relevant, 1 = relevant, 2 = highly relevant)
```

**Step 2 — IDCG@k** (Ideal DCG): DCG of the perfect ranking (best documents first).

**Step 3 — NDCG@k**:
```
NDCG@k = DCG@k / IDCG@k    ∈ [0, 1]
```

**Example** (k=3, relevance labels [2, 0, 1]):
```
DCG@3 = 2/log₂(2) + 0/log₂(3) + 1/log₂(4)
      = 2/1 + 0 + 1/2
      = 2.5

Ideal ranking would be [2, 1, 0]:
IDCG@3 = 2/1 + 1/log₂(3) + 0 = 2.631

NDCG@3 = 2.5 / 2.631 = 0.950
```

**Primary metric in this study**: NDCG@10. Standard BEIR benchmark metric — compares fairly across datasets with different relevance scales.

---

### MRR@k — Mean Reciprocal Rank

Measures how high the **first** relevant document appears in the ranking. Appropriate for tasks where a single correct answer exists (QA, fact lookup).

```
         1   Q      1
MRR@k = ─── Σ  ──────────
         Q  q=1  rank_q(1)

where rank_q(1) = rank of the first relevant document for query q
      Q = total number of queries
```

**Example** (3 queries):
```
Query 1: first relevant doc at rank 2   →  1/2 = 0.500
Query 2: first relevant doc at rank 1   →  1/1 = 1.000
Query 3: first relevant doc at rank 4   →  1/4 = 0.250

MRR@10 = (0.500 + 1.000 + 0.250) / 3 = 0.583
```

**Used for**: NQ and SciFact (single-answer factoid queries). If no relevant doc appears in top-k, contribution is 0.

---

### MAP@k — Mean Average Precision

Measures **precision at every recall point** — rewards systems that retrieve many relevant documents early. Most informative when queries have multiple relevant documents.

**Average Precision for a single query**:
```
         1    k
AP@k =  ─── Σ  P@i × rel(i)
         R  i=1

where P@i     = precision at cut-off i  (# relevant in top-i / i)
      rel(i)  = 1 if document at rank i is relevant, else 0
      R       = total number of relevant documents for this query
      k       = cut-off (typically 100)
```

**MAP@k** = mean of AP@k over all queries.

**Example** (query with 3 relevant docs, ranks 1, 3, 6):
```
P@1 × rel(1) = 1/1 × 1 = 1.000
P@2 × rel(2) = 1/2 × 0 = 0.000
P@3 × rel(3) = 2/3 × 1 = 0.667
P@4 × rel(4) = 2/4 × 0 = 0.000
P@5 × rel(5) = 2/5 × 0 = 0.000
P@6 × rel(6) = 3/6 × 1 = 0.500

AP@100 = (1.000 + 0.000 + 0.667 + 0 + 0 + 0.500) / 3 = 0.722
```

---

### Recall@k

Measures what fraction of all relevant documents are retrieved in the top-k. Critical for **RAG pipelines** — a document must be in the top-k to be passed to the generator.

```
                    |{retrieved docs in top-k} ∩ {relevant docs}|
Recall@k  =  ────────────────────────────────────────────────────
                              |{relevant docs}|
```

**Example**:
```
Relevant docs for query = {doc_A, doc_B, doc_C, doc_D}   (4 total)
Top-100 retrieved = {doc_A, doc_X, doc_C, doc_B, ...}

Recall@100 = |{doc_A, doc_C, doc_B}| / 4 = 3/4 = 0.75
```

**Values in this study**: Recall@10 (tight budget), Recall@50, Recall@100 (first-stage budget for reranker).

A reranker can only improve precision within the first-stage recall budget. If Recall@100 = 0.70, a perfect reranker can at best achieve NDCG@10 based on 70% of the truly relevant documents.

---

### RAGAS Context Precision

**What it measures**: of the passages retrieved for a query, how many are semantically relevant — as judged by a local LLM, *without* needing a reference answer.

```
                     k     precision@i × relevance(i)
ContextPrecision@k = Σ  ─────────────────────────────────
                    i=1          |{relevant passages}|

where:
  precision@i    = (# LLM-judged relevant passages in top-i) / i
  relevance(i)   = 1 if the i-th passage is LLM-judged relevant, else 0
  denominator    = total LLM-judged relevant passages in top-k
```

**Relationship to NDCG**: both reward relevant documents appearing early. NDCG uses qrel annotations (human-labeled); Context Precision uses LLM judgment (may catch unannotated relevant passages).

**Where they diverge**: on TREC-COVID and FIQA (annotation-gap datasets), LLM judgment may give higher scores to dense/hybrid methods that retrieve unlabeled-but-relevant passages that qrel-based NDCG penalises.

---

## Metric Summary Table

| Metric | Graded Relevance | Position Sensitivity | Multiple Relevant Docs | Use case |
|--------|-----------------|----------------------|------------------------|----------|
| NDCG@10 | Yes (0/1/2) | Yes — log discount | Yes | Standard BEIR benchmark |
| MRR@10 | No (first hit) | Yes — only first | No | Single-answer QA |
| MAP@100 | No (binary) | Yes — per recall point | Yes | Multi-relevant, pipeline eval |
| Recall@100 | No (binary) | No | Yes | RAG / first-stage budget |
| Context Precision | LLM-graded | Yes | Yes | Label-sparse validation |

---

## Model Size & Inference Speed Reference

| Model | Params | Embedding Dim | MPS Inference (SciFact 5K) |
|-------|--------|---------------|---------------------------|
| `all-MiniLM-L6-v2` | 22M | 384 | ~15 sec |
| `all-mpnet-base-v2` | 109M | 768 | ~40 sec |
| `facebook/dpr-ctx_encoder` | 110M | 768 | ~2 min |
| `BAAI/bge-base-en-v1.5` | 109M | 768 | ~40 sec |
| `intfloat/e5-base-v2` | 109M | 768 | ~40 sec |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | — (scores) | ~30 sec / 100 pairs |

*Timings on Apple M5, MPS backend, batch_size=128, SciFact corpus (5,183 passages).*
