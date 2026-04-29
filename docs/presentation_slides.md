# Presentation Slide Content
# Evolution of Information Retrieval: From Bag-of-Words to Hybrid Neural Systems

Format: Each slide is separated by `---`. Fields: **TITLE**, **BODY** (bullet points), **NOTES** (speaker notes / things to say aloud), **VISUAL** (what image/diagram/table to put).

---

## SLIDE 1 — Title

**TITLE:** Evolution of Information Retrieval
**SUBTITLE:** From Bag-of-Words to Hybrid Neural Systems

**BODY:**
- A practical, end-to-end study across three open-domain benchmarks
- Covering 6 generations of retrieval: BOW → TF-IDF → BM25 → Word2Vec → Dense → Hybrid + Fine-tuning

**NOTES:**
This study traces how retrieval has evolved over ~30 years — from counting words to encoding meaning in neural vector spaces. Each generation solved the previous one's fundamental problem, and the experiments make that visible with real numbers.

**VISUAL:** Clean title slide. Can use a horizontal timeline arrow: BOW → TF-IDF → BM25 → Word2Vec → Dense → Hybrid

---

## SLIDE 2 — Agenda

**TITLE:** What We'll Cover

**BODY:**
1. Datasets & Benchmarks — what we're measuring on
2. Evaluation Metrics — what the numbers mean
3. Phase 1: Sparse Retrieval (BOW, TF-IDF, BM25)
4. Phase 2: Early Neural (Word2Vec)
5. Phase 3: Dense Retrieval (SBERT, DPR, MPNet, BGE, E5)
6. Phase 4: Hybrid Retrieval (RRF + Reranker)
7. Phase 5: LLM-as-Judge Validation
8. Phase 6: Fine-tuning Encoders
9. Full Comparison & Takeaways

**VISUAL:** Simple numbered list or progress bar

---

## SLIDE 3 — Motivation

**TITLE:** Why Information Retrieval Matters Now More Than Ever

**BODY:**
- Retrieval is the foundation of every RAG (Retrieval-Augmented Generation) pipeline
- LLMs are only as good as the passages they are given
- Bad retrieval → irrelevant context → hallucination
- The field has changed dramatically in 5 years: BM25 → bi-encoders → hybrid → fine-tuned models

**KEY QUESTION:** When should you use each generation? What does each one actually fix?

**NOTES:**
Framing: this is not just academic. If you are building a chatbot, a document QA system, or a search engine, you need to choose a retrieval backend. The study gives you data to make that choice.

**VISUAL:** RAG pipeline diagram: [User Query] → [Retrieval] → [Top-K Passages] → [LLM] → [Answer]. Highlight the retrieval step.

---

## SLIDE 4 — Datasets Overview

**TITLE:** Three Benchmark Datasets — Three Different Challenges

**BODY:**
| Dataset | Domain | Corpus | Test Queries | Relevance |
|---------|--------|--------|-------------|-----------|
| TREC-COVID | Biomedical literature | 171,331 passages | 50 queries | Graded (0/1/2) |
| FIQA | Financial Q&A | 57,638 passages | 648 queries | Binary (0/1) |
| SciFact | Scientific claim verification | 5,183 passages | 300 queries | Binary (0/1) |

**BODY (continued):**
- All from the **BEIR benchmark** — standardised evaluation suite for retrieval models
- All use **test split only** — corpus is full, but queries and relevance labels are from the test set
- Three very different domains → tests generalisation, not just in-domain performance

**NOTES:**
These three datasets are chosen deliberately to stress different failure modes. TREC-COVID has huge corpus and biomedical vocabulary. FIQA has conversational, ambiguous questions. SciFact has short, precise scientific claims. A good retrieval system should handle all three.

**VISUAL:** Three dataset cards side-by-side, each showing: domain, corpus size, query count, avg passage length

---

## SLIDE 5 — Dataset Deep Dive: TREC-COVID

**TITLE:** TREC-COVID — Biomedical Literature Retrieval

**BODY:**
- **Task:** Given a clinical/research question about COVID-19, find relevant biomedical abstracts
- **Corpus:** 171,331 abstracts from CORD-19 (published COVID-19 papers)
- **Queries:** 50 — short, precise clinical/research questions (e.g., *"What are the symptoms of COVID-19?"*)
- **Relevance:** Graded — score 0 (not relevant), 1 (partially relevant), 2 (highly relevant); 10,456 positive pairs
- **Passage length:** median 168 words, mean 161 words
- **Challenge for dense models:** vocabulary is highly specialised — drug names, gene symbols, medical acronyms

**EXAMPLE QUERY:** *"What do we know about transmission, incubation, and environmental stability?"*

**NOTES:**
The 50-query setup is important — TREC assessors manually judged every retrieved result, making it a gold-standard evaluation. Very few false negatives in the qrels. Biomedical vocabulary is the primary challenge — terms like "ACE2 receptor", "RT-PCR", "cytokine storm" don't appear in general-domain training sets.

**VISUAL:** [PLACEHOLDER — architecture/data diagram showing biomedical abstract → query matching]

---

## SLIDE 6 — Dataset Deep Dive: FIQA

**TITLE:** FIQA — Financial Question Answering

**BODY:**
- **Task:** Given a financial question (from Reddit/StackExchange), find a relevant answer passage
- **Corpus:** 57,638 passages — community forum answers, financial documents
- **Queries:** 6,648 total; 648 in the test set with relevance labels
- **Relevance:** Binary (0/1) — originally graded in the FiQA challenge, BEIR binarised all qrels to score=1
- **Passage length:** median 90 words, mean 133 words (highly variable, max 2,973 words)
- **Challenge:** Conversational, informal phrasing — *"Best way to invest small amounts?"* — no single-term match

**EXAMPLE QUERY:** *"How should I invest my retirement savings given current market conditions?"*

**NOTES:**
FIQA is the hardest dataset for dense models overall. The informal phrasing means queries rarely share exact vocabulary with answers. Interestingly, MPNet outperforms newer/larger models like BGE and E5 on this dataset — possibly because MPNet's permuted LM training captured some conversational text patterns.

**VISUAL:** [PLACEHOLDER — example Q&A pair from FIQA showing informal phrasing vs formal answer passage]

---

## SLIDE 7 — Dataset Deep Dive: SciFact

**TITLE:** SciFact — Scientific Claim Verification

**BODY:**
- **Task:** Given a scientific claim (as a query), find abstracts that support or refute it
- **Corpus:** 5,183 biomedical research abstracts
- **Queries:** 1,109 total; 300 in the test set with relevance labels
- **Relevance:** Binary — 339 relevant pairs across 300 queries (~1.1 relevant passage per query on average)
- **Passage length:** median 204 words, mean 215 words (longest of the three)
- **Challenge:** Short, precise scientific claims with specialised terminology

**EXAMPLE QUERY:** *"Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164."*

**NOTES:**
SciFact is unique because the query is a claim, not a natural question. The task is to find evidence — supporting or contradicting. This means vocabulary overlap is high when terms match, but the model needs to understand the claim's precise meaning. Small corpus (5K) means many runs complete quickly — good for fast iteration.

**VISUAL:** [PLACEHOLDER — diagram showing claim as query → abstract as passage → binary relevant/not]

---

## SLIDE 8 — Dataset Stats Comparison

**TITLE:** Key Stats at a Glance

**BODY:**
| Statistic | TREC-COVID | FIQA | SciFact |
|-----------|-----------|------|---------|
| Corpus size | 171,331 | 57,638 | 5,183 |
| Test queries | 50 | 648 | 300 |
| Relevant pairs | 24,673 | 1,706 | 339 |
| Avg relevant/query | 493 | 2.6 | 1.1 |
| Passage length (median) | 168 words | 90 words | 204 words |
| Passage length (max) | 18,010 | 2,973 | 1,541 |
| Domain | Biomedical | Financial | Scientific |
| Query style | Natural language | Conversational | Precise claims |

**NOTES:**
The "avg relevant per query" stat is critical. TREC-COVID has 493 relevant passages per query because assessors found many related abstracts. FIQA has 2.6 and SciFact just 1.1. This directly affects Recall@10 — on TREC-COVID it's almost impossible to recall all relevant docs in top-10.

**VISUAL:** Bar chart comparing corpus sizes (log scale) + table above

---

## SLIDE 9 — EDA: Passage Length Distributions

**TITLE:** Passage Length Varies Widely Across Datasets

**BODY:**
- **FIQA:** Short, bimodal distribution — forum-style answers (median 90 words) but heavy tail to 2,973
- **TREC-COVID:** Medium-length abstracts (median 168 words), relatively uniform
- **SciFact:** Longest on average (median 204 words) — full research abstracts

**WHY IT MATTERS:**
- Long passages → averaging-based methods (Word2Vec, TF-IDF) degrade more
- Variable length → BM25's length normalisation term (b parameter) matters more
- Short passages → dense models have less context to work with

**NOTES:**
This came up in the BM25 investigation — TF-IDF beats BM25 on FIQA and SciFact because their passages are shorter and more uniform. BM25's length normalisation was calibrated for MS MARCO (longer, more variable). This is a real-world lesson: default hyperparameters may not be optimal for your domain.

**VISUAL:** [PLACEHOLDER — three histogram plots of passage length distributions, one per dataset]

---

## SLIDE 10 — EDA: Relevance Distributions

**TITLE:** How Relevance Is Labelled Across Datasets

**BODY:**
- **TREC-COVID (graded):** 0 = not relevant (41,661), 1 = partially relevant (10,456), 2 = highly relevant (14,217). Metrics treat 1 and 2 as relevant (score > 0).
- **FIQA (binary):** Originally graded in the FiQA 2018 challenge; BEIR binarised all 1,706 qrels to score=1
- **SciFact (binary):** 339 relevant pairs; ~1.1 relevant passage per query — very sparse

**KEY INSIGHT:** Sparse relevance (SciFact) means a model that misses the one relevant passage gets NDCG=0 for that query. High-recall models are rewarded more on SciFact than on TREC-COVID.

**VISUAL:** [PLACEHOLDER — pie charts / bar charts showing score distributions per dataset]

---

## SLIDE 11 — Evaluation Metrics

**TITLE:** What the Numbers Mean

**BODY:**
- **NDCG@10** (Normalised Discounted Cumulative Gain): Measures ranking quality in top-10. Discounts lower-ranked results — finding the relevant passage at rank 1 is better than rank 10. Our primary metric.
- **MRR@10** (Mean Reciprocal Rank): Average of 1/rank-of-first-relevant-result. Rewards finding *any* relevant result quickly.
- **Recall@10 / @100**: What fraction of all relevant passages appear in the top-10 or top-100 results. Tells you how much you can retrieve with a deeper search.
- **MAP@100** (Mean Average Precision): Average precision across all retrieved results up to rank 100.

**RULE OF THUMB:**
- NDCG@10 = *"how good is my top-10 list?"*
- MRR@10 = *"how fast do I find the first good result?"*
- Recall@100 = *"is the right answer even retrievable?"*

**VISUAL:** Simple diagram: ranked list [1][2][3]...[10] → relevant items highlighted → discount formula

---

## SLIDE 12 — Retrieval Evolution Overview

**TITLE:** Six Generations of Retrieval

**BODY:**
| Generation | Method | Core Idea | Key Limitation |
|------------|--------|-----------|----------------|
| 1 | BOW | Count word occurrences | Common words dominate |
| 2 | TF-IDF | Down-weight frequent words | No word ordering or semantics |
| 3 | BM25 | Add length normalisation + saturation | Still purely lexical — no synonyms |
| 4 | Word2Vec averaging | Map words to vectors, average | Static, context-free, domain-general |
| 5 | Dense bi-encoders | Encode full meaning in one vector | Single-vector bottleneck, domain gap |
| 6 | Hybrid + Reranker | Combine lexical + semantic + rerank | Complexity, latency |

**NOTES:**
Each row solves the row above it. This is the spine of the presentation — every experiment we ran tests whether the fix actually works. The answer is almost always "yes, on average, but with nuance."

**VISUAL:** Vertical timeline or staircase diagram showing evolution. Add approximate year: BOW (1960s), TF-IDF (1970s), BM25 (1994), Word2Vec (2013), SBERT (2019), Hybrid/BGE/E5 (2022–23)

---

## SLIDE 13 — Phase 1: Sparse Retrieval — How It Works

**TITLE:** Phase 1 — Sparse Retrieval: Counting Words

**BODY:**
**Bag of Words (BOW):**
- Build a vocabulary of all unique words in the corpus
- Represent each document and query as a vector of raw word counts
- Similarity = cosine similarity between count vectors
- Problem: "the", "is", "a" dominate every document

**TF-IDF:**
- Weight = TF (term frequency in doc) × IDF (inverse document frequency across corpus)
- IDF = log(N / df(t)) — rare terms get higher weight
- Effectively down-weights common words, up-weights discriminative terms

**BM25 (Best Match 25):**
- Adds two corrections to TF-IDF:
  - **Saturation (k₁):** Term frequency contribution plateaus — 10 occurrences isn't 10× better than 1
  - **Length normalisation (b):** Penalises long documents proportionally
- Formula: `score(q,d) = Σ IDF(t) × [TF × (k₁+1)] / [TF + k₁×(1 - b + b×|d|/avgdl)]`
- Default: k₁=1.5, b=0.75

**VISUAL:** [PLACEHOLDER — BOW: word → count matrix diagram; TF-IDF: formula; BM25: formula with k1/b annotated]

---

## SLIDE 14 — Phase 1: Sparse Results

**TITLE:** Sparse Retrieval Results — NDCG@10

**BODY:**
| Method | TREC-COVID | FIQA | SciFact |
|--------|-----------|------|---------|
| BOW | 0.176 | 0.066 | 0.365 |
| TF-IDF | 0.287 | **0.179** | **0.629** |
| BM25 | **0.447** | 0.159 | 0.560 |

**KEY FINDING — Unexpected BM25 vs TF-IDF inversion:**
- BM25 wins on TREC-COVID (longer, variable passages — length normalisation helps)
- TF-IDF wins on FIQA and SciFact (shorter, uniform passages — cosine normalisation is better)
- Best BM25 on TREC-COVID is at b=0.5, not the default b=0.75

**ROOT CAUSE:** BM25's length normalisation is calibrated for long, variable documents (MS MARCO). Shorter, denser corpora like FIQA and SciFact benefit from TF-IDF's global L2 normalisation instead.

**NOTES:**
This was one of the first surprising findings. The conventional wisdom is "BM25 is the best sparse baseline." That's true on the classic benchmarks (MS MARCO, NQ) but not on all domains. This motivates careful evaluation before assuming defaults are optimal.

**VISUAL:** Bar chart — 3 groups (TREC-COVID, FIQA, SciFact), 3 bars each (BOW, TF-IDF, BM25), highlight winner per dataset

---

## SLIDE 15 — Phase 1: BM25 Ceiling

**TITLE:** The BM25 Ceiling — What Sparse Retrieval Cannot Do

**BODY:**
**The synonym problem:** BM25 scores purely on lexical overlap. If the query says "headache" and the relevant passage says "migraine" — score is 0. No credit for meaning, only for shared terms.

**Example from SciFact:**
- Query: *"Headaches are not correlated with cognitive impairment"*
- Relevant passage uses "cephalgia" and "neurocognitive function"
- BM25 NDCG@10 = 0 for this query
- Word2Vec NDCG@10 = 0.667 for this query (bridges the vocabulary gap)

**BM25 best achievable NDCG@10:** ~0.45–0.56 on our datasets → this is the ceiling that motivates the next generation.

**VISUAL:** Side-by-side: Query text ↔ Passage text, with matching terms highlighted in green (few/none for synonym case) vs rich overlap

---

## SLIDE 16 — Phase 2: Word2Vec — Architecture

**TITLE:** Phase 2 — Early Neural Retrieval: Word2Vec

**BODY:**
**What is Word2Vec?**
- Neural network trained on ~100B words (Google News) to predict surrounding words
- Output: 300-dimensional vector for each word — semantically similar words cluster nearby
- "king" - "man" + "woman" ≈ "queen" — vector arithmetic captures analogy

**How it's used for retrieval (Mean Pooling):**
- Represent each passage: d = (1/|T|) × Σ v(t) — average all word vectors
- Represent each query: same
- Similarity = cosine similarity between averaged vectors

**IDF-Weighted Pooling (improvement):**
- d = Σ idf(t) × v(t) / Σ idf(t) — weight each word by its IDF score
- Suppresses stop words ("the", "is") that contribute noise when averaged
- Same intuition as TF-IDF but applied in vector space

**VISUAL:** [PLACEHOLDER — Word2Vec architecture: CBOW or Skip-gram diagram showing word → vector space; then averaging diagram: 3 word vectors → mean vector]

---

## SLIDE 17 — Phase 2: Word2Vec Limitations

**TITLE:** Word2Vec — Where It Helps and Where It Breaks

**BODY:**
**Where Word2Vec WINS over BM25 (synonym/paraphrase queries):**
- *"Headaches are not correlated with cognitive impairment"* → +0.667 NDCG
- *"Low nucleosome occupancy correlates with low methylation levels"* → +0.667 NDCG
- *"A deficiency of vitamin B12 increases blood levels of homocysteine"* → +0.613 NDCG
- Pattern: everyday vocabulary, meaning expressible in common English words

**Where BM25 WINS (specialised terminology, OOV problem):**
- *"Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164"* → Word2Vec loses by −1.0
- *"cSMAC formation enhances weak ligand signalling"* → Word2Vec loses by −1.0
- *"The minor G allele of FOXO3 is related to Crohn's Disease"* → Word2Vec loses by −1.0
- Pattern: precise biomedical identifiers are **out-of-vocabulary (OOV)** — silently dropped

**ROOT CAUSE:** Google News vocabulary ≠ biomedical vocabulary. OOV tokens are dropped → query has no signal → BM25 exact-match wins easily.

**VISUAL:** Two-column layout: "Word2Vec wins" examples (green) vs "BM25 wins" examples (red), with OOV terms highlighted

---

## SLIDE 18 — Phase 2: Word2Vec Results

**TITLE:** Word2Vec Results — Still Below BM25 Overall

**BODY:**
| Method | TREC-COVID | FIQA | SciFact |
|--------|-----------|------|---------|
| BM25 | **0.447** | **0.159** | **0.560** |
| Word2Vec (mean) | 0.339 | 0.060 | 0.269 |
| Word2Vec (IDF) | 0.436 | 0.089 | 0.310 |

**KEY OBSERVATIONS:**
- BM25 wins all three datasets — the OOV losses outnumber the synonym wins
- IDF-weighting consistently helps (+0.097 on TREC-COVID, +0.040 on SciFact) — suppressing stop words matters
- The gap tracks domain distance from Google News: closest is TREC-COVID (closest vocabulary), furthest is FIQA/SciFact
- **What this motivates:** We need models that (1) handle OOV via subword tokenisation, (2) are trained on retrieval data directly, (3) produce contextual embeddings (same word = different vector in different sentences)

**VISUAL:** Bar chart comparing BM25, W2V mean, W2V IDF across 3 datasets; annotate Word2Vec's TREC-COVID as "closest to BM25 — vocabulary gap smallest"

---

## SLIDE 19 — Phase 3: Dense Retrieval — Bi-encoder Architecture

**TITLE:** Phase 3 — Dense Retrieval: Encoding Meaning in Vectors

**BODY:**
**Bi-encoder (Siamese Network) Architecture:**
- **Query encoder:** Transformer → [CLS] token or mean-pooled vector (e.g., 384d or 768d)
- **Passage encoder:** Same or separate transformer → same-size vector
- **Similarity:** Dot product or cosine similarity between query and passage vectors
- **At inference:** Encode all passages offline → store in index → encode query → nearest-neighbour search

**Key advantage over Word2Vec:**
- Contextual: "bank" in a financial query gets a different vector than "bank" in a river query
- Subword tokenisation: no OOV problem — *"PCNA K164"* gets encoded, not dropped
- Contrastive training: vectors are optimised for retrieval, not word co-occurrence

**[PLACEHOLDER: Bi-encoder architecture diagram]**
*Suggested diagram: Two transformer towers (one for query, one for passage), both producing a vector, connected by cosine similarity. Contrast with cross-encoder (single tower processing both → interaction).*

**VISUAL:** [PLACEHOLDER — bi-encoder diagram: Query → BERT → q_vec; Passage → BERT → p_vec; cosine(q_vec, p_vec) → score]

---

## SLIDE 20 — Phase 3: Models Used

**TITLE:** Dense Models — A Family Tree

**BODY:**
| Model | Params | Year | Key Idea |
|-------|--------|------|----------|
| DPR | 2×110M | 2020 | Task-specific fine-tuning on Natural Questions (QA pairs) |
| all-MiniLM-L6-v2 | 22M | 2021 | Knowledge distillation from MPNet; 1B+ sentence pairs; very fast |
| all-mpnet-base-v2 | 110M | 2021 | MPNet: MLM + Permuted LM objective → richer representations |
| BGE-base-en-v1.5 | 110M | 2023 | Hard negative mining + instruction prefix training; MTEB SOTA |
| E5-base-v2 | 110M | 2023 | "Text Embeddings by Weakly-Supervised Contrastive Pre-training" |

**Training objective for MiniLM/MPNet/BGE/E5 (MultipleNegativesRankingLoss / InfoNCE):**
- For each (query, positive_passage) pair, all other passages in the batch are negative examples
- Loss pushes (q, p+) vectors together, pushes (q, p−) apart
- In-batch negatives = free negatives from the batch → scales efficiently

**[PLACEHOLDER: Model family tree diagram]**
*Suggested: BERT (2018) → DPR (2020) → SBERT/MiniLM (2021) → MPNet (2021) → BGE/E5 (2023), with arrows showing distillation and training improvements*

**VISUAL:** [PLACEHOLDER — model family tree + contrastive loss diagram showing positive pair pulled together, negatives pushed apart]

---

## SLIDE 21 — Phase 3: DPR — The Task-Specific Trap

**TITLE:** DPR: When Task-Specific Training Backfires

**BODY:**
- DPR was fine-tuned on 58,880 Natural Questions (NQ) QA pairs — state-of-the-art in 2020 for open-domain QA
- Works well when test distribution matches NQ (factoid English QA)
- **Collapses on any other domain**

| Model | TREC-COVID | FIQA | SciFact |
|-------|-----------|------|---------|
| DPR | 0.144 | 0.060 | 0.219 |
| BM25 | 0.447 | 0.159 | 0.560 |
| MiniLM | 0.473 | 0.369 | 0.645 |

**DPR is worse than BM25 on all three datasets** (and worse than Word2Vec on most).

**LESSON:** Narrow task-specific fine-tuning overfits the source domain. Training on diverse data (MiniLM: 1B+ sentence pairs from 28 sources) produces far better zero-shot generalisation.

**VISUAL:** Bar chart showing DPR vs BM25 vs MiniLM across 3 datasets, with DPR bars clearly below BM25

---

## SLIDE 22 — Phase 3: Dense Results

**TITLE:** Dense Retrieval Results — NDCG@10

**BODY:**
| Model | TREC-COVID | FIQA | SciFact |
|-------|-----------|------|---------|
| BM25 (baseline) | 0.447 | 0.159 | 0.560 |
| MiniLM-L6 (22M) | 0.473 | 0.369 | 0.645 |
| MPNet-base (110M) | 0.513 | **0.500** | 0.656 |
| E5-base-v2 (110M) | 0.696 | 0.399 | 0.719 |
| BGE-base-v1.5 (110M) | **0.781** | 0.406 | **0.740** |

**KEY FINDINGS:**
1. All general sentence encoders beat BM25 across every dataset — semantic generalisation wins
2. BGE is the strongest overall (+0.334 over BM25 on TREC-COVID)
3. MPNet is the best on FIQA despite being 2 years older than BGE/E5 — its training objective suits conversational queries
4. DPR is the cautionary tale — task-specific training without diversity = fragile

**VISUAL:** Bar chart: 3 panels (one per dataset), BM25 dashed reference line, colour-coded bars per model

---

## SLIDE 23 — Phase 3: Domain Shift Analysis

**TITLE:** Does Domain Shift Hurt Dense Models?

**BODY:**
**Setup:** All models trained on MS MARCO (web search text). TREC-COVID is biomedical literature. Does the vocabulary/domain gap hurt?

| Model | TREC-COVID NDCG@10 | vs BM25 |
|-------|-------------------|---------|
| BM25 | 0.447 | — |
| DPR | 0.144 | **−0.303** (huge loss) |
| MiniLM | 0.473 | +0.025 (barely wins) |
| MPNet | 0.513 | +0.066 |
| E5 | 0.696 | **+0.249** |
| BGE | 0.781 | **+0.334** |

**CONCLUSION:**
- Domain shift **does** hurt DPR — its narrow NQ training has no medical vocabulary
- Domain shift **does not** hurt general encoders — contrastive training on diverse data generalises
- Better-trained models survive domain shift more easily: BGE/E5 > MPNet > MiniLM

**NOTES:**
The key insight: it's not the domain gap that matters most — it's the breadth of the training distribution. BGE was trained on 1.2M+ diverse pairs. DPR was trained on 58K NQ pairs. Diversity > domain specificity for zero-shot generalisation.

**VISUAL:** Horizontal bar chart showing delta from BM25 per model, sorted ascending, DPR at bottom in red, BGE at top in green

---

## SLIDE 24 — Phase 4: Hybrid Retrieval — Architecture

**TITLE:** Phase 4 — Hybrid Retrieval: Best of Both Worlds

**BODY:**
**The problem with pure dense retrieval:**
- Single-vector bottleneck: all query semantics compressed into one 768-d vector
- Exact-match signals lost: acronyms, named entities, dataset-specific IDs
- Example: "COVID-19 PCR test" — BM25 matches "PCR" exactly; dense may not

**Hybrid RRF (Reciprocal Rank Fusion):**
1. Run BM25 → get ranked list L₁
2. Run BGE dense → get ranked list L₂
3. Fuse: `score(d) = 1/(k + rank_BM25(d)) + 1/(k + rank_BGE(d))` where k=60
4. Re-rank by fused score

**Two-stage Reranker (second stage):**
1. Hybrid RRF → top-100 candidates
2. Cross-encoder reranker scores each (query, passage) pair directly
3. Cross-encoder sees both query and passage together → deeper interaction, slower but more accurate
4. Output: reranked top-10

**[PLACEHOLDER: Two-stage retrieval pipeline diagram]**
*Suggested: [Query] → BM25 (top-100) + BGE (top-100) → RRF → top-100 → Cross-encoder → top-10*

**VISUAL:** [PLACEHOLDER — pipeline diagram: dual first stage → RRF merge → cross-encoder rerank → final result]

---

## SLIDE 25 — Phase 4: Cross-Encoder Architecture

**TITLE:** Cross-Encoder vs Bi-Encoder

**BODY:**
**Bi-encoder (Phase 3):**
- Encodes query and passage independently → fast, index offline
- Limitation: no query-passage interaction during encoding

**Cross-encoder:**
- Encodes [CLS] + query + [SEP] + passage together in one forward pass
- Full attention between every query token and every passage token
- Output: single relevance score (0–1)
- **Cannot** pre-compute passage encodings → must run for every (query, passage) pair

| | Bi-encoder | Cross-encoder |
|--|------------|---------------|
| Speed | Fast (pre-index) | Slow (per pair) |
| Accuracy | Good | Better |
| Use case | First stage (100K passages) | Reranker (top-100 only) |

**Model used:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

**[PLACEHOLDER: Cross-encoder architecture diagram]**
*Suggested: Query + Passage concatenated → single BERT tower → [CLS] → linear layer → relevance score*

**VISUAL:** [PLACEHOLDER — cross-encoder diagram: concatenated input → single transformer → score]

---

## SLIDE 26 — Phase 4: Hybrid Results

**TITLE:** Hybrid + Reranker Results

**BODY:**
| Method | TREC-COVID | FIQA | SciFact |
|--------|-----------|------|---------|
| BM25 | 0.447 | 0.159 | 0.560 |
| BGE (best dense) | **0.781** | 0.406 | 0.740 |
| Hybrid RRF (BM25+BGE) | 0.710 | 0.292 | 0.667 |
| Hybrid + Reranker | 0.763 | 0.374 | **0.689** |

**SURPRISING FINDING — Hybrid is NOT always better than pure dense:**
- On TREC-COVID: BGE alone (0.781) > Hybrid RRF (0.710) > Hybrid+Reranker (0.763)
- BM25 is weak on biomedical vocabulary → adding it to RRF drags down a strong BGE signal
- Reranker partially recovers (+0.053 over RRF alone) but doesn't reach BGE alone

**WHEN HYBRID HELPS:**
- When BM25 and dense have complementary strengths (neither clearly dominates)
- SciFact: Hybrid+Reranker (0.689) > BGE (0.740) — no! BGE still wins here too
- Reranker consistently adds 2–8 NDCG points over RRF

**LESSON:** If your dense model is strong, hybrid doesn't always help. Cross-encoder reranking helps more consistently.

**VISUAL:** Grouped bar chart showing all 4 methods across 3 datasets; annotate "BGE alone beats hybrid on TREC-COVID"

---

## SLIDE 27 — Phase 4: Recall Metrics

**TITLE:** What About Recall? The Reranker's Hidden Strength

**BODY:**
| Method | Recall@10 | Recall@50 | Recall@100 |
|--------|-----------|-----------|------------|
| **SciFact** | | | |
| BM25 | 0.686 | 0.777 | 0.793 |
| BGE | 0.874 | — | 0.967 |
| Hybrid RRF | 0.798 | 0.934 | 0.953 |
| Hybrid+Reranker | **0.809** | 0.930 | 0.953 |
| **TREC-COVID** | | | |
| BM25 | 0.011 | 0.043 | 0.071 |
| BGE | 0.022 | — | 0.141 |
| Hybrid RRF | 0.019 | 0.072 | 0.120 |
| Hybrid+Reranker | 0.021 | 0.084 | 0.119 |

**NOTE on TREC-COVID:** Low absolute recall because there are 493 relevant passages per query — recall@100 can only find ~12% of all relevant passages. This is a corpus coverage limit, not a model limit.

**NOTES:**
Recall@100 stays the same for reranker vs RRF because the reranker only reorders the top-100, not filters them. Recall improves from RRF→Reranker only at lower ranks (recall@10) because the reranker pushes relevant results higher. This is the reranker's value: better precision at small K.

**VISUAL:** Table as shown; annotate that TREC-COVID recall@100 is low due to 493 relevant/query, not model failure

---

## SLIDE 28 — Phase 5: LLM-as-Judge

**TITLE:** Phase 5 — LLM-as-Judge: Beyond Annotation Gaps

**BODY:**
**Motivation:** BEIR qrels have annotation gaps — not every relevant passage is labelled. A passage that IS relevant but NOT in qrels scores 0. LLM judges can catch these.

**Setup:**
- Model: `llama3.2:3b` running locally via Ollama
- Task: For each (query, top-5 passages) pair — is this passage relevant? (yes/no)
- Metric: **Context Precision** = fraction of top-5 passages judged relevant
- Datasets evaluated: SciFact (n=150) and TREC-COVID (n=50)

**Results:**
| Dataset | LLM Context Precision | NDCG@10 (IR) | Agree? |
|---------|----------------------|--------------|--------|
| TREC-COVID | 0.628 | 0.763 | Yes — both high |
| SciFact | 0.257 | 0.689 | **No — diverge sharply** |

**KEY FINDING:** SciFact is a claim-verification task. The LLM (Llama 3.2) doesn't naturally handle scientific claim verification — it applies strict topical relevance rather than "does this abstract support/refute the claim?". IR metrics (NDCG) and LLM judgment measure different things on non-factoid tasks.

**VISUAL:** [PLACEHOLDER — diagram showing the LLM judge pipeline: (query, passage) → Ollama prompt → yes/no → aggregate]

---

## SLIDE 29 — Phase 6: Fine-tuning — Setup

**TITLE:** Phase 6 — Fine-tuning Encoders on Domain Data

**BODY:**
**Goal:** Can a small amount of in-domain data improve a general-purpose model?

**Base model:** `all-MiniLM-L6-v2` (22M parameters) — already good at zero-shot retrieval

**Training data:**
- FIQA train split: 14,131 (query, relevant passage) pairs
- SciFact train split: 919 (claim, abstract) pairs
- Combined (FIQA + SciFact): 15,050 pairs

**Training:**
- Loss: **MultipleNegativesRankingLoss (MNR / InfoNCE)**
  - Every other passage in the batch is a "free" hard negative
  - No explicit negative mining needed
- Batch size: 64, 3 epochs, LR 2e-5, warmup 10%
- Hardware: Apple M5 MPS
- Training time: 6.3 min (FIQA), 1.1 min (SciFact), 6.7 min (combined)

**VISUAL:** [PLACEHOLDER — MNR loss diagram: batch of 4 pairs → query 1 sees pair 1 as positive, pairs 2/3/4 as negatives → contrastive loss pushes positives together, negatives apart]

---

## SLIDE 30 — Phase 6: Fine-tuning Results

**TITLE:** Fine-tuning Results — NDCG@10

**BODY:**
| Model | TREC-COVID | FIQA | SciFact |
|-------|-----------|------|---------|
| MiniLM base | 0.473 | 0.369 | 0.645 |
| ft-FIQA only | 0.487 | **0.396** | 0.645 |
| ft-SciFact only | 0.461 | 0.359 | **0.695** |
| ft-Combined | **0.503** | **0.400** | 0.660 |

**Delta from base:**
| Fine-tuned on | TREC-COVID | FIQA | SciFact |
|---------------|-----------|------|---------|
| FIQA | +0.014 | **+0.028** | ±0.000 |
| SciFact | −0.012 | −0.009 | **+0.050** |
| Combined | **+0.030** | **+0.031** | +0.015 |

**NOTES:**
Two key stories here: (1) In-domain fine-tuning on 919 examples gives +7.7% NDCG on SciFact — remarkable data efficiency. (2) ft-SciFact shows negative transfer to FIQA/TREC-COVID — domain specialisation trades breadth for depth. ft-Combined avoids this trade-off: it improves all three simultaneously.

**VISUAL:** Before/after grouped bar chart showing base vs fine-tuned variants per dataset; highlight combined row in green

---

## SLIDE 31 — Phase 6: Fine-tuning Key Lessons

**TITLE:** What Fine-tuning Teaches Us

**BODY:**
1. **In-domain specialisation is powerful:** 919 SciFact training pairs → +7.7% NDCG on SciFact in 1.1 minutes. Small data, large gain when distribution matches.

2. **Negative transfer is real but small:** ft-SciFact loses ~1 NDCG point on FIQA/TREC-COVID. The model "forgets" some general-purpose alignment to specialise.

3. **Combined training generalises without negative transfer:** ft-Combined improves all three datasets simultaneously (+3% TREC-COVID, +3.1% FIQA, +1.5% SciFact). When deployment domain is mixed, combined training is the better choice.

4. **Recall@10 jump is striking:** SciFact: 0.783 → 0.865 (+8.2 points). Fine-tuning especially improves early-recall — the model learns to put the right passage at rank 1 or 2 rather than 5.

5. **Small model + fine-tuning can beat larger un-tuned models:** ft-SciFact MiniLM (22M, 0.695) approaches E5 (110M, 0.719) and exceeds MPNet (110M, 0.656) on SciFact. Domain data beats model size.

**VISUAL:** Callout card for each lesson; or annotated scatter plot: model size vs NDCG, showing fine-tuned 22M near 110M models

---

## SLIDE 32 — Full Comparison: All Methods

**TITLE:** Complete Picture — All Methods, All Datasets

**BODY:**
| Method | TREC-COVID | FIQA | SciFact |
|--------|-----------|------|---------|
| BOW | 0.176 | 0.066 | 0.365 |
| TF-IDF | 0.287 | 0.179 | 0.629 |
| BM25 | 0.447 | 0.159 | 0.560 |
| Word2Vec mean | 0.339 | 0.060 | 0.269 |
| Word2Vec IDF | 0.436 | 0.089 | 0.310 |
| MiniLM-L6 (22M) | 0.473 | 0.369 | 0.645 |
| MPNet-base (110M) | 0.513 | **0.500** | 0.656 |
| E5-base-v2 (110M) | 0.696 | 0.399 | 0.719 |
| BGE-base-v1.5 (110M) | 0.781 | 0.406 | 0.740 |
| Hybrid RRF (BM25+BGE) | 0.710 | 0.292 | 0.667 |
| Hybrid + Reranker | 0.763 | 0.374 | 0.689 |
| MiniLM ft-Combined | 0.503 | 0.400 | 0.660 |

**HEADLINE NUMBERS:**
- Best overall: BGE on TREC-COVID (0.781), BGE on SciFact (0.740), MPNet on FIQA (0.500)
- Biggest leap: BM25→MiniLM on FIQA (+0.210 NDCG — dense solves the synonym gap)
- Biggest surprise: BGE alone > Hybrid+Reranker on TREC-COVID

**VISUAL:** Heatmap — rows = methods, columns = datasets, cell colour = NDCG@10 value (white→green scale)

---

## SLIDE 33 — Key Takeaways

**TITLE:** Key Takeaways

**BODY:**
1. **BM25 is not always the best sparse baseline** — TF-IDF beats it on short/uniform corpora (FIQA, SciFact). Always tune or compare before assuming defaults.

2. **Word2Vec bridged the vocabulary gap for common-language queries** but the OOV problem kills it on specialised domains. It's a useful diagnostic, not a production choice.

3. **General-purpose dense models beat BM25 across all domains** — contrastive training on diverse data generalises better than domain-specific tuning. BGE/E5 are the current practical choice.

4. **DPR is a cautionary tale** — narrow task-specific fine-tuning without data diversity produces brittle models that collapse outside their training distribution.

5. **Hybrid ≠ always better** — if your dense model is strong, adding BM25 via RRF can hurt by dragging in its weaknesses. Test before assuming hybrid helps.

6. **Cross-encoder reranking consistently helps** — +2 to +8 NDCG points over first-stage retrieval. Production pipelines should include a reranker.

7. **Fine-tuning with small in-domain data is highly efficient** — 919 pairs, 1.1 minutes → +7.7% NDCG. When you have labelled domain data, use it.

8. **Combined training avoids negative transfer** — when deploying across multiple domains, train on all together rather than specialised models per domain.

**VISUAL:** 8 takeaway cards, each with a short headline and 1-line detail

---

## SLIDE 34 — Practical Decision Guide

**TITLE:** Which Method Should You Use?

**BODY:**
| Scenario | Recommendation |
|----------|---------------|
| No labels, no GPU, need something fast | BM25 (tune b for your corpus length) |
| No labels, have GPU | BGE-base or E5-base (zero-shot) |
| Have some labelled domain pairs (100+) | Fine-tune MiniLM on your domain |
| Mixed domain deployment | Fine-tune on combined domain data |
| Need highest precision, latency OK | BGE first stage + cross-encoder reranker |
| Feeding a RAG pipeline | BGE/E5 + cross-encoder reranker |
| Very short/uniform passages | TF-IDF may beat BM25 — test both |

**NOTES:**
This is the practical output of the whole study. Not "dense is always better" but "here is when to reach for each tool." The key variable is usually whether you have domain-labelled data and whether latency is a constraint.

**VISUAL:** Decision flowchart: Do you have domain labels? → fine-tune vs zero-shot. Do you need speed? → bi-encoder vs cross-encoder. Is your corpus long/variable? → BM25 vs TF-IDF.

---

## SLIDE 35 — Conclusions & Next Steps

**TITLE:** Conclusions

**BODY:**
**What we showed:**
- Traced retrieval evolution across 6 generations with real experiments on 3 diverse benchmarks
- Confirmed that each generation's fix works — but with important caveats
- Identified surprising results: TF-IDF > BM25 on short corpora; BGE > Hybrid on biomedical; fine-tuning data efficiency

**Open questions / Next steps:**
- Late interaction models (ColBERT) — token-level matching beyond single-vector bottleneck
- Larger corpora (MS MARCO, NQ) for scale testing
- RAG end-to-end: does better retrieval actually improve LLM answer quality proportionally?
- Instruction-tuned embeddings (E5-instruct, BGE-M3) — task-aware at inference time

**VISUAL:** Closing slide with the full evolution timeline, all NDCG@10 numbers annotated on a line chart per dataset (x-axis = method generation, y-axis = NDCG@10, three lines for three datasets)

---

## APPENDIX A — BM25 Hyperparameter Investigation

**TITLE:** BM25 Hyperparameter Sweep Results

**BODY:**
b sweep (k₁=1.5 fixed):
| Dataset | b=0.0 | b=0.25 | b=0.5 | b=0.75 (default) | b=1.0 | TF-IDF |
|---------|-------|--------|-------|--------|-------|--------|
| SciFact | 0.547 | 0.558 | 0.556 | 0.560 | 0.563 | **0.629** |
| FIQA | 0.099 | 0.147 | 0.162 | 0.159 | 0.039 | **0.179** |
| TREC-COVID | 0.337 | 0.460 | **0.479** | 0.447 | 0.317 | 0.287 |

**CONCLUSION:** TF-IDF advantage on SciFact/FIQA is structural — even b=0 doesn't close the gap. BM25's explicit length normalisation is tuned for long variable passages (MS MARCO). TF-IDF's implicit global L2 normalisation suits short/uniform corpora better.

---

## APPENDIX B — Word2Vec Per-Query Analysis (SciFact)

**TITLE:** Word2Vec vs BM25 — Per-Query NDCG@10 on SciFact

**BODY:**
**Top 5 queries where Word2Vec > BM25:**
| NDCG delta | Query |
|-----------|-------|
| +0.667 | Headaches are not correlated with cognitive impairment |
| +0.667 | Low nucleosome occupancy correlates with low methylation levels across species |
| +0.631 | Flexible molecules experience greater steric hindrance in the tumor microenvironment |
| +0.631 | Modifying the epigenome in the brain affects normal aging by affecting neurogenesis genes |
| +0.613 | A deficiency of vitamin B12 increases blood levels of homocysteine |

**Top 5 queries where BM25 > Word2Vec:**
| NDCG delta | Query |
|-----------|-------|
| −1.000 | The minor G allele of FOXO3 is related to more severe symptoms of Crohn's Disease |
| −1.000 | There is no association between HNF4A mutations and diabetes risks |
| −1.000 | Transplanted human glial cells can differentiate within the host animal |
| −1.000 | Ubiquitin ligase UBC13 generates a K63-linked polyubiquitin moiety at PCNA K164 |
| −1.000 | cSMAC formation enhances weak ligand signalling |

---

## APPENDIX C — Finetuning Full Results

**TITLE:** Fine-tuning — Full Results Including MRR@10

**BODY:**
| Model | TC NDCG | TC MRR | FIQA NDCG | FIQA MRR | SF NDCG | SF MRR |
|-------|---------|--------|-----------|----------|---------|--------|
| MiniLM base | 0.473 | 0.727 | 0.369 | 0.454 | 0.645 | 0.611 |
| ft-FIQA | 0.487 | 0.726 | 0.396 | 0.479 | 0.645 | 0.614 |
| ft-SciFact | 0.461 | 0.690 | 0.359 | 0.431 | 0.695 | 0.644 |
| ft-Combined | 0.503 | 0.730 | 0.400 | 0.475 | 0.660 | 0.620 |

---

## NOTES ON ARCHITECTURE DIAGRAMS (PLACEHOLDERS)

The following diagrams are marked [PLACEHOLDER] and should be added manually or sourced from papers:

1. **Slide 3:** RAG pipeline — simple block diagram
2. **Slide 16:** Word2Vec Skip-gram or CBOW architecture + averaging diagram
3. **Slide 19:** Bi-encoder (two BERT towers) diagram
4. **Slide 20:** Model family tree (BERT → DPR → SBERT → BGE/E5)
5. **Slide 24:** Two-stage hybrid pipeline (BM25 + BGE → RRF → cross-encoder → top-10)
6. **Slide 25:** Cross-encoder architecture (concatenated input → single BERT → score)
7. **Slide 29:** LLM judge pipeline
8. **Slide 29:** MNR loss / InfoNCE contrastive diagram

**Suggested sources for diagrams:**
- SBERT paper (Reimers & Gurevych, 2019) — bi-encoder diagram
- DPR paper (Karpukhin et al., 2020) — dual encoder
- ColBERT paper — late interaction (if added later)
- Hugging Face SBERT docs — training loss diagrams
