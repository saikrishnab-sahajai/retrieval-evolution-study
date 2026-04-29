"""
Dense retrieval: Word2Vec averaging, SentenceTransformer bi-encoders, DPR.

All retrievers expose the same interface as sparse.py:
    retriever.index(corpus)           → build index
    retriever.retrieve(queries, k)    → dict[qid, dict[doc_id, score]]

Hardware: Apple M5 — MPS backend for embedding; numpy exact search for retrieval.
Note: FAISS removed — on macOS, FAISS and PyTorch both ship libomp.dylib, causing
      OMP Error #15 (double-init SIGABRT). Numpy dot-product search is exact and
      sufficient for corpora ≤200K docs.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MPS device helper
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Word2Vec mean / IDF-weighted pooling
# ---------------------------------------------------------------------------

class Word2VecRetriever:
    """
    Dense retrieval using Word2Vec (Google News 300d) with pooling.
    Demonstrates the failure of context-free averaging.
    """

    def __init__(self, model_key: str = "word2vec-google-news-300", pooling: str = "mean"):
        self.model_key = model_key
        self.pooling = pooling  # "mean" or "idf_weighted"
        self.wv = None
        self.idf: Optional[Dict[str, float]] = None
        self.doc_vectors: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self.index_time: float = 0.0

    def _load_model(self) -> None:
        import gensim.downloader as api
        logger.info(f"Loading {self.model_key} (may take a few minutes first time)...")
        wv_model = api.load(self.model_key)
        self.wv = wv_model

    def _compute_idf(self, texts: List[str]) -> Dict[str, float]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(analyzer="word", min_df=1)
        vec.fit(texts)
        idf_arr = vec.idf_
        vocab = vec.get_feature_names_out()
        return dict(zip(vocab, idf_arr))

    def _embed(self, text: str) -> Optional[np.ndarray]:
        tokens = text.lower().split()
        vecs = []
        weights = []
        for tok in tokens:
            if tok in self.wv:
                vecs.append(self.wv[tok])
                w = self.idf.get(tok, 1.0) if self.pooling == "idf_weighted" else 1.0
                weights.append(w)
        if not vecs:
            return None
        vecs = np.array(vecs)
        if self.pooling == "idf_weighted":
            w_arr = np.array(weights)[:, None]
            return (vecs * w_arr).sum(axis=0) / (w_arr.sum() + 1e-9)
        return vecs.mean(axis=0)

    def index(self, corpus: Dict[str, Dict]) -> None:
        if self.wv is None:
            self._load_model()

        t0 = time.time()
        self.doc_ids = list(corpus.keys())
        texts = [
            (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
            for d in self.doc_ids
        ]
        if self.pooling == "idf_weighted":
            self.idf = self._compute_idf(texts)

        vecs = []
        for txt in tqdm(texts, desc="Word2Vec embed corpus"):
            v = self._embed(txt)
            vecs.append(v if v is not None else np.zeros(self.wv.vector_size))
        self.doc_vectors = np.array(vecs, dtype=np.float32)

        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True) + 1e-9
        self.doc_vectors /= norms

        self.index_time = time.time() - t0
        logger.info(f"Word2Vec index built in {self.index_time:.1f}s ({len(self.doc_ids):,} docs)")

    def retrieve(
        self, queries: Dict[str, str], top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for qid, q_text in tqdm(queries.items(), desc=f"Word2Vec ({self.pooling}) retrieve"):
            qv = self._embed(q_text)
            if qv is None:
                results[qid] = {}
                continue
            qv = qv.astype(np.float32)
            qv /= np.linalg.norm(qv) + 1e-9
            scores = self.doc_vectors @ qv
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            results[qid] = {self.doc_ids[j]: float(scores[j]) for j in top_idx}
        return results


# ---------------------------------------------------------------------------
# Doc2Vec PV-DBOW (Le & Mikolov, 2014)
# ---------------------------------------------------------------------------

class Doc2VecRetriever:
    """
    Doc2Vec PV-DBOW retriever.
    Trains a dedicated document vector per passage — no word averaging.
    dm=0 (PV-DBOW) trains faster and outperforms PV-DM on retrieval tasks.
    """

    def __init__(
        self,
        vector_size: int = 300,
        epochs: int = 40,
        dm: int = 0,
        min_count: int = 2,
        workers: int = 4,
    ):
        self.vector_size = vector_size
        self.epochs = epochs
        self.dm = dm
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.doc_ids: List[str] = []
        self.doc_matrix: Optional[np.ndarray] = None
        self.index_time: float = 0.0

    def _prepare_tagged(self, corpus: Dict[str, Dict]):
        from gensim.models.doc2vec import TaggedDocument
        return [
            TaggedDocument(
                words=(corpus[pid].get("title", "") + " " + corpus[pid].get("text", "")).lower().split(),
                tags=[pid],
            )
            for pid in corpus
        ]

    def index(self, corpus: Dict[str, Dict]) -> None:
        from gensim.models.doc2vec import Doc2Vec
        t0 = time.time()
        self.doc_ids = list(corpus.keys())
        tagged = self._prepare_tagged(corpus)

        logger.info(
            f"Training Doc2Vec (dm={self.dm}, epochs={self.epochs}, "
            f"dim={self.vector_size}) on {len(tagged):,} docs..."
        )
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=self.dm,
            workers=self.workers,
        )
        self.model.build_vocab(tagged)
        self.model.train(tagged, total_examples=self.model.corpus_count, epochs=self.model.epochs)

        doc_vecs = np.array([self.model.dv[pid] for pid in self.doc_ids], dtype=np.float32)
        norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9
        self.doc_matrix = doc_vecs / norms

        self.index_time = time.time() - t0
        logger.info(f"Doc2Vec index built in {self.index_time:.1f}s ({len(self.doc_ids):,} docs)")

    def retrieve(
        self, queries: Dict[str, str], top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for qid, q_text in tqdm(queries.items(), desc="Doc2Vec retrieve"):
            tokens = q_text.lower().split()
            qv = self.model.infer_vector(tokens, epochs=20).astype(np.float32)
            qv /= np.linalg.norm(qv) + 1e-9
            scores = self.doc_matrix @ qv
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            results[qid] = {self.doc_ids[j]: float(scores[j]) for j in top_idx}
        return results


# ---------------------------------------------------------------------------
# SentenceTransformer bi-encoder (MPS-aware)
# ---------------------------------------------------------------------------

class BiEncoderRetriever:
    """
    Dense retrieval using SentenceTransformer bi-encoders.
    Covers: MiniLM, MPNet, BGE, E5.
    Numpy exact search (replaces FAISS to avoid OpenMP conflict on macOS).
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        batch_size: int = 128,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        self.model_id = model_id
        self.device = device or get_device()
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model = None
        self.doc_matrix: Optional[np.ndarray] = None  # shape: (N, D), L2-normalised
        self.doc_ids: List[str] = []
        self.index_time: float = 0.0
        self.embedding_dim: Optional[int] = None

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading {self.model_id} on {self.device}...")
        self.model = SentenceTransformer(self.model_id, device=self.device)

    def _encode(self, texts: List[str], prefix: str = "", show_progress: bool = True) -> np.ndarray:
        if prefix:
            texts = [prefix + t for t in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine via inner product
        )
        return embeddings.astype(np.float32)

    def index(self, corpus: Dict[str, Dict]) -> None:
        if self.model is None:
            self._load_model()

        t0 = time.time()
        self.doc_ids = list(corpus.keys())
        texts = [
            (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
            for d in self.doc_ids
        ]

        logger.info(f"Encoding {len(texts):,} passages with {self.model_id}...")
        embeddings = self._encode(texts, prefix=self.passage_prefix)
        self.embedding_dim = embeddings.shape[1]
        self.doc_matrix = embeddings  # already L2-normalised by _encode
        self.index_time = time.time() - t0
        logger.info(f"Index built in {self.index_time:.1f}s ({len(self.doc_ids):,} docs, dim={self.embedding_dim})")

    def retrieve(
        self, queries: Dict[str, str], top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        q_ids = list(queries.keys())
        q_texts = list(queries.values())

        logger.info(f"Encoding {len(q_texts):,} queries...")
        q_embeddings = self._encode(q_texts, prefix=self.query_prefix, show_progress=True)

        logger.info(f"Numpy exact search top-{top_k}...")
        # q_embeddings: (Q, D), doc_matrix: (N, D) — both L2-normalised
        score_matrix = q_embeddings @ self.doc_matrix.T  # (Q, N)

        results: Dict[str, Dict[str, float]] = {}
        for i, qid in enumerate(q_ids):
            row = score_matrix[i]
            top_idx = np.argpartition(row, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            results[qid] = {self.doc_ids[j]: float(row[j]) for j in top_idx}
        return results


# ---------------------------------------------------------------------------
# DPR — separate question / context encoders
# ---------------------------------------------------------------------------

class DPRRetriever:
    """
    Dense Passage Retrieval (Karpukhin et al., 2020).
    Uses separate question and context encoders.
    """

    def __init__(
        self,
        ctx_encoder_id: str = "facebook/dpr-ctx_encoder-single-nq-base",
        q_encoder_id: str = "facebook/dpr-question_encoder-single-nq-base",
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.ctx_encoder_id = ctx_encoder_id
        self.q_encoder_id = q_encoder_id
        self.device = device or get_device()
        self.batch_size = batch_size
        self.ctx_encoder = None
        self.q_encoder = None
        self.doc_matrix: Optional[np.ndarray] = None  # shape: (N, D), L2-normalised
        self.doc_ids: List[str] = []
        self.index_time: float = 0.0

    def _load_models(self) -> None:
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        logger.info("Loading DPR context encoder...")
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.ctx_encoder_id)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(self.ctx_encoder_id).to(self.device)
        self.ctx_encoder.eval()

        logger.info("Loading DPR question encoder...")
        self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.q_encoder_id)
        self.q_encoder = DPRQuestionEncoder.from_pretrained(self.q_encoder_id).to(self.device)
        self.q_encoder.eval()

    def _encode_passages(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="DPR ctx encode"):
            batch = texts[i : i + self.batch_size]
            inputs = self.ctx_tokenizer(
                batch, truncation=True, max_length=512,
                padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                embs = self.ctx_encoder(**inputs).pooler_output
            all_embeddings.append(embs.cpu().float().numpy())
        embeddings = np.vstack(all_embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        return (embeddings / norms).astype(np.float32)

    def _encode_queries(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.q_tokenizer(
                batch, truncation=True, max_length=128,
                padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                embs = self.q_encoder(**inputs).pooler_output
            all_embeddings.append(embs.cpu().float().numpy())
        embeddings = np.vstack(all_embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        return (embeddings / norms).astype(np.float32)

    def index(self, corpus: Dict[str, Dict]) -> None:
        if self.ctx_encoder is None:
            self._load_models()

        t0 = time.time()
        self.doc_ids = list(corpus.keys())
        texts = [
            (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
            for d in self.doc_ids
        ]
        embeddings = self._encode_passages(texts)
        self.doc_matrix = embeddings  # already L2-normalised by _encode_passages
        self.index_time = time.time() - t0
        logger.info(f"DPR index built in {self.index_time:.1f}s ({len(self.doc_ids):,} docs)")

    def retrieve(
        self, queries: Dict[str, str], top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        q_ids = list(queries.keys())
        q_texts = list(queries.values())
        q_embeddings = self._encode_queries(q_texts)

        # q_embeddings: (Q, D), doc_matrix: (N, D) — both L2-normalised
        score_matrix = q_embeddings @ self.doc_matrix.T  # (Q, N)
        results: Dict[str, Dict[str, float]] = {}
        for i, qid in enumerate(q_ids):
            row = score_matrix[i]
            top_idx = np.argpartition(row, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            results[qid] = {self.doc_ids[j]: float(row[j]) for j in top_idx}
        return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: List[Dict[str, Dict[str, float]]],
    k: int = 60,
) -> Dict[str, Dict[str, float]]:
    """
    Merge multiple ranked lists via RRF.
    result_lists: list of {qid: {doc_id: score}} dicts
    Returns fused {qid: {doc_id: rrf_score}} dict.
    """
    all_qids = set()
    for r in result_lists:
        all_qids.update(r.keys())

    fused: Dict[str, Dict[str, float]] = {}
    for qid in all_qids:
        rrf_scores: Dict[str, float] = {}
        for result in result_lists:
            if qid not in result:
                continue
            ranked = sorted(result[qid].items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        fused[qid] = dict(
            sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        )
    return fused
