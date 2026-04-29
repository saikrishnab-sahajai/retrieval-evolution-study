"""
Sparse retrieval: BOW, TF-IDF, BM25.

All retrievers expose the same interface:
    retriever.index(corpus)           → build index
    retriever.retrieve(queries, k)    → dict[qid, dict[doc_id, score]]
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> List[str]:
    return text.lower().split()


class BOWRetriever:
    """Bag-of-Words with cosine similarity (sklearn CountVectorizer)."""

    def __init__(self, max_features: int = 100_000, binary: bool = False):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            binary=binary,
            analyzer="word",
        )
        self.doc_matrix = None
        self.doc_ids: List[str] = []
        self.index_time: float = 0.0

    def index(self, corpus: Dict[str, Dict]) -> None:
        """Build BOW index from corpus dict {doc_id: {'title': ..., 'text': ...}}."""
        t0 = time.time()
        self.doc_ids = list(corpus.keys())
        texts = [
            (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
            for d in self.doc_ids
        ]
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self.index_time = time.time() - t0
        logger.info(f"BOW index built in {self.index_time:.1f}s ({len(self.doc_ids):,} docs)")

    def retrieve(
        self, queries: Dict[str, str], top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Return top-k docs per query."""
        q_ids = list(queries.keys())
        q_texts = list(queries.values())
        q_matrix = self.vectorizer.transform(q_texts)

        results: Dict[str, Dict[str, float]] = {}
        for i, qid in enumerate(tqdm(q_ids, desc="BOW retrieve")):
            scores = cosine_similarity(q_matrix[i], self.doc_matrix).flatten()
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            results[qid] = {self.doc_ids[j]: float(scores[j]) for j in top_idx}
        return results


class TFIDFRetriever:
    """TF-IDF with cosine similarity (sklearn TfidfVectorizer)."""

    def __init__(self, max_features: int = 100_000, sublinear_tf: bool = True):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            analyzer="word",
        )
        self.doc_matrix = None
        self.doc_ids: List[str] = []
        self.index_time: float = 0.0

    def index(self, corpus: Dict[str, Dict]) -> None:
        t0 = time.time()
        self.doc_ids = list(corpus.keys())
        texts = [
            (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
            for d in self.doc_ids
        ]
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self.index_time = time.time() - t0
        logger.info(f"TF-IDF index built in {self.index_time:.1f}s ({len(self.doc_ids):,} docs)")

    def retrieve(
        self, queries: Dict[str, str], top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        q_ids = list(queries.keys())
        q_texts = list(queries.values())
        q_matrix = self.vectorizer.transform(q_texts)

        results: Dict[str, Dict[str, float]] = {}
        for i, qid in enumerate(tqdm(q_ids, desc="TF-IDF retrieve")):
            scores = cosine_similarity(q_matrix[i], self.doc_matrix).flatten()
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            results[qid] = {self.doc_ids[j]: float(scores[j]) for j in top_idx}
        return results


class BM25Retriever:
    """BM25 (Okapi) retrieval via rank_bm25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.bm25 = None
        self.doc_ids: List[str] = []
        self.index_time: float = 0.0

    def index(self, corpus: Dict[str, Dict]) -> None:
        t0 = time.time()
        self.doc_ids = list(corpus.keys())
        tokenised = [
            _tokenise(
                (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
            )
            for d in self.doc_ids
        ]
        self.bm25 = BM25Okapi(tokenised, k1=self.k1, b=self.b, epsilon=self.epsilon)
        self.index_time = time.time() - t0
        logger.info(f"BM25 index built in {self.index_time:.1f}s ({len(self.doc_ids):,} docs)")

    def retrieve(
        self, queries: Dict[str, str], top_k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for qid, q_text in tqdm(queries.items(), desc="BM25 retrieve"):
            tokens = _tokenise(q_text)
            scores = self.bm25.get_scores(tokens)
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            results[qid] = {self.doc_ids[j]: float(scores[j]) for j in top_idx}
        return results
