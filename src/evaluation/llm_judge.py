"""
LLM-as-judge evaluation via direct Ollama API (local, no API cost).

Metric: context_precision@k
  "Of the top-k retrieved passages, what fraction does the LLM judge as relevant
   to the query?" — does NOT require a reference answer or generated response.

Prompt-based, no RAGAS dependency — avoids RAGAS dataset column compatibility issues.

Usage:
    judge = LLMJudge(ollama_model="llama3.2:3b")
    scores = judge.evaluate(queries, results, corpus, sample_n=150, top_k=5)
    # scores: {qid: context_precision_score (0.0–1.0)}
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_RELEVANCE_PROMPT = """\
You are a relevance judge. Given a search query and a passage, decide if the passage \
is relevant to the query.

Query: {query}

Passage: {passage}

Is this passage relevant to the query? Reply with exactly one word: yes or no."""


class LLMJudge:
    """
    Judges retrieval quality by prompting a local Ollama LLM to score
    each (query, passage) pair as relevant/not-relevant, then computes
    context_precision@k = (relevant in top-k) / k.
    """

    def __init__(
        self,
        ollama_model: str = "llama3.2:3b",
        ollama_base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.timeout = timeout

    def _check_ollama(self) -> bool:
        try:
            resp = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            base = self.ollama_model.split(":")[0]
            if not any(base in m for m in models):
                logger.warning(
                    f"Model '{self.ollama_model}' not in Ollama. "
                    f"Run: ollama pull {self.ollama_model}"
                )
                return False
            logger.info(f"Ollama running — model '{self.ollama_model}' available")
            return True
        except Exception as e:
            logger.error(f"Ollama not reachable at {self.ollama_base_url}: {e}")
            logger.error("Start with: ollama serve")
            return False

    def _judge_passage(self, query: str, passage: str) -> bool:
        """Returns True if the LLM judges the passage as relevant."""
        prompt = _RELEVANCE_PROMPT.format(query=query, passage=passage[:1000])
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 5},
        }
        try:
            resp = httpx.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip().lower()
            return answer.startswith("yes")
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return False

    def evaluate(
        self,
        queries: Dict[str, str],
        results: Dict[str, Dict[str, float]],
        corpus: Dict[str, Dict],
        sample_n: int = 150,
        top_k: int = 5,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Compute context_precision@top_k on a random sample of queries.

        Returns {qid: precision_score (0.0–1.0)}.
        """
        if not self._check_ollama():
            raise RuntimeError(
                f"Ollama not available. Run 'ollama serve' and "
                f"'ollama pull {self.ollama_model}'"
            )

        qids = list(queries.keys())
        random.seed(seed)
        sample_qids = random.sample(qids, min(sample_n, len(qids)))

        scores: Dict[str, float] = {}
        total = len(sample_qids)

        for i, qid in enumerate(sample_qids, 1):
            q_text = queries[qid]
            top_docs = sorted(
                results.get(qid, {}).items(), key=lambda x: x[1], reverse=True
            )[:top_k]

            relevant = 0
            judged = 0
            for did, _ in top_docs:
                if did not in corpus:
                    continue
                passage = (
                    corpus[did].get("title", "") + " " + corpus[did].get("text", "")
                ).strip()
                if self._judge_passage(q_text, passage):
                    relevant += 1
                judged += 1

            precision = relevant / judged if judged > 0 else 0.0
            scores[qid] = precision

            if i % 10 == 0 or i == total:
                running_mean = sum(scores.values()) / len(scores)
                logger.info(
                    f"  [{i}/{total}] mean context_precision so far: {running_mean:.4f}"
                )

        return scores

    def mean_score(self, scores: Dict[str, float]) -> float:
        vals = [v for v in scores.values() if v == v]  # exclude NaN
        return sum(vals) / len(vals) if vals else 0.0
