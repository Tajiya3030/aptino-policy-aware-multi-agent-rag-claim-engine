from typing import List, Dict, Any


class CrossEncoderReranker:
    """
    Lightweight reranker for memory-constrained deployments.

    Uses BM25/RRF score plus keyword overlap instead of loading
    a CrossEncoder model.
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        if not candidates:
            return []

        q_words = set(query.lower().split())

        for c in candidates:
            base_score = c.get("rrf_score", 0.0) * 50.0

            c_words = set(
                c.get("content", "").lower().split()
            )

            overlap = len(
                q_words.intersection(c_words)
            )

            c["rerank_score"] = round(
                base_score + (overlap * 0.1),
                4
            )

        reranked = sorted(
            candidates,
            key=lambda c: c["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]