from typing import List, Dict, Any

class CrossEncoderReranker:
    """
    Reranks retrieved policy chunks using sentence similarity / cross-encoder scores
    to refine candidate order before presenting evidence to agents.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._cross_encoder = None

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.model_name)
            except Exception:
                self._cross_encoder = None
        return self._cross_encoder

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        ce = self.cross_encoder
        if ce is not None:
            pairs = [[query, c["content"]] for c in candidates]
            scores = ce.predict(pairs)
            for idx, s in enumerate(scores):
                candidates[idx]["rerank_score"] = round(float(s), 4)
            reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        else:
            # Fallback re-weighting using RRF score & domain key terms matching
            for c in candidates:
                base_score = c.get("rrf_score", 0.0) * 50.0
                q_words = set(query.lower().split())
                c_words = set(c["content"].lower().split())
                overlap = len(q_words.intersection(c_words))
                c["rerank_score"] = round(base_score + (overlap * 0.1), 4)
            reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

        return reranked[:top_k]

if __name__ == "__main__":
    reranker = CrossEncoderReranker()
    cands = [
        {"chunk_id": "C1", "content": "Normal room rent is 1% of Sum Insured per day.", "rrf_score": 0.03},
        {"chunk_id": "C2", "content": "Pre-existing diseases waiting period is 48 months.", "rrf_score": 0.02}
    ]
    res = reranker.rerank("What is the room rent limit?", cands, top_k=2)
    print("Reranked:", res)
