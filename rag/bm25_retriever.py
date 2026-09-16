import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """Sparse lexical search using BM25Okapi over policy clause tokens."""

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi = None

    def _tokenize(self, text: str) -> List[str]:
        # Split into alphanumeric tokens, ignoring pure symbols
        words = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return [w for w in words if len(w) > 0]

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        self.corpus_tokens = [self._tokenize(c["content"]) for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Rank by score descending
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            c = self.chunks[idx].copy()
            c["bm25_score"] = round(score, 4)
            results.append(c)
            
        return results

if __name__ == "__main__":
    bm25 = BM25Retriever()
    sample = [
        {"chunk_id": "C1", "content": "Normal room rent limit is 1% of Sum Insured.", "page": 7, "section": "SCOPE", "clause": "Room Rent", "source": "policy.pdf"},
        {"chunk_id": "C2", "content": "Intensive Care Unit (ICU) sublimit is 2% per day.", "page": 7, "section": "SCOPE", "clause": "ICU Rent", "source": "policy.pdf"}
    ]
    bm25.index_chunks(sample)
    print("BM25 Hits:", bm25.search("room rent 1%"))
