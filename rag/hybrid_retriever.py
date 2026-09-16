from typing import List, Dict, Any

class HybridRetriever:
    """
    Combines dense semantic vector hits (ChromaDB) and sparse lexical hits (BM25)
    using Reciprocal Rank Fusion (RRF) with formula: score(d) = sum(1 / (60 + rank)).
    """

    def __init__(self, vector_store, bm25_retriever, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, dense_k: int = 10, bm25_k: int = 10, final_candidates_k: int = 12) -> Dict[str, Any]:
        """
        Executes dense and sparse retrieval, combines ranks with RRF,
        and returns both candidates list and retrieval metadata statistics.
        """
        dense_results = self.vector_store.search(query, top_k=dense_k)
        bm25_results = self.bm25_retriever.search(query, top_k=bm25_k)

        # Build map of chunk_id -> combined candidate info & RRF score calculation
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process dense ranks
        for rank, hit in enumerate(dense_results):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            if cid not in chunk_map:
                chunk_map[cid] = hit.copy()
            chunk_map[cid]["dense_rank"] = rank + 1

        # Process BM25 ranks
        for rank, hit in enumerate(bm25_results):
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            if cid not in chunk_map:
                chunk_map[cid] = hit.copy()
            chunk_map[cid]["bm25_rank"] = rank + 1

        # Sort candidate chunks by RRF score descending
        sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        
        candidates = []
        for cid in sorted_cids[:final_candidates_k]:
            item = chunk_map[cid]
            item["rrf_score"] = round(scores[cid], 5)
            candidates.append(item)

        retrieval_metadata = {
            "query": query,
            "dense_hits": len(dense_results),
            "bm25_hits": len(bm25_results),
            "rrf_candidates": len(candidates),
            "reranked_top_k": min(5, len(candidates)),
            "top_score": candidates[0]["rrf_score"] if candidates else 0.0
        }

        return {
            "candidates": candidates,
            "retrieval_metadata": retrieval_metadata
        }
