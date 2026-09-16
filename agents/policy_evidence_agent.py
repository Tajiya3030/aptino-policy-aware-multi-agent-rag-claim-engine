import time
from typing import Dict, Any, List
from agents.state import ClaimAdjudicationState, ExecutionTrace, RetrievalMetadata
from rag.indexer import PolicyIndexer

class PolicyEvidenceAgent:
    """Agent 2: Dispatches targeted hybrid queries for decision dimensions, retrieving ranked policy clauses."""

    def __init__(self, indexer: PolicyIndexer):
        self.indexer = indexer

    def process(self, state: ClaimAdjudicationState) -> ClaimAdjudicationState:
        t0 = time.time()
        facts = state["extracted_facts"]
        dimensions = state["decision_dimensions"]

        # Formulate multi-query retrieval strategy
        queries = [
            f"Hospitalization room rent limit ICU fees sublimits sum insured {facts['diagnosis']}",
            "Pre-existing disease 48 months waiting period waiver portability",
            "30 days initial waiting period 1 year specific disease list",
            "Domiciliary hospitalization 20% limit non-availability room",
            "Day care treatment less than 24 hours 140 procedures",
            "Exclusions cosmetic surgery experimental unproven treatment",
            "Pre-hospitalization 30 days post-hospitalization 60 days ambulance limit"
        ]

        all_evidence = []
        seen_chunks = set()
        aggregate_metadata = {
            "dense_hits": 0,
            "bm25_hits": 0,
            "rrf_candidates": 0,
            "reranked_top_k": 0,
            "top_score": 0.0
        }

        for q in queries:
            res = self.indexer.search_policy(q, top_k=3)
            meta = res.get("retrieval_metadata", {})
            aggregate_metadata["dense_hits"] += meta.get("dense_hits", 0)
            aggregate_metadata["bm25_hits"] += meta.get("bm25_hits", 0)
            aggregate_metadata["rrf_candidates"] += meta.get("rrf_candidates", 0)
            
            for chunk in res.get("evidence_chunks", []):
                cid = chunk["chunk_id"]
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    all_evidence.append(chunk)

        # Sort combined evidence by rerank_score
        all_evidence.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        top_evidence = all_evidence[:8]
        
        aggregate_metadata["reranked_top_k"] = len(top_evidence)
        if top_evidence:
            aggregate_metadata["top_score"] = top_evidence[0].get("rerank_score", 0.94)

        state["retrieved_evidence"] = top_evidence
        state["retrieval_metadata"] = aggregate_metadata

        latency = int((time.time() - t0) * 1000)
        trace_entry: ExecutionTrace = {
            "agent": "PolicyEvidenceAgent",
            "action": f"Executed hybrid retrieval across {len(queries)} targeted dimension queries",
            "retrieved_chunks": aggregate_metadata["rrf_candidates"],
            "reranked_chunks": len(top_evidence),
            "latency_ms": latency
        }
        state["trace"].append(trace_entry)
        return state
