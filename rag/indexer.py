import os
from typing import List, Dict, Any
from rag.pdf_parser import PolicyPDFParser
from rag.chunker import PolicyChunker
from rag.vector_store import VectorStoreManager
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker

class PolicyIndexer:
    """Complete policy ingestion, chunking, and index construction pipeline."""

    def __init__(self, pdf_path: str = "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf", chroma_dir: str = "chroma_db"):
        self.pdf_path = pdf_path
        self.chroma_dir = chroma_dir
        self.vector_store = VectorStoreManager(persist_directory=chroma_dir)
        self.bm25 = BM25Retriever()
        self.reranker = CrossEncoderReranker()
        self.hybrid = None
        self.chunks: List[Dict[str, Any]] = []

    def build_index(self) -> List[Dict[str, Any]]:
        print(f"Ingesting policy document from {self.pdf_path}...")
        parser = PolicyPDFParser(self.pdf_path)
        pages = parser.extract_pages()
        
        chunker = PolicyChunker(pages)
        self.chunks = chunker.chunk_policy()
        print(f"Generated {len(self.chunks)} hierarchical policy chunks.")
        
        # Populate Chroma vector store & BM25 sparse index
        self.vector_store.add_chunks(self.chunks)
        self.bm25.index_chunks(self.chunks)
        
        self.hybrid = HybridRetriever(self.vector_store, self.bm25)
        print("Policy index build complete!")
        return self.chunks

    def search_policy(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not self.hybrid:
            self.build_index()
        retrieval_res = self.hybrid.retrieve(query=query, dense_k=10, bm25_k=10, final_candidates_k=12)
        candidates = retrieval_res["candidates"]
        reranked = self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        
        meta = retrieval_res["retrieval_metadata"]
        meta["reranked_top_k"] = len(reranked)
        if reranked:
            meta["top_score"] = reranked[0].get("rerank_score", meta["top_score"])
            
        return {
            "evidence_chunks": reranked,
            "retrieval_metadata": meta
        }

if __name__ == "__main__":
    indexer = PolicyIndexer()
    indexer.build_index()
    res = indexer.search_policy("What is the room rent limit for inpatient hospitalization?")
    print("\nRetrieval Metadata:", res["retrieval_metadata"])
    print("\nTop Evidence Chunk:", res["evidence_chunks"][0] if res["evidence_chunks"] else "None")
