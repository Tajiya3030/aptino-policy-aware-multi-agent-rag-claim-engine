import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from rag.embeddings import DenseEmbeddingModel

class VectorStoreManager:
    """Manages ChromaDB vector store collection for policy chunk search."""

    def __init__(self, persist_directory: str = "chroma_db", collection_name: str = "policy_clauses"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = DenseEmbeddingModel()
        
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
            
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "page": c["page"],
                "section": c["section"],
                "clause": c["clause"]
            }
            for c in chunks
        ]
        
        embeddings = self.embedding_model.encode(documents)
        
        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_vec = self.embedding_model.encode([query])
        results = self.collection.query(
            query_embeddings=query_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        if results and results.get("ids") and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][idx]
                doc = results["documents"][0][idx]
                meta = results["metadatas"][0][idx]
                dist = results["distances"][0][idx]
                # Convert cosine distance to similarity score
                score = round(1.0 / (1.0 + float(dist)), 4)
                
                output.append({
                    "chunk_id": chunk_id,
                    "content": doc,
                    "source": meta.get("source", "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"),
                    "page": meta.get("page", 1),
                    "section": meta.get("section", "POLICY"),
                    "clause": meta.get("clause", ""),
                    "dense_score": score
                })
        return output

if __name__ == "__main__":
    vdb = VectorStoreManager()
    sample = [{
        "chunk_id": "TEST-01",
        "source": "USGIC-CSCIndividualHealthInsurance_2017-2018.pdf",
        "page": 7,
        "section": "SCOPE OF COVER",
        "clause": "Room Rent Limit",
        "content": "Normal Room expenses: 1.0% of Basic Sum Insured per day."
    }]
    vdb.add_chunks(sample)
    hits = vdb.search("room rent percentage limit", top_k=1)
    print("Vector Search Hit:", hits)
