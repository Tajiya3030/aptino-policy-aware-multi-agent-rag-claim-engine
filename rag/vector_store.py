from typing import List, Dict, Any


class VectorStoreManager:
    """
    Lightweight vector-store interface.

    Dense SentenceTransformer retrieval is disabled for low-memory
    deployment. Retrieval is handled by BM25 in the hybrid pipeline.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "policy_clauses"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.collection = None

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        No-op for lightweight deployment.

        BM25 handles policy retrieval, so Chroma/SentenceTransformer
        embeddings are not loaded.
        """
        return None

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Dense retrieval is disabled.

        Returns an empty list so HybridRetriever can rely on BM25.
        """
        return []