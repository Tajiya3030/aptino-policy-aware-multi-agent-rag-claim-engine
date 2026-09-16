from typing import List
import numpy as np

class DenseEmbeddingModel:
    """Provides dense vector embeddings using SentenceTransformers with fallback."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

if __name__ == "__main__":
    emb = DenseEmbeddingModel()
    res = emb.encode(["Hospitalization expenses room rent limit", "Pre-existing disease waiting period"])
    print(f"Encoded {len(res)} sentences into vectors of dimension {len(res[0])}.")
