from typing import List
import re
import math
from collections import Counter


class DenseEmbeddingModel:
    """
    Lightweight CPU-only embedding replacement for low-memory deployment.

    Uses TF-IDF-style sparse vectors instead of SentenceTransformers/PyTorch.
    This avoids loading large ML models on Render.
    """

    def __init__(self, model_name: str = "lightweight-tfidf"):
        self.model_name = model_name
        self._vocabulary = {}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def _build_vocabulary(self, texts: List[str]):
        vocab = set()

        for text in texts:
            vocab.update(self._tokenize(text))

        self._vocabulary = {
            word: idx for idx, word in enumerate(sorted(vocab))
        }

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Build vocabulary from the supplied texts.
        self._build_vocabulary(texts)

        vectors = []

        for text in texts:
            tokens = self._tokenize(text)
            counts = Counter(tokens)

            total = max(len(tokens), 1)

            vector = [0.0] * len(self._vocabulary)

            for word, count in counts.items():
                if word in self._vocabulary:
                    idx = self._vocabulary[word]
                    vector[idx] = count / total

            # L2 normalize
            norm = math.sqrt(sum(x * x for x in vector))

            if norm > 0:
                vector = [x / norm for x in vector]

            vectors.append(vector)

        return vectors


if __name__ == "__main__":
    emb = DenseEmbeddingModel()

    res = emb.encode([
        "Hospitalization expenses room rent limit",
        "Pre-existing disease waiting period"
    ])

    print(
        f"Encoded {len(res)} sentences into vectors "
        f"of dimension {len(res[0])}."
    )