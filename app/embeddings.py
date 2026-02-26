import hashlib
import os

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.fake_mode = os.getenv("EMBEDDER_FAKE", "0") == "1"
        self.model = None if self.fake_mode else SentenceTransformer(model_name)
        self.use_e5_prompting = "multilingual-e5" in model_name.lower() or "/e5" in model_name.lower()

    @staticmethod
    def _fake_embedding(text: str, dims: int = 64) -> list[float]:
        # Deterministic embedding for tests/CI when model downloads are not available.
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        vals = []
        for i in range(dims):
            b = digest[i % len(digest)]
            vals.append((float(b) / 255.0) * 2.0 - 1.0)
        norm = sum(v * v for v in vals) ** 0.5
        if norm > 0:
            vals = [v / norm for v in vals]
        return vals

    def _format_query(self, text: str) -> str:
        if self.use_e5_prompting:
            return f"query: {text}"
        return text

    def _format_passage(self, text: str) -> str:
        if self.use_e5_prompting:
            return f"passage: {text}"
        return text

    def embed_text(self, text: str) -> list[float]:
        if self.fake_mode:
            return self._fake_embedding(self._format_query(text))
        vec = self.model.encode(self._format_query(text), normalize_embeddings=True)
        return vec.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if self.fake_mode:
            return [self._fake_embedding(self._format_passage(t)) for t in texts]
        formatted = [self._format_passage(t) for t in texts]
        vecs = self.model.encode(formatted, normalize_embeddings=True)
        return [v.tolist() for v in vecs]
