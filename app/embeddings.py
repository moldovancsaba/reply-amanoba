from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.use_e5_prompting = "multilingual-e5" in model_name.lower() or "/e5" in model_name.lower()

    def _format_query(self, text: str) -> str:
        if self.use_e5_prompting:
            return f"query: {text}"
        return text

    def _format_passage(self, text: str) -> str:
        if self.use_e5_prompting:
            return f"passage: {text}"
        return text

    def embed_text(self, text: str) -> list[float]:
        vec = self.model.encode(self._format_query(text), normalize_embeddings=True)
        return vec.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        formatted = [self._format_passage(t) for t in texts]
        vecs = self.model.encode(formatted, normalize_embeddings=True)
        return [v.tolist() for v in vecs]
