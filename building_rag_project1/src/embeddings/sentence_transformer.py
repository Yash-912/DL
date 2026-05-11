from typing import List, Optional

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )[0]
        return vector.tolist()
