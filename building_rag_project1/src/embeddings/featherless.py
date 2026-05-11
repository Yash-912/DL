import os
import requests
from typing import List


class FeatherlessEmbeddings:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FEATHERLESS_API_KEY")
        if not self.api_key:
            raise ValueError("FEATHERLESS_API_KEY environment variable not set")

        self.model = model or os.getenv("FEATHERLESS_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
        self.base_url = os.getenv("FEATHERLESS_API_BASE", "https://api.featherless.ai/v1/embeddings")
        self.session = requests.Session()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": inputs
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = self.session.post(self.base_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "data" not in data:
            raise ValueError(f"Unexpected embeddings response: {data}")

        return [item["embedding"] for item in data["data"]]
