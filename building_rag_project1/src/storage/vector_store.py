import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import weaviate
from weaviate.auth import AuthApiKey
from langchain_core.documents import Document

from src.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.embeddings.cache import EmbeddingCache


class VectorStore:
    def __init__(self, collection_name: Optional[str] = None):
        self.url = os.getenv("WEAVIATE_URL")
        if not self.url:
            raise ValueError("WEAVIATE_URL environment variable not set")

        if not self.url.startswith("http"):
            self.url = f"https://{self.url}"

        api_key = os.getenv("WEAVIATE_API_KEY")
        auth = AuthApiKey(api_key) if api_key else None

        self.class_name = collection_name or os.getenv("WEAVIATE_CLASS", "DocumentChunk")
        try:
            # Weaviate client v3
            self.client = weaviate.Client(self.url, auth_client_secret=auth)
        except TypeError as exc:
            raise RuntimeError(
                "Incompatible weaviate-client version detected. "
                "Install weaviate-client<4 to use the current storage adapter."
            ) from exc

        self.embeddings = SentenceTransformerEmbeddings()
        cache_path = os.getenv("EMBEDDING_CACHE_PATH", "embedding_cache.sqlite")
        self.cache = EmbeddingCache(db_path=Path(cache_path).resolve())

        self._ensure_schema()

    def _ensure_schema(self) -> None:
        schema = self.client.schema.get()
        if any(entry.get("class") == self.class_name for entry in schema.get("classes", [])):
            return

        class_schema = {
            "class": self.class_name,
            "description": "RAG chunk collection",
            "vectorizer": "none",
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "chunk_id", "dataType": ["text"]},
                {"name": "content_hash", "dataType": ["text"]},
                {"name": "source_file", "dataType": ["text"]},
                {"name": "page_number", "dataType": ["int"]},
                {"name": "chunk_index", "dataType": ["int"]},
                {"name": "total_chunks", "dataType": ["int"]},
                {"name": "doc_type", "dataType": ["text"]},
                {"name": "strategy", "dataType": ["text"]},
                {"name": "parent_id", "dataType": ["text"]},
                {"name": "ingested_at", "dataType": ["date"]},
            ],
        }
        self.client.schema.create_class(class_schema)

    def _content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _prepare_vectors(self, documents: List[Document]) -> Dict[str, List[float]]:
        content_hashes = []
        text_map: Dict[str, str] = {}

        for doc in documents:
            content_hash = doc.metadata.get("content_hash") or self._content_hash(doc.page_content)
            doc.metadata["content_hash"] = content_hash
            content_hashes.append(content_hash)
            text_map[content_hash] = doc.page_content

        cached = self.cache.get_many(content_hashes)
        missing_hashes = [hash_value for hash_value in content_hashes if hash_value not in cached]

        if missing_hashes:
            missing_texts = [text_map[hash_value] for hash_value in missing_hashes]
            vectors = self.embeddings.embed_documents(missing_texts)
            new_vectors = dict(zip(missing_hashes, vectors))
            self.cache.set_many(new_vectors)
            cached.update(new_vectors)

        return cached

    def add_documents(self, documents: List[Document]):
        if not documents:
            return

        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                vectors = self._prepare_vectors(documents)
                self._batch_upsert(documents, vectors)
                return
            except Exception:
                if attempt < attempts:
                    time.sleep(2.0 * attempt)
                    continue
                raise

    def _batch_upsert(self, documents: List[Document], vectors: Dict[str, List[float]]) -> None:
        self.client.batch.configure(batch_size=64)
        with self.client.batch as batch:
            for doc in documents:
                metadata = dict(doc.metadata)
                content_hash = metadata.get("content_hash") or self._content_hash(doc.page_content)
                source_file = metadata.get("source_file", "")
                chunk_index = metadata.get("chunk_index", 0)
                stable_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{content_hash}:{source_file}:{chunk_index}")

                metadata.setdefault("chunk_id", str(stable_id))
                metadata["content_hash"] = content_hash

                properties = {
                    "text": doc.page_content,
                    "chunk_id": metadata.get("chunk_id"),
                    "content_hash": content_hash,
                    "source_file": metadata.get("source_file"),
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata.get("chunk_index"),
                    "total_chunks": metadata.get("total_chunks"),
                    "doc_type": metadata.get("doc_type"),
                    "strategy": metadata.get("strategy"),
                    "parent_id": metadata.get("parent_id"),
                    "ingested_at": metadata.get("ingested_at"),
                }

                vector = vectors.get(content_hash)
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    uuid=str(stable_id),
                    vector=vector,
                )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        vector = self.embeddings.embed_query(query)
        return self.similarity_search_by_vector(vector, k=k)

    def similarity_search_by_vector(self, vector: List[float], k: int = 5) -> List[Document]:
        response = (
            self.client.query.get(self.class_name, self._return_fields())
            .with_near_vector({"vector": vector})
            .with_limit(k)
            .do()
        )
        return self._to_documents(response)

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
        response = (
            self.client.query.get(self.class_name, self._return_fields())
            .with_hybrid(query=query, alpha=alpha)
            .with_limit(k)
            .do()
        )
        return self._to_documents(response)

    def delete_collection(self) -> None:
        if self.client.schema.exists(self.class_name):
            self.client.schema.delete_class(self.class_name)

    def _return_fields(self) -> List[str]:
        return [
            "text",
            "chunk_id",
            "content_hash",
            "source_file",
            "page_number",
            "chunk_index",
            "total_chunks",
            "doc_type",
            "strategy",
            "parent_id",
            "ingested_at",
        ]

    def _to_documents(self, response: Dict[str, Dict]) -> List[Document]:
        data = response.get("data", {}).get("Get", {}).get(self.class_name)
        if not data:
            return []
        results: List[Document] = []
        for item in data:
            metadata = dict(item)
            text = metadata.pop("text", "")
            results.append(Document(page_content=text, metadata=metadata))
        return results
