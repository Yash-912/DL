import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from src.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.embeddings.cache import EmbeddingCache


class VectorStore:
    def __init__(self, collection_name: Optional[str] = None):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        self.index_name = os.getenv("PINECONE_INDEX", "rag-index")
        self.cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.region = os.getenv("PINECONE_REGION", "us-east-1")
        self.namespace = collection_name or os.getenv("PINECONE_NAMESPACE", "default")

        self.embeddings = SentenceTransformerEmbeddings()
        cache_path = os.getenv("EMBEDDING_CACHE_PATH", "embedding_cache.sqlite")
        self.cache = EmbeddingCache(db_path=Path(cache_path).resolve())

        self.client = Pinecone(api_key=api_key)
        self._ensure_index()
        self.index = self.client.Index(self.index_name)

    def _ensure_index(self) -> None:
        if self.index_name in self.client.list_indexes().names():
            return

        dimension = self._embedding_dimension()
        self.client.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )

    def _embedding_dimension(self) -> int:
        sample_vector = self.embeddings.embed_query("dimension probe")
        return len(sample_vector)

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
        batch = []
        for doc in documents:
            metadata = dict(doc.metadata)
            content_hash = metadata.get("content_hash") or self._content_hash(doc.page_content)
            source_file = metadata.get("source_file", "")
            chunk_index = metadata.get("chunk_index", 0)
            stable_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{content_hash}:{source_file}:{chunk_index}")

            metadata.setdefault("chunk_id", str(stable_id))
            metadata["content_hash"] = content_hash

            vector = vectors.get(content_hash)
            if vector is None:
                continue

            batch.append(
                (
                    str(stable_id),
                    vector,
                    self._clean_metadata({
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
                    }),
                )
            )

        if batch:
            self.index.upsert(vectors=batch, namespace=self.namespace)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        vector = self.embeddings.embed_query(query)
        return self.similarity_search_by_vector(vector, k=k)

    def similarity_search_by_vector(self, vector: List[float], k: int = 5) -> List[Document]:
        response = self.index.query(
            namespace=self.namespace,
            vector=vector,
            top_k=k,
            include_metadata=True,
        )
        return self._to_documents(response)

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
        # Pinecone dense-only for now; alpha kept for compatibility with retriever.
        _ = alpha
        return self.similarity_search(query, k=k)

    def delete_collection(self) -> None:
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
        except Exception:
            pass

    def _to_documents(self, response) -> List[Document]:
        matches = getattr(response, "matches", None) or response.get("matches", [])
        results: List[Document] = []
        for match in matches:
            metadata = match.get("metadata") if isinstance(match, dict) else match.metadata
            if not metadata:
                continue
            text = metadata.get("text", "")
            clean_metadata = dict(metadata)
            clean_metadata.pop("text", None)
            results.append(Document(page_content=text, metadata=clean_metadata))
        return results

    def _clean_metadata(self, metadata: Dict[str, Optional[object]]) -> Dict[str, object]:
        return {key: value for key, value in metadata.items() if value is not None}
