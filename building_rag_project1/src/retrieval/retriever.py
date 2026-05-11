import hashlib
from typing import Dict, List, Any

from src.storage.vector_store import VectorStore
from src.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from src.retrieval.query_transformer import QueryTransformer, QueryVariant

class DocumentRetriever:
    def __init__(self, vector_store: VectorStore, transformer: QueryTransformer | None = None):
        self.vector_store = vector_store
        self.transformer = transformer or QueryTransformer()
        self.embeddings = SentenceTransformerEmbeddings()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        transform_mode: str = "none",
        alpha: float = 0.5,
        rrf_k: int = 60,
        multi_count: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant chunks using hybrid search + query transforms.
        """
        variants = self._build_variants(query, transform_mode, multi_count)
        result_lists: List[List] = []

        for variant in variants:
            if variant.kind == "hyde":
                vector = self.embeddings.embed_query(variant.text)
                docs = self.vector_store.similarity_search_by_vector(vector, k=top_k)
            else:
                docs = self.vector_store.hybrid_search(variant.text, k=top_k, alpha=alpha)
            result_lists.append(docs)

        fused_docs = self._rrf_fuse(result_lists, rrf_k)
        return [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in fused_docs[:top_k]
        ]

    def _build_variants(self, query: str, mode: str, multi_count: int) -> List[QueryVariant]:
        if mode == "none":
            return [QueryVariant(text=query, kind="original")]
        return self.transformer.build_variants(query, mode=mode, multi_count=multi_count)

    def _rrf_fuse(self, result_lists: List[List], k: int) -> List:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}

        for docs in result_lists:
            for rank, doc in enumerate(docs, start=1):
                doc_id = self._doc_id(doc)
                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
                doc_map[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in ranked]

    def _doc_id(self, doc) -> str:
        metadata = doc.metadata or {}
        if metadata.get("chunk_id"):
            return str(metadata["chunk_id"])
        if metadata.get("content_hash"):
            return str(metadata["content_hash"])
        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        return content_hash
