import time
from typing import List
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory

        # Use Featherless (Qwen) embeddings for both indexing and query
        from src.embeddings.featherless import FeatherlessEmbeddings
        self.embeddings = FeatherlessEmbeddings()
        
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document]):
        """
        Add chunked documents to the ChromaDB index.
        """
        if not documents:
            return

        # Retry on transient embedding failures
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                self.db.add_documents(documents)
                return
            except ValueError as exc:
                if "No embedding data received" in str(exc) and attempt < attempts:
                    time.sleep(2.0 * attempt)
                    continue
                raise
            # chroma handles persistence automatically in newer versions

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top-k most similar chunks for a given query.
        """
        return self.db.similarity_search(query, k=k)
