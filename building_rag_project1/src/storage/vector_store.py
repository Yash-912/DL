import os
from typing import List
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        # Configure embedding to use OpenRouter
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model="openai/text-embedding-3-small"
        )
        
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document]):
        """
        Add chunked documents to the ChromaDB index.
        """
        if documents:
            self.db.add_documents(documents)
            # chroma handles persistence automatically in newer versions

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve top-k most similar chunks for a given query.
        """
        return self.db.similarity_search(query, k=k)
