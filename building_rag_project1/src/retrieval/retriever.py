from typing import List, Dict, Any
from src.storage.vector_store import VectorStore

class DocumentRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most relevant chunks for the given query.
        Formats the output for the generation layer.
        """
        docs = self.vector_store.similarity_search(query, k=top_k)
        
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
            
        return results
