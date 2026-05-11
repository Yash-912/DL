import hashlib
from typing import List, Dict, Any
from datetime import datetime, timezone
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def chunk(self, parsed_data: List[Dict[str, Any]]) -> List[Document]:
        """
        Takes parsed data elements and chunks them.
        """
        if not parsed_data:
            return []
            
        # Group text by page number to keep some continuity, or just join all text for the document
        # For simplicity in MVP, we'll combine text from the same document but process by element
        # Actually, it's better to reconstruct the document string with page markers or just split the whole text
        
        # Let's combine all text and then split, but we lose page numbers that way.
        # Alternatively, split each parsed element individually, though elements might be small.
        # If unstructured gives us paragraph-level elements, we can combine them into a single string
        # while keeping track of page numbers, or just split element by element.
        # Unstructured usually splits text into small blocks (e.g. paragraphs).
        
        # We will create Langchain Document objects and then split them
        docs = []
        for item in parsed_data:
            docs.append(Document(page_content=item["text"], metadata=item["metadata"]))
            
        # If the parsed data is already small (paragraph level), the text splitter will just wrap them
        # if they are large, it will split them.
        # For better context, we might want to group elements before splitting, but for MVP this is fine.
        print(docs)
        chunked_docs = self.splitter.split_documents(docs)
        
        # Add additional metadata
        ingested_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        total_chunks = len(chunked_docs)
        
        for i, doc in enumerate(chunked_docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["ingested_at"] = ingested_at
            doc.metadata["total_chunks"] = total_chunks
            content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
            doc.metadata["content_hash"] = content_hash
            
        return chunked_docs
