'''
Chunking Strategy Comparison Lab
=================================
This file contains multiple chunking strategies to experiment with and compare.
Each class takes the same input (parsed_data from parser.py) and returns the
same output (List[Document] with metadata).

Strategies implemented:
1. FixedSizeChunker       — Pure Python, sliding window over characters
2. SentenceChunker        — Splits on sentence boundaries using nltk
3. SemanticChunker        — Groups sentences by embedding similarity (finds topic shifts)
4. RecursiveCharChunker   — LangChain's RecursiveCharacterTextSplitter (your original)

Advanced (Phase 2, not implemented yet):
- Document-aware (split on headings/sections)
- Parent-child (small retrieval chunks, large context chunks)
- Late chunking
- RAPTOR
'''
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helper: attach standard metadata to a list of Documents
# ---------------------------------------------------------------------------
def _add_metadata(chunks: List[Document]) -> List[Document]:
    """Add chunk_index and ingested_at to every chunk."""
    ingested_at = datetime.utcnow().isoformat()
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_index"] = i
        doc.metadata["ingested_at"] = ingested_at
    return chunks


# ===========================================================================
# Strategy 1: Fixed Size Chunker (Pure Python — no libraries)
# ===========================================================================
class FixedSizeChunker:
    """
    The simplest possible chunker.
    Slides a window of `chunk_size` characters across the text,
    stepping forward by `chunk_size - overlap` each time.

    WHY IT EXISTS:
    - Easy to understand and implement
    - Predictable chunk sizes

    WHERE IT FAILS:
    - Cuts mid-sentence, mid-word — chunks lose meaning at boundaries
    - Overlap helps, but doesn't fully solve the boundary problem
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, parsed_data: List[Dict[str, Any]]) -> List[Document]:
        if not parsed_data:
            return []

        all_chunks = []

        for item in parsed_data:
            text = item["text"]
            metadata = item["metadata"]
            step = self.chunk_size - self.overlap

            # Slide a window across the text
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()

                if chunk_text:  # skip empty chunks
                    all_chunks.append(
                        Document(
                            page_content=chunk_text,
                            metadata={**metadata, "strategy": "fixed_size"}
                        )
                    )
                start += step

        return _add_metadata(all_chunks)


# ===========================================================================
# Strategy 2: Sentence Chunker (nltk sentence tokenizer)
# ===========================================================================
class SentenceChunker:
    """
    Splits text into sentences first, then groups sentences until
    the group reaches `max_chunk_size` characters.

    WHY IT EXISTS:
    - Never cuts mid-sentence — each chunk is semantically complete
    - Respects natural language boundaries

    WHERE IT FAILS:
    - Very long sentences can exceed max_chunk_size
    - Doesn't understand topic shifts — groups unrelated sentences if they fit
    
    REQUIRES: pip install nltk
    (First run will download 'punkt_tab' tokenizer data automatically)
    """

    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size

    def chunk(self, parsed_data: List[Dict[str, Any]]) -> List[Document]:
        import nltk
        try:
            # Try to tokenize a dummy sentence to see if punkt is installed
            nltk.sent_tokenize("Test.")
        except Exception:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        if not parsed_data:
            return []

        all_chunks = []

        for item in parsed_data:
            text = item["text"]
            metadata = item["metadata"]

            # Split into individual sentences
            sentences = nltk.sent_tokenize(text)

            # Group sentences into chunks that fit within max_chunk_size
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                # If adding this sentence would exceed the limit, save current chunk
                if current_length + len(sentence) > self.max_chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk).strip()
                    all_chunks.append(
                        Document(
                            page_content=chunk_text,
                            metadata={**metadata, "strategy": "sentence"}
                        )
                    )
                    current_chunk = []
                    current_length = 0

                current_chunk.append(sentence)
                current_length += len(sentence)

            # Don't forget the last chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                all_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={**metadata, "strategy": "sentence"}
                    )
                )

        return _add_metadata(all_chunks)


# ===========================================================================
# Strategy 3: Semantic Chunker (embedding similarity between sentences)
# ===========================================================================
class SemanticChunker:
    """
    Splits text into sentences, embeds each one, then groups consecutive
    sentences that are semantically similar. A new chunk starts when the
    cosine similarity between adjacent sentences drops below a threshold.

    WHY IT EXISTS:
    - Chunks align with actual topic boundaries, not arbitrary character counts
    - Best retrieval precision — each chunk is about ONE topic

    WHERE IT FAILS:
    - Slow at ingestion (needs to embed every sentence individually)
    - Costs money if using API embeddings
    - Threshold tuning is tricky — too high = too many tiny chunks, too low = no splits

    REQUIRES: pip install nltk numpy
    Also needs your embedding model to be available (uses OpenRouter).
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold

    def _cosine_similarity(self, vec_a, vec_b):
        """Compute cosine similarity between two vectors using numpy."""
        import numpy as np
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def chunk(self, parsed_data: List[Dict[str, Any]], embeddings_model=None) -> List[Document]:
        """
        NOTE: This chunker needs an embeddings model to compute sentence vectors.
        Pass your VectorStore's embeddings object, e.g.:
            from src.storage.vector_store import VectorStore
            vs = VectorStore()
            semantic_chunker.chunk(parsed_data, embeddings_model=vs.embeddings)
        """
        import nltk
        try:
            # Try to tokenize a dummy sentence to see if punkt is installed
            nltk.sent_tokenize("Test.")
        except Exception:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        if not parsed_data:
            return []

        if embeddings_model is None:
            raise ValueError(
                "SemanticChunker requires an embeddings model. "
                "Pass it as: chunker.chunk(data, embeddings_model=your_embeddings)"
            )

        all_chunks = []

        for item in parsed_data:
            text = item["text"]
            metadata = item["metadata"]

            # Step 1: Split into sentences
            sentences = nltk.sent_tokenize(text)
            if len(sentences) <= 1:
                all_chunks.append(
                    Document(
                        page_content=text.strip(),
                        metadata={**metadata, "strategy": "semantic"}
                    )
                )
                continue

            # Step 2: Embed every sentence
            print(f"  Embedding {len(sentences)} sentences for semantic chunking...")
            sentence_embeddings = embeddings_model.embed_documents(sentences)

            # Step 3: Find topic boundaries by comparing adjacent sentences
            # If similarity drops below threshold → new chunk starts here
            groups = []
            current_group = [sentences[0]]

            for i in range(1, len(sentences)):
                sim = self._cosine_similarity(
                    sentence_embeddings[i - 1],
                    sentence_embeddings[i]
                )
                if sim < self.similarity_threshold:
                    # Topic shift detected — save current group, start new one
                    groups.append(current_group)
                    current_group = [sentences[i]]
                else:
                    # Same topic — keep grouping
                    current_group.append(sentences[i])

            # Don't forget the last group
            groups.append(current_group)

            # Step 4: Convert groups to Document objects
            for group in groups:
                chunk_text = " ".join(group).strip()
                if chunk_text:
                    all_chunks.append(
                        Document(
                            page_content=chunk_text,
                            metadata={**metadata, "strategy": "semantic"}
                        )
                    )

        return _add_metadata(all_chunks)


# ===========================================================================
# Strategy 4: Recursive Character Chunker (LangChain — your original)
# ===========================================================================
class RecursiveCharChunker:
    """
    Uses LangChain's RecursiveCharacterTextSplitter.
    Tries to split on paragraph breaks first, then newlines, then sentences,
    then spaces, then individual characters — in that priority order.

    WHY IT EXISTS:
    - Good balance between respecting boundaries and controlling chunk size
    - The most commonly used strategy in production RAG systems

    WHERE IT FAILS:
    - Still character-count based — doesn't understand semantics
    - Can split mid-paragraph if the paragraph is too long
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        # pyrefly: ignore [missing-import]
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def chunk(self, parsed_data: List[Dict[str, Any]]) -> List[Document]:
        if not parsed_data:
            return []

        docs = []
        for item in parsed_data:
            docs.append(Document(page_content=item["text"], metadata=item["metadata"]))

        chunked_docs = self.splitter.split_documents(docs)

        # Tag with strategy name
        for doc in chunked_docs:
            doc.metadata["strategy"] = "recursive_char"

        return _add_metadata(chunked_docs)
