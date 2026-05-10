import os
import sys

# Add root project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concept_building.chunker import (
    FixedSizeChunker,
    SentenceChunker,
    SemanticChunker,
    RecursiveCharChunker,
)

def load_demo_data():
    """Load the HR leave policy as simulated parser output."""
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hr_leave_policy.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return [
        {
            "text": content,
            "metadata": {
                "source_file": "hr_leave_policy.txt",
                "page_number": 1,
                "doc_type": "txt"
            }
        }
    ]

def print_chunks(chunks, strategy_name):
    """Print a summary of chunks from a given strategy."""
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    
    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    print(f"Total characters: {total_chars}")
    print(f"Avg chunk size: {avg_chars:.0f} chars")
    
    # for i, chunk in enumerate(chunks):
    #     print(f"\n  [Chunk {i}] ({len(chunk.page_content)} chars)")
    #     # Show first 120 chars of each chunk for quick comparison
    #     preview = chunk.page_content[:120].replace("\n", " ")
    #     print(f"  Preview: {preview}...")
    #     print(f"  Metadata: {chunk.metadata}")


def run_comparison():
    data = load_demo_data()
    print(f"Loaded document: {len(data[0]['text'])} characters total\n")

    # --- Strategy 1: Fixed Size ---
    fixed = FixedSizeChunker(chunk_size=200, overlap=30)
    fixed_chunks = fixed.chunk(data)
    print_chunks(fixed_chunks, "Fixed Size (200 chars, 30 overlap)")

    # --- Strategy 2: Sentence ---
    sentence = SentenceChunker(max_chunk_size=200)
    sentence_chunks = sentence.chunk(data)
    print_chunks(sentence_chunks, "Sentence (max 200 chars)")

    # --- Strategy 3: Recursive Character ---
    recursive = RecursiveCharChunker(chunk_size=200, overlap=30)
    recursive_chunks = recursive.chunk(data)
    print_chunks(recursive_chunks, "Recursive Character (200 chars, 30 overlap)")

    # --- Strategy 4: Semantic (FREE local embeddings, no API cost) ---
    print("\n\nLoading local HuggingFace embedding model (first run downloads ~90MB)...")
    # Disable TensorFlow to prevent protobuf import crashes, force PyTorch only
    os.environ["USE_TF"] = "0"
    os.environ["USE_TENSORFLOW"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    semantic = SemanticChunker(similarity_threshold=0.75)
    semantic_chunks = semantic.chunk(data, embeddings_model=embeddings)
    print_chunks(semantic_chunks, "Semantic (threshold=0.75, local model)")

    # --- Quick comparison table ---
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<35} {'Chunks':<10} {'Avg Size':<10}")
    print(f"{'-'*55}")
    for name, chunks in [
        ("Fixed Size", fixed_chunks),
        ("Sentence", sentence_chunks),
        ("Recursive Character", recursive_chunks),
        ("Semantic", semantic_chunks),
    ]:
        avg = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
        print(f"{name:<35} {len(chunks):<10} {avg:<10.0f}")


if __name__ == "__main__":
    run_comparison()
