import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.embeddings.featherless import FeatherlessEmbeddings

# Allow running from the scripts/ directory by adding project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.parser import DocumentParser
from src.storage.vector_store import VectorStore
from src.generation.generator import AnswerGenerator


@dataclass
class ParsedDoc:
    source_file: str
    doc_type: str
    text: str


def load_documents(data_dir: str) -> List[ParsedDoc]:
    parser = DocumentParser()
    docs: List[ParsedDoc] = []

    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            parsed = parser.parse(path)
        except Exception:
            continue

        if not parsed:
            continue

        text = "\n\n".join([item["text"] for item in parsed if item.get("text")])
        if not text.strip():
            continue

        doc_type = parsed[0]["metadata"].get("doc_type", "unknown")
        docs.append(ParsedDoc(source_file=os.path.basename(path), doc_type=doc_type, text=text))

    return docs


def fixed_size_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - chunk_overlap)

    return chunks


def sentence_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) + 1 > chunk_size and current:
            chunks.append(" ".join(current).strip())
            if chunk_overlap > 0:
                overlap_text = " ".join(current)[-chunk_overlap:]
                current = [overlap_text] if overlap_text.strip() else []
                current_len = len(overlap_text)
            else:
                current = []
                current_len = 0

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def semantic_chunks(text: str, chunk_size: int, chunk_overlap: int, embeddings: FeatherlessEmbeddings) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    vectors = embeddings.embed_documents(sentences)
    distances: List[float] = []

    for i in range(1, len(vectors)):
        a = vectors[i - 1]
        b = vectors[i]
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            distances.append(1.0)
        else:
            cosine = dot / (norm_a * norm_b)
            distances.append(1.0 - cosine)

    mean = sum(distances) / len(distances)
    variance = sum((d - mean) ** 2 for d in distances) / len(distances)
    threshold = mean + (variance ** 0.5) * 0.5

    chunks: List[str] = []
    current: List[str] = [sentences[0]]
    current_len = len(sentences[0])

    for idx in range(1, len(sentences)):
        distance = distances[idx - 1]
        sentence = sentences[idx]
        should_split = distance > threshold or (current_len + len(sentence) + 1 > chunk_size)

        if should_split and current:
            chunks.append(" ".join(current).strip())
            if chunk_overlap > 0:
                overlap_text = " ".join(current)[-chunk_overlap:]
                current = [overlap_text] if overlap_text.strip() else []
                current_len = len(overlap_text)
            else:
                current = []
                current_len = 0

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def recursive_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs if doc.page_content.strip()]


def build_documents(parsed_docs: List[ParsedDoc], strategy: str, chunk_size: int, chunk_overlap: int,
                    embeddings: FeatherlessEmbeddings) -> Tuple[List[Document], int]:
    chunked_docs: List[Document] = []
    total_chunks = 0

    for doc in parsed_docs:
        if strategy == "fixed":
            chunks = fixed_size_chunks(doc.text, chunk_size, chunk_overlap)
        elif strategy == "sentence":
            chunks = sentence_chunks(doc.text, chunk_size, chunk_overlap)
        elif strategy == "semantic":
            chunks = semantic_chunks(doc.text, chunk_size, chunk_overlap, embeddings)
        elif strategy == "recursive":
            chunks = recursive_chunks(doc.text, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for idx, chunk in enumerate(chunks):
            metadata = {
                "source_file": doc.source_file,
                "doc_type": doc.doc_type,
                "chunk_index": idx
            }
            chunked_docs.append(Document(page_content=chunk, metadata=metadata))

        total_chunks += len(chunks)

    return chunked_docs, total_chunks


def load_questions(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def evaluate_strategy(name: str, vector_store: VectorStore, questions: List[Dict[str, str]],
                      k: int, generate_answers: bool) -> Dict[str, float]:
    hits = 0
    mrr_total = 0.0
    answer_hits = 0
    total_retrieved_chars = 0
    total_latency = 0.0

    generator = AnswerGenerator() if generate_answers else None

    for item in questions:
        question = item["question"]
        ground_truth = item["ground_truth"]

        start = time.perf_counter()
        docs = vector_store.similarity_search(question, k=k)
        total_latency += time.perf_counter() - start

        retrieved_texts = [doc.page_content for doc in docs]
        total_retrieved_chars += sum(len(t) for t in retrieved_texts)

        rank = None
        for idx, text in enumerate(retrieved_texts, start=1):
            if ground_truth.lower() in text.lower():
                rank = idx
                break

        if rank is not None:
            hits += 1
            mrr_total += 1.0 / rank

        if generate_answers and generator is not None:
            context_chunks = [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ]
            answer, _ = generator.generate(question, context_chunks)
            if ground_truth.lower() in answer.lower():
                answer_hits += 1

    total = max(1, len(questions))

    return {
        "strategy": name,
        "recall_at_k": hits / total,
        "mrr": mrr_total / total,
        "answer_match": (answer_hits / total) if generate_answers else -1.0,
        "avg_retrieved_chars": total_retrieved_chars / total,
        "avg_retrieval_latency_s": total_latency / total
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate chunking strategies for RAG retrieval.")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--questions", default=os.path.join(os.path.dirname(__file__), "eval_questions.json"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--strategies", default="fixed,sentence,recursive,semantic")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument(
        "--sweep",
        default="",
        help="Comma-separated size:overlap pairs, e.g. 300:30,500:50,800:100"
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--generate-answers", action="store_true")

    args = parser.parse_args()

    parsed_docs = load_documents(args.data_dir)
    if not parsed_docs:
        raise SystemExit("No documents found to evaluate.")

    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit("No evaluation questions found.")

    if not os.getenv("FEATHERLESS_API_KEY"):
        raise SystemExit("Missing API key. Set FEATHERLESS_API_KEY for Featherless embeddings.")

    embeddings = FeatherlessEmbeddings()

    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db_eval")
    os.makedirs(base_dir, exist_ok=True)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    sweep_pairs: List[Tuple[int, int]] = []
    if args.sweep:
        for pair in args.sweep.split(","):
            pair = pair.strip()
            if not pair:
                continue
            size_str, overlap_str = pair.split(":", 1)
            sweep_pairs.append((int(size_str), int(overlap_str)))
    else:
        sweep_pairs = [(args.chunk_size, args.chunk_overlap)]

    results: List[Dict[str, float]] = []

    for chunk_size, chunk_overlap in sweep_pairs:
        for strategy in strategies:
            persist_dir = os.path.join(base_dir, f"{strategy}_{chunk_size}_{chunk_overlap}")
            if args.reset and os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)

            vector_store = VectorStore(persist_directory=persist_dir)

            if args.reset or not os.path.exists(persist_dir) or not os.listdir(persist_dir):
                docs, total_chunks = build_documents(
                    parsed_docs,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    embeddings=embeddings
                )
                vector_store.add_documents(docs)
            else:
                total_chunks = -1

            stats = evaluate_strategy(
                name=strategy,
                vector_store=vector_store,
                questions=questions,
                k=args.k,
                generate_answers=args.generate_answers
            )
            stats["total_chunks"] = total_chunks
            stats["chunk_size"] = chunk_size
            stats["chunk_overlap"] = chunk_overlap
            results.append(stats)

    print("\nChunking Evaluation Results")
    print("-" * 86)
    header = [
        "Size",
        "Strategy",
        f"Recall@{args.k}",
        "MRR",
        "AnswerMatch",
        "AvgChars",
        "AvgLatency(s)",
        "Chunks"
    ]
    print("{:<9} {:<10} {:>10} {:>8} {:>12} {:>12} {:>12} {:>8}".format(*header))

    for row in results:
        answer_match = "n/a" if row["answer_match"] < 0 else f"{row['answer_match']:.2f}"
        chunks = "n/a" if row["total_chunks"] < 0 else str(row["total_chunks"])
        size_label = f"{row['chunk_size']}/{row['chunk_overlap']}"
        print(
            "{:<9} {:<10} {:>10.2f} {:>8.2f} {:>12} {:>12.0f} {:>12.3f} {:>8}".format(
                size_label,
                row["strategy"],
                row["recall_at_k"],
                row["mrr"],
                answer_match,
                row["avg_retrieved_chars"],
                row["avg_retrieval_latency_s"],
                chunks
            )
        )


if __name__ == "__main__":
    main()
