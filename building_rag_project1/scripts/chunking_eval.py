from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import statistics
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.generation.generator import AnswerGenerator
from src.ingestion.parser import DocumentParser
from src.storage.vector_store import VectorStore
from src.embeddings.sentence_transformer import SentenceTransformerEmbeddings


DEFAULT_STRATEGIES = ["fixed_size", "recursive", "sentence", "semantic", "parent_child"]


@dataclass
class StrategyBundle:
    index_docs: List[Document]
    generation_docs: List[Document]
    parent_lookup: Dict[str, Document]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [token for token in normalize_text(text).split(" ") if token]


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def contains_ground_truth(prediction: str, ground_truth: str) -> bool:
    prediction_norm = normalize_text(prediction)
    truth_norm = normalize_text(ground_truth)
    return truth_norm in prediction_norm or prediction_norm in truth_norm


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)
    if not pred_tokens or not truth_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    truth_counts = Counter(truth_tokens)
    overlap = sum(min(pred_counts[token], truth_counts[token]) for token in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def sentence_split(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def attach_metadata(chunks: List[Document], strategy: str) -> List[Document]:
    ingested_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for index, doc in enumerate(chunks):
        doc.metadata["chunk_index"] = index
        doc.metadata["strategy"] = strategy
        doc.metadata["ingested_at"] = ingested_at
    return chunks


def make_documents(parsed_data: List[Dict[str, Any]], strategy: str) -> List[Document]:
    documents = []
    for item in parsed_data:
        documents.append(Document(page_content=item["text"], metadata=dict(item["metadata"])))
    return attach_metadata(documents, strategy)


def build_fixed_size(parsed_data: List[Dict[str, Any]], chunk_size: int, overlap: int) -> StrategyBundle:
    index_docs: List[Document] = []
    step = max(1, chunk_size - overlap)

    for item in parsed_data:
        text = item["text"]
        metadata = dict(item["metadata"])
        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size].strip()
            if chunk_text:
                index_docs.append(Document(page_content=chunk_text, metadata=dict(metadata)))

    index_docs = attach_metadata(index_docs, "fixed_size")
    return StrategyBundle(index_docs=index_docs, generation_docs=index_docs, parent_lookup={})


def build_recursive(parsed_data: List[Dict[str, Any]], chunk_size: int, overlap: int) -> StrategyBundle:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    docs = make_documents(parsed_data, "recursive_input")
    index_docs = splitter.split_documents(docs)
    index_docs = attach_metadata(index_docs, "recursive")
    return StrategyBundle(index_docs=index_docs, generation_docs=index_docs, parent_lookup={})


def build_sentence(parsed_data: List[Dict[str, Any]], max_chunk_size: int) -> StrategyBundle:
    index_docs: List[Document] = []
    for item in parsed_data:
        sentences = sentence_split(item["text"])
        if not sentences:
            continue

        current_sentences: List[str] = []
        current_length = 0
        for sentence in sentences:
            if current_sentences and current_length + len(sentence) > max_chunk_size:
                chunk_text = " ".join(current_sentences).strip()
                index_docs.append(Document(page_content=chunk_text, metadata=dict(item["metadata"])))
                current_sentences = []
                current_length = 0

            current_sentences.append(sentence)
            current_length += len(sentence)

        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            index_docs.append(Document(page_content=chunk_text, metadata=dict(item["metadata"])))

    index_docs = attach_metadata(index_docs, "sentence")
    return StrategyBundle(index_docs=index_docs, generation_docs=index_docs, parent_lookup={})


def _cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_semantic(parsed_data: List[Dict[str, Any]], embeddings: SentenceTransformerEmbeddings, similarity_threshold: float) -> StrategyBundle:
    index_docs: List[Document] = []

    for item in parsed_data:
        sentences = sentence_split(item["text"])
        if len(sentences) <= 1:
            index_docs.append(Document(page_content=item["text"].strip(), metadata=dict(item["metadata"])))
            continue

        sentence_vectors = embeddings.embed_documents(sentences)
        current_group = [sentences[0]]

        for idx in range(1, len(sentences)):
            similarity = _cosine_similarity(sentence_vectors[idx - 1], sentence_vectors[idx])
            if similarity < similarity_threshold:
                chunk_text = " ".join(current_group).strip()
                if chunk_text:
                    index_docs.append(Document(page_content=chunk_text, metadata=dict(item["metadata"])))
                current_group = [sentences[idx]]
            else:
                current_group.append(sentences[idx])

        chunk_text = " ".join(current_group).strip()
        if chunk_text:
            index_docs.append(Document(page_content=chunk_text, metadata=dict(item["metadata"])))

    index_docs = attach_metadata(index_docs, "semantic")
    return StrategyBundle(index_docs=index_docs, generation_docs=index_docs, parent_lookup={})


def build_parent_child(
    parsed_data: List[Dict[str, Any]],
    parent_chunk_size: int,
    parent_overlap: int,
    child_chunk_size: int,
    child_overlap: int,
) -> StrategyBundle:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    parent_docs: List[Document] = []
    for item in parsed_data:
        parent_docs.append(Document(page_content=item["text"], metadata=dict(item["metadata"])))

    parent_docs = parent_splitter.split_documents(parent_docs)

    parent_lookup: Dict[str, Document] = {}
    parent_ready_docs: List[Document] = []
    for index, parent_doc in enumerate(parent_docs):
        parent_id = f"parent_{index}"
        parent_doc.metadata["parent_id"] = parent_id
        parent_doc.metadata["chunk_role"] = "parent"
        parent_ready_docs.append(parent_doc)
        parent_lookup[parent_id] = parent_doc

    child_docs: List[Document] = []
    for parent_doc in parent_ready_docs:
        parent_id = parent_doc.metadata["parent_id"]
        for child_doc in child_splitter.split_documents([parent_doc]):
            child_doc.metadata["parent_id"] = parent_id
            child_doc.metadata["chunk_role"] = "child"
            child_docs.append(child_doc)

    child_docs = attach_metadata(child_docs, "parent_child")
    for parent_doc in parent_ready_docs:
        parent_doc.metadata["strategy"] = "parent_child"

    return StrategyBundle(index_docs=child_docs, generation_docs=parent_ready_docs, parent_lookup=parent_lookup)


def load_questions(questions_path: Path) -> List[Dict[str, str]]:
    with questions_path.open("r", encoding="utf-8") as handle:
        raw_questions = json.load(handle)

    questions: List[Dict[str, str]] = []
    for item in raw_questions:
        question = item.get("question", "").strip()
        ground_truth = item.get("ground_truth", "").strip()
        if question and ground_truth:
            questions.append({"question": question, "ground_truth": ground_truth})
    return questions


def load_corpus(data_dir: Path, parser: DocumentParser, max_files: Optional[int] = None) -> List[Dict[str, Any]]:
    supported_files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".pptx", ".txt", ".md"}
    )

    if max_files is not None:
        supported_files = supported_files[:max_files]

    parsed_data: List[Dict[str, Any]] = []
    for file_path in supported_files:
        parsed_data.extend(parser.parse(str(file_path)))

    return parsed_data


def docs_to_context_chunks(documents: List[Document]) -> List[Dict[str, Any]]:
    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in documents]


def retrieval_contains_truth(documents: List[Document], ground_truth: str) -> bool:
    return any(contains_ground_truth(doc.page_content, ground_truth) for doc in documents)


def retrieval_truth_rank(documents: List[Document], ground_truth: str) -> Optional[int]:
    for index, doc in enumerate(documents, start=1):
        if contains_ground_truth(doc.page_content, ground_truth):
            return index
    return None


def expand_parent_documents(documents: List[Document], parent_lookup: Dict[str, Document]) -> List[Document]:
    if not parent_lookup:
        return documents

    expanded: List[Document] = []
    seen_parent_ids = set()
    for doc in documents:
        parent_id = doc.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parent_ids:
            continue
        parent_doc = parent_lookup.get(parent_id)
        if parent_doc is not None:
            expanded.append(parent_doc)
            seen_parent_ids.add(parent_id)
    return expanded


def build_strategy(
    strategy: str,
    parsed_data: List[Dict[str, Any]],
    embeddings: SentenceTransformerEmbeddings,
    args: argparse.Namespace,
) -> StrategyBundle:
    if strategy == "fixed_size":
        return build_fixed_size(parsed_data, args.fixed_chunk_size, args.fixed_overlap)
    if strategy == "recursive":
        return build_recursive(parsed_data, args.recursive_chunk_size, args.recursive_overlap)
    if strategy == "sentence":
        return build_sentence(parsed_data, args.sentence_chunk_size)
    if strategy == "semantic":
        return build_semantic(parsed_data, embeddings, args.semantic_threshold)
    if strategy == "parent_child":
        return build_parent_child(
            parsed_data,
            args.parent_chunk_size,
            args.parent_overlap,
            args.child_chunk_size,
            args.child_overlap,
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def evaluate_strategy(
    strategy: str,
    parsed_data: List[Dict[str, Any]],
    questions: List[Dict[str, str]],
    embeddings: SentenceTransformerEmbeddings,
    generator: AnswerGenerator,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    bundle = build_strategy(strategy, parsed_data, embeddings, args)

    with tempfile.TemporaryDirectory(prefix=f"chunk_eval_{strategy}_", ignore_cleanup_errors=True) as tmp_dir:
        vector_store = VectorStore(persist_directory=tmp_dir)
        try:
            vector_store.add_documents(bundle.index_docs)

            question_rows: List[Dict[str, Any]] = []
            retrieval_hits = []
            top1_hits = []
            reciprocal_ranks = []
            exact_matches = []
            containment_hits = []
            context_hits = []
            f1_scores = []
            latencies_ms = []
            context_sizes = []
            truth_ranks = []

            for question_row in questions:
                question = question_row["question"]
                ground_truth = question_row["ground_truth"]

                start_time = time.perf_counter()
                retrieved_docs = vector_store.similarity_search(question, k=args.top_k)
                retrieval_latency_ms = (time.perf_counter() - start_time) * 1000.0

                generation_docs = expand_parent_documents(retrieved_docs, bundle.parent_lookup)
                if not generation_docs:
                    generation_docs = retrieved_docs

                retrieval_hit = retrieval_contains_truth(retrieved_docs, ground_truth)
                truth_rank = retrieval_truth_rank(retrieved_docs, ground_truth)
                context_hit = retrieval_contains_truth(generation_docs, ground_truth)
                context_chunks = docs_to_context_chunks(generation_docs)

                generation_start = time.perf_counter()
                answer, sources = generator.generate(question, context_chunks)
                generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0

                total_latency_ms = retrieval_latency_ms + generation_latency_ms
                latencies_ms.append(total_latency_ms)
                context_sizes.append(len(generation_docs))
                retrieval_hits.append(1.0 if retrieval_hit else 0.0)
                top1_hits.append(1.0 if truth_rank == 1 else 0.0)
                reciprocal_ranks.append((1.0 / truth_rank) if truth_rank else 0.0)
                truth_ranks.append(truth_rank if truth_rank is not None else 0)
                context_hits.append(1.0 if context_hit else 0.0)
                exact_matches.append(1.0 if exact_match(answer, ground_truth) else 0.0)
                containment_hits.append(1.0 if contains_ground_truth(answer, ground_truth) else 0.0)
                f1_scores.append(token_f1(answer, ground_truth))

                question_rows.append(
                    {
                        "strategy": strategy,
                        "question": question,
                        "ground_truth": ground_truth,
                        "answer": answer,
                        "retrieval_hit": retrieval_hit,
                        "truth_rank": truth_rank if truth_rank is not None else "",
                        "context_hit": context_hit,
                        "exact_match": exact_match(answer, ground_truth),
                        "contains_ground_truth": contains_ground_truth(answer, ground_truth),
                        "token_f1": token_f1(answer, ground_truth),
                        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                        "generation_latency_ms": round(generation_latency_ms, 2),
                        "total_latency_ms": round(total_latency_ms, 2),
                        "retrieved_chunks": len(generation_docs),
                        "sources": " | ".join(sources),
                    }
                )

            summary = {
                "strategy": strategy,
                "questions": len(questions),
                "retrieval_recall_at_k": round(statistics.mean(retrieval_hits), 4) if retrieval_hits else 0.0,
                "retrieval_top1_hit_rate": round(statistics.mean(top1_hits), 4) if top1_hits else 0.0,
                "retrieval_mrr": round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
                "avg_truth_rank": round(statistics.mean([rank for rank in truth_ranks if rank > 0]), 2) if any(truth_ranks) else 0.0,
                "context_hit_rate": round(statistics.mean(context_hits), 4) if context_hits else 0.0,
                "answer_exact_match": round(statistics.mean(exact_matches), 4) if exact_matches else 0.0,
                "answer_contains_ground_truth": round(statistics.mean(containment_hits), 4) if containment_hits else 0.0,
                "answer_token_f1": round(statistics.mean(f1_scores), 4) if f1_scores else 0.0,
                "avg_total_latency_ms": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
                "avg_retrieved_context_chunks": round(statistics.mean(context_sizes), 2) if context_sizes else 0.0,
                "index_chunks": len(bundle.index_docs),
                "generation_chunks": len(bundle.generation_docs),
            }
            summary["composite_score"] = round(
                (summary["retrieval_top1_hit_rate"] * 0.45)
                + (summary["retrieval_mrr"] * 0.35)
                + (summary["retrieval_recall_at_k"] * 0.10)
                + (summary["answer_contains_ground_truth"] * 0.05)
                + (summary["answer_token_f1"] * 0.05),
                4,
            )

            return summary, question_rows
        finally:
            del vector_store
            gc.collect()


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "strategy",
        "top1",
        "mrr",
        "recall@k",
        "exact",
        "contains",
        "f1",
        "latency_ms",
        "chunks",
        "score",
    ]
    print("\nSummary")
    print("-" * 100)
    print(f"{headers[0]:<16} {headers[1]:>8} {headers[2]:>8} {headers[3]:>8} {headers[4]:>10} {headers[5]:>8} {headers[6]:>12} {headers[7]:>8} {headers[8]:>8}")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['strategy']:<16} "
            f"{row['retrieval_top1_hit_rate']:>8.4f} "
            f"{row['retrieval_mrr']:>8.4f} "
            f"{row['retrieval_recall_at_k']:>8.4f} "
            f"{row['answer_exact_match']:>8.4f} "
            f"{row['answer_contains_ground_truth']:>10.4f} "
            f"{row['answer_token_f1']:>8.4f} "
            f"{row['avg_total_latency_ms']:>12.2f} "
            f"{row['index_chunks']:>8} "
            f"{row['composite_score']:>8.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare chunking strategies over the arxiv corpus.")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "arxiv"), help="Corpus directory")
    parser.add_argument("--questions", default=str(REPO_ROOT / "scripts" / "eval_questions.json"), help="Question bank path")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "scripts" / "chunking_eval_outputs"), help="Directory for CSV/JSON outputs")
    parser.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES, choices=DEFAULT_STRATEGIES, help="Chunking strategies to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Retriever top-k")
    parser.add_argument("--limit-questions", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--limit-files", type=int, default=None, help="Evaluate only the first N corpus files")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model used for answers")
    parser.add_argument("--fixed-chunk-size", type=int, default=500)
    parser.add_argument("--fixed-overlap", type=int, default=50)
    parser.add_argument("--recursive-chunk-size", type=int, default=1000)
    parser.add_argument("--recursive-overlap", type=int, default=100)
    parser.add_argument("--sentence-chunk-size", type=int, default=500)
    parser.add_argument("--semantic-threshold", type=float, default=0.75)
    parser.add_argument("--parent-chunk-size", type=int, default=1200)
    parser.add_argument("--parent-overlap", type=int, default=150)
    parser.add_argument("--child-chunk-size", type=int, default=350)
    parser.add_argument("--child-overlap", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(Path(args.questions))
    if args.limit_questions is not None:
        questions = questions[: args.limit_questions]

    parser = DocumentParser()
    parsed_data = load_corpus(Path(args.data_dir), parser, max_files=args.limit_files)
    if not parsed_data:
        raise ValueError(f"No parseable documents found in {args.data_dir}")

    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    embeddings = SentenceTransformerEmbeddings()
    generator = AnswerGenerator(model_name=args.model)

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for strategy in args.strategies:
        print(f"\n=== Evaluating {strategy} ===")
        summary, question_rows = evaluate_strategy(strategy, parsed_data, questions, embeddings, generator, args)
        summary_rows.append(summary)
        detail_rows.extend(question_rows)
        print(
            f"{strategy}: recall@k={summary['retrieval_recall_at_k']:.4f}, "
            f"contains={summary['answer_contains_ground_truth']:.4f}, "
            f"f1={summary['answer_token_f1']:.4f}"
        )

    summary_rows.sort(
        key=lambda row: (
            -row["retrieval_recall_at_k"],
            -row["answer_contains_ground_truth"],
            -row["answer_token_f1"],
            row["avg_total_latency_ms"],
        )
    )

    print_summary_table(summary_rows)

    summary_path = output_dir / "chunking_eval_summary.json"
    detail_path = output_dir / "chunking_eval_details.csv"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        if detail_rows:
            writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    if summary_rows:
        best = summary_rows[0]["strategy"]
        print(f"\nBest strategy by ranking: {best}")
        print(f"Summary written to: {summary_path}")
        print(f"Question-level details written to: {detail_path}")


if __name__ == "__main__":
    main()
