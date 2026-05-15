from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.storage.vector_store import VectorStore
from src.retrieval.retriever import DocumentRetriever
from src.generation.generator import AnswerGenerator


DEFAULT_TRANSFORMS = ["none", "rewrite", "multi", "step_back", "hyde", "all"]


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def contains_ground_truth(prediction: str, ground_truth: str) -> bool:
    prediction_norm = normalize_text(prediction)
    truth_norm = normalize_text(ground_truth)
    return truth_norm in prediction_norm or prediction_norm in truth_norm


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0
    overlap = len(set(pred_tokens) & set(truth_tokens))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def load_questions(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return [
        {"question": item["question"].strip(), "ground_truth": item["ground_truth"].strip()}
        for item in raw
        if item.get("question") and item.get("ground_truth")
    ]


def docs_to_context_chunks(documents: List[Document]) -> List[Dict[str, Any]]:
    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in documents]


@dataclass
class EvalResult:
    summary: Dict[str, Any]
    details: List[Dict[str, Any]]


def run_eval(
    transform: str,
    retriever: DocumentRetriever,
    generator: AnswerGenerator,
    questions: List[Dict[str, str]],
    top_k: int,
    alpha: float,
) -> EvalResult:
    retrieval_hits = []
    top1_hits = []
    mrrs = []
    truth_ranks = []
    exact_matches = []
    contains_hits = []
    f1_scores = []
    latencies = []
    details: List[Dict[str, Any]] = []

    for row in questions:
        question = row["question"]
        ground_truth = row["ground_truth"]

        start_time = time.perf_counter()
        context_chunks = retriever.retrieve(
            question,
            top_k=top_k,
            transform_mode=transform,
            alpha=alpha,
        )
        retrieval_latency = (time.perf_counter() - start_time) * 1000.0

        docs = [Document(page_content=chunk["text"], metadata=chunk["metadata"]) for chunk in context_chunks]
        truth_rank = _truth_rank(docs, ground_truth)

        generation_start = time.perf_counter()
        answer, sources = generator.generate(question, context_chunks)
        generation_latency = (time.perf_counter() - generation_start) * 1000.0

        total_latency = retrieval_latency + generation_latency

        retrieval_hits.append(1.0 if truth_rank else 0.0)
        top1_hits.append(1.0 if truth_rank == 1 else 0.0)
        mrrs.append((1.0 / truth_rank) if truth_rank else 0.0)
        truth_ranks.append(truth_rank if truth_rank else 0)
        exact_matches.append(1.0 if exact_match(answer, ground_truth) else 0.0)
        contains_hits.append(1.0 if contains_ground_truth(answer, ground_truth) else 0.0)
        f1_scores.append(token_f1(answer, ground_truth))
        latencies.append(total_latency)

        details.append(
            {
                "transform": transform,
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "truth_rank": truth_rank if truth_rank else "",
                "exact_match": exact_match(answer, ground_truth),
                "contains_ground_truth": contains_ground_truth(answer, ground_truth),
                "token_f1": token_f1(answer, ground_truth),
                "retrieval_latency_ms": round(retrieval_latency, 2),
                "generation_latency_ms": round(generation_latency, 2),
                "total_latency_ms": round(total_latency, 2),
                "sources": " | ".join(sources),
            }
        )

    summary = {
        "transform": transform,
        "questions": len(questions),
        "retrieval_recall_at_k": round(statistics.mean(retrieval_hits), 4) if retrieval_hits else 0.0,
        "retrieval_top1_hit_rate": round(statistics.mean(top1_hits), 4) if top1_hits else 0.0,
        "retrieval_mrr": round(statistics.mean(mrrs), 4) if mrrs else 0.0,
        "avg_truth_rank": round(statistics.mean([rank for rank in truth_ranks if rank > 0]), 2) if any(truth_ranks) else 0.0,
        "answer_exact_match": round(statistics.mean(exact_matches), 4) if exact_matches else 0.0,
        "answer_contains_ground_truth": round(statistics.mean(contains_hits), 4) if contains_hits else 0.0,
        "answer_token_f1": round(statistics.mean(f1_scores), 4) if f1_scores else 0.0,
        "avg_total_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
    }
    summary["composite_score"] = round(
        (summary["retrieval_top1_hit_rate"] * 0.45)
        + (summary["retrieval_mrr"] * 0.35)
        + (summary["retrieval_recall_at_k"] * 0.10)
        + (summary["answer_contains_ground_truth"] * 0.05)
        + (summary["answer_token_f1"] * 0.05),
        4,
    )

    return EvalResult(summary=summary, details=details)


def _truth_rank(documents: List[Document], ground_truth: str) -> int | None:
    for index, doc in enumerate(documents, start=1):
        if contains_ground_truth(doc.page_content, ground_truth):
            return index
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate query transformation strategies.")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "arxiv"))
    parser.add_argument("--questions", default=str(REPO_ROOT / "scripts" / "eval_questions.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "scripts" / "query_transform_eval_outputs"))
    parser.add_argument("--transforms", nargs="*", default=DEFAULT_TRANSFORMS, choices=DEFAULT_TRANSFORMS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--limit-questions", type=int, default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--collection-name", default=None)
    parser.add_argument("--reuse-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(Path(args.questions))
    if args.limit_questions is not None:
        questions = questions[: args.limit_questions]

    parser = DocumentParser()
    parsed_data: List[Dict[str, Any]] = []
    if not args.reuse_index:
        parsed_data = load_corpus(Path(args.data_dir), parser, max_files=args.limit_files)
        if not parsed_data:
            raise ValueError(f"No parseable documents found in {args.data_dir}")

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable not set")

    collection_name = args.collection_name or f"QueryTransformEval_{uuid.uuid4().hex[:8]}"
    vector_store = VectorStore(collection_name=collection_name)

    try:
        if not args.reuse_index:
            chunker = DocumentChunker()
            chunks = chunker.chunk(parsed_data)
            vector_store.add_documents(chunks)

        retriever = DocumentRetriever(vector_store)
        generator = AnswerGenerator(model_name=args.model)

        summaries: List[Dict[str, Any]] = []
        details: List[Dict[str, Any]] = []

        for transform in args.transforms:
            print(f"\n=== Evaluating {transform} ===")
            result = run_eval(transform, retriever, generator, questions, args.top_k, args.alpha)
            summaries.append(result.summary)
            details.extend(result.details)
            print(
                f"{transform}: top1={result.summary['retrieval_top1_hit_rate']:.4f}, "
                f"mrr={result.summary['retrieval_mrr']:.4f}, "
                f"recall@k={result.summary['retrieval_recall_at_k']:.4f}"
            )

        summaries.sort(
            key=lambda row: (
                -row["retrieval_top1_hit_rate"],
                -row["retrieval_mrr"],
                -row["retrieval_recall_at_k"],
                row["avg_total_latency_ms"],
            )
        )

        summary_path = output_dir / "query_transform_eval_summary.json"
        detail_path = output_dir / "query_transform_eval_details.csv"

        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2)

        with detail_path.open("w", encoding="utf-8", newline="") as handle:
            if details:
                writer = csv.DictWriter(handle, fieldnames=list(details[0].keys()))
                writer.writeheader()
                writer.writerows(details)

        if summaries:
            print(f"\nBest transform by ranking: {summaries[0]['transform']}")
            print(f"Summary written to: {summary_path}")
            print(f"Question-level details written to: {detail_path}")
    finally:
        if not args.reuse_index:
            vector_store.delete_collection()


def load_corpus(data_dir: Path, parser: DocumentParser, max_files: int | None = None) -> List[Dict[str, Any]]:
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


if __name__ == "__main__":
    main()
