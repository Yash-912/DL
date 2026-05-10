# Multi-modal Enterprise Knowledge Base (RAG)
## Product Requirements Document — v1.0

> **MVP → Production-grade build plan covering all 15 system layers**

| Version | Status | Date | Author |
|---------|--------|------|--------|
| 1.0 | Draft | May 2026 | Engineering |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Phase 1 — MVP](#3-phase-1--mvp)
   - [Point 1 — Document Parsing](#mvp-point-1--document-parsing)
   - [Point 2 — Chunking Strategy](#mvp-point-2--chunking-strategy)
   - [Point 3 — Embedding Model](#mvp-point-3--embedding-model)
   - [Point 4 — Vector Store](#mvp-point-4--vector-store)
   - [Point 5 — Basic Retrieval](#mvp-point-5--basic-retrieval)V
   - [Point 6 — Generation Prompt](#mvp-point-6--generation-prompt)
   - [Point 7 — Basic Metadata](#mvp-point-7--basic-metadata)
   - [Point 8 — Manual Evaluation](#mvp-point-8--manual-evaluation)
   - [MVP Deliverable — API + Interface](#mvp-deliverable--api--interface)
   - [MVP Technology Stack](#mvp-technology-stack)
4. [Phase 2 — Production-grade](#4-phase-2--production-grade)
   - [Point 1 — Full Multi-modal Parsing](#production-point-1--full-multi-modal-document-parsing)
   - [Point 2 — Advanced Chunking](#production-point-2--advanced-chunking-strategies)
   - [Point 3 — Embedding Upgrade](#production-point-3--embedding-model-upgrade)
   - [Point 4 — Weaviate Vector Store](#production-point-4--vector-store-weaviate)
   - [Point 5 — BM25 Sparse Retrieval](#production-point-5--bm25-sparse-retrieval)
   - [Point 6 — Hybrid Retrieval + RRF](#production-point-6--hybrid-retrieval--rrf)
   - [Point 7 — Metadata Enrichment](#production-point-7--metadata-enrichment)
   - [Point 8 — Query Transformation](#production-point-8--query-transformation)
   - [Point 9 — Cross-encoder Reranking](#production-point-9--cross-encoder-reranking)
   - [Point 10 — Generation Prompt Engineering](#production-point-10--generation-prompt-engineering)
   - [Point 11 — RAGAS Evaluation Suite](#production-point-11--ragas-evaluation-suite)
   - [Point 12 — LangFuse Observability](#production-point-12--langfuse-observability)
   - [Point 13 — CI Evaluation Gate](#production-point-13--ci-evaluation-gate)
   - [Point 14 — Index Monitoring](#production-point-14--index-monitoring)
   - [Point 15 — Production Infrastructure](#production-point-15--production-infrastructure)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Risks and Mitigations](#6-risks-and-mitigations)
7. [Success Metrics](#7-success-metrics-summary)
8. [Appendix: Key Technical Concepts](#8-appendix-key-technical-concepts)

---

## 1. Executive Summary

This document defines the product requirements for the **Multi-modal Enterprise Knowledge Base**, a Retrieval-Augmented Generation (RAG) system that makes internal enterprise knowledge queryable in plain language. Documents locked in PDFs, PowerPoints, code repositories, and wikis become instantly accessible through natural language queries with source-cited answers.

The build is structured in two phases: an **MVP** that validates the core pipeline end-to-end, and a **Production** phase that adds hybrid retrieval, advanced query transformation, reranking, evaluation, and full observability. Each phase is independently deployable and valuable.

> **Why this system:** Most companies have critical knowledge trapped in documents. People ask colleagues instead of finding it themselves, spend hours searching, or make decisions with incomplete information. This system eliminates that friction by turning every document into queryable, cited knowledge.

### 1.1 Problem Statement

Enterprise knowledge is fragmented across formats (PDF, PPTX, DOCX, code, Markdown), tools (Confluence, SharePoint, Google Drive, GitHub), and teams. Existing search returns documents, not answers. LLMs without retrieval hallucinate when asked about private or recent information. The gap between knowledge existing in the organization and knowledge being usable is enormous.

### 1.2 Solution

A RAG system that ingests documents in any format, chunks and embeds them intelligently, retrieves the most relevant context for any query, and generates grounded, cited answers. The system is measurable (RAGAS evaluation), observable (LangFuse tracing), and regression-proof (CI eval gates).

---

## 2. System Architecture Overview

The system has four planes that together constitute a complete production RAG architecture. Every feature in this PRD maps to one or more of these planes.

| Plane | Responsibility | Key Components |
|-------|---------------|----------------|
| **Ingestion** | Parse, clean, chunk, enrich, embed documents | Parser, Chunker, Contextual Enricher, Embedder |
| **Storage** | Index vectors and text for retrieval | Vector store, Sparse index, Document store, Metadata DB |
| **Serving** | Transform queries and retrieve context for generation | Query rewriter, Hybrid retriever, Reranker, LLM generator |
| **Observability** | Measure, trace, and protect system quality | RAGAS eval, LangFuse, CI gate, Index monitor |

---

## 3. Phase 1 — MVP

> **Build and validate the core pipeline end-to-end. Prioritize correctness over sophistication.**

The MVP goal is to prove the system works on real documents and returns grounded answers. It deliberately avoids advanced techniques to keep complexity low and feedback loops fast. You should be able to build and demo Phase 1 in **2–4 weeks**.

> **MVP success criteria:** Given 20–30 representative documents from your domain, the system correctly answers 70%+ of test questions with proper source citations, with no crashes or timeouts on documents up to 100 pages.

---

### MVP Point 1 — Document Parsing

Parse raw documents into clean, normalized text. At MVP stage, handle the two or three formats most common in your domain and defer the rest.

**What to build:**
- PDF parsing using PyMuPDF (`fitz`) for text-heavy PDFs and `pdfplumber` for tables
- PowerPoint parsing using `python-pptx` extracting slide text and speaker notes
- Plain text and Markdown as passthrough (no parsing needed)
- Strip headers, footers, and page numbers using heuristics (repeated lines across pages)
- Output: list of `(text, metadata)` tuples per document

**What to defer to production:**
- Scanned PDF / OCR support (Tesseract, AWS Textract)
- Image captioning (GPT-4V)
- Code file parsing (tree-sitter AST)
- HTML and web scraping

**Libraries:**
```bash
pip install pymupdf pdfplumber python-pptx unstructured
```
Use Unstructured.io for format normalization — it wraps PyMuPDF and classifies element types (Title, NarrativeText, Table, Image, ListItem).

---

### MVP Point 2 — Chunking Strategy

Split documents into pieces that can be individually embedded and retrieved. At MVP, use recursive character splitting which respects natural language boundaries without requiring ML at ingestion time.

**What to build:**
- `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=100`
- Separator hierarchy: paragraph break → line break → sentence end → space → character
- Each chunk stores: text, source filename, page number, chunk index, total chunks

**Configuration:**

| Parameter | MVP Default | Notes |
|-----------|-------------|-------|
| `chunk_size` | 1000 tokens | Covers ~3–4 paragraphs. Adjust based on domain |
| `chunk_overlap` | 100 tokens | ~10% of chunk size. Prevents boundary information loss |
| `separators` | `["\n\n", "\n", ". ", " "]` | Recursive fallback order |

**What to defer to production:**
- Semantic chunking (requires sentence-level embedding at ingestion)
- Parent-child hierarchical chunking
- Proposition chunking (LLM call per passage)
- AST-aware code chunking
- RAPTOR tree summarization

---

### MVP Point 3 — Embedding Model

Convert each chunk into a dense vector that captures its semantic meaning. At MVP, use OpenAI's API to avoid infrastructure complexity.

**What to build:**
- Embed all chunks using `text-embedding-3-small` (cost-efficient) or `text-embedding-3-large` (higher quality)
- Batch embedding calls (up to 2048 texts per API call) to minimize latency and cost
- Store embedding vectors alongside chunk metadata
- Cache embeddings by content hash so re-ingesting the same document doesn't re-embed

| Model | Dimensions | Cost | Quality | Recommendation |
|-------|-----------|------|---------|----------------|
| `text-embedding-3-small` | 1536 | $0.02/1M tokens | Good | Use for MVP |
| `text-embedding-3-large` | 3072 | $0.13/1M tokens | Excellent | Use for production |
| `BGE-large-en-v1.5` | 1024 | Free (self-hosted) | Very good | Use if data privacy needed |

---

### MVP Point 4 — Vector Store

Store and search embeddings using approximate nearest neighbor (HNSW) indexing. At MVP, use a simple setup that is easy to run locally.

**What to build:**
- **ChromaDB** for local development — zero infrastructure, in-process, persists to disk
- Schema: `chunk_id`, `text`, `embedding`, `source_file`, `page_number`, `chunk_index`
- Cosine similarity search returning top-k results
- Basic metadata filtering (filter by `source_file` or `doc_type`)

> **MVP storage choice:** Use ChromaDB locally. It requires no server, persists to a local directory, and has a simple Python API. Migrate to Weaviate in Phase 2 when you need hybrid search, multi-tenancy, and production reliability.

**What to defer to production:**
- Weaviate with HNSW configuration (M, ef_construction, ef_search tuning)
- Hybrid search (dense + sparse in one query)
- Multi-tenancy and access control
- Horizontal scaling and sharding

---

### MVP Point 5 — Basic Retrieval

Given a user query, find the most relevant chunks. At MVP, dense-only retrieval is sufficient to validate the system.

**What to build:**
- Embed the user query using the same model used at ingestion
- Run cosine similarity search against the vector store, returning top-5 chunks
- Pass retrieved chunks directly to the LLM — no reranking yet

**What to defer to production:**
- BM25 sparse retrieval
- Hybrid retrieval with RRF fusion
- Query rewriting and HyDE
- Cross-encoder reranking

---

### MVP Point 6 — Generation Prompt

Construct the LLM prompt with retrieved context and enforce grounding rules. This is where RAG actually becomes RAG rather than a regular LLM call.

**What to build:**
- System prompt enforcing: answer only from context, cite sources, say "I don't know" if context is insufficient
- Format context chunks with source labels: `[Source: filename, page X]`
- User message: formatted context block + user question
- Use Claude Sonnet or GPT-4o as the generator
- Return both the generated answer and the list of source chunks used

**System prompt (MVP):**
```
You are a precise question-answering assistant. Answer the user's question 
using ONLY the provided context. Cite your sources using [Source: filename, 
page X] after each claim. If the context is insufficient, say "I cannot 
answer this from the available documents." Do not use outside knowledge.
```

---

### MVP Point 7 — Basic Metadata

Attach essential metadata to every chunk at ingestion time. This enables source citations and basic filtering from day one.

**Required metadata fields at MVP:**
- `source_file` — original filename
- `page_number` — page within the document
- `chunk_index` — position of this chunk within the document
- `doc_type` — `pdf` / `pptx` / `markdown` / `txt`
- `ingested_at` — ISO timestamp of ingestion

---

### MVP Point 8 — Manual Evaluation

Before building the automated eval pipeline, validate correctness manually. Create 20–30 question-answer pairs and check the system by hand.

**What to build:**
- A spreadsheet or simple script that runs a list of test questions through the system
- Output: question, generated answer, source chunks retrieved, expected answer
- Manually rate each answer: correct / partially correct / wrong / hallucinated
- Track: how often the right source chunk was retrieved, how often the answer was faithful

> **Manual eval is not optional.** The purpose of MVP is to find out whether the system works on your data. Manual evaluation is the fastest way to discover fundamental problems: wrong chunking, bad document parsing, retrieval returning irrelevant chunks. Fix these before adding sophistication.

---

### MVP Deliverable — API + Interface

The MVP ships as a simple FastAPI application exposing two endpoints.

**Endpoints:**
- `POST /ingest` — accepts a file upload, parses, chunks, embeds, and stores it
- `POST /query` — accepts a question string, returns answer + sources

**Optional: Streamlit UI:**
- File upload widget that calls `/ingest`
- Chat interface that calls `/query` and displays answer with source citations
- Enough to demo to stakeholders and collect real feedback

---

### MVP Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Document parsing | Unstructured.io, PyMuPDF, python-pptx | Handles most formats, good defaults |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Simple, reliable, respects language boundaries |
| Embedding | OpenAI `text-embedding-3-small` | No infrastructure, fast API, good quality |
| Vector store | ChromaDB (local) | Zero setup, in-process, persists to disk |
| Retrieval | Dense cosine similarity, top-5 | Sufficient to validate the pipeline |
| Generation | Claude Sonnet / GPT-4o | Instruction-following, citation-capable |
| API | FastAPI + Uvicorn | Lightweight, async, well-documented |
| UI (optional) | Streamlit | Fastest path to a demo interface |

---

## 4. Phase 2 — Production-grade

> **Upgrade every layer to industry standards. Add hybrid retrieval, advanced chunking, evaluation, and full observability.**

Phase 2 upgrades the MVP into a system you would be comfortable running at company scale. Each section maps to one or more of the 15 technical layers. They can be implemented incrementally in the order shown — each upgrade independently improves the system.

> **Production success criteria:** RAGAS scores: faithfulness ≥ 0.85, context precision ≥ 0.80, context recall ≥ 0.75, answer relevancy ≥ 0.82. P95 query latency < 3 seconds. CI eval gate blocking regressions on every PR.

---

### Production Point 1 — Full Multi-modal Document Parsing

Extend parsing to handle every document format in your corpus, including images and code.

**Upgrades from MVP:**
- **Scanned PDFs:** integrate Tesseract OCR or AWS Textract for image-based documents
- **Images:** send to GPT-4V or Claude Vision to generate descriptive text captions. Store caption as chunk text with metadata `doc_type: image`
- **Code files:** use tree-sitter for AST-aware parsing. Extract at function and class level so each chunk is a complete syntactic unit
- **HTML / web content:** use BeautifulSoup + trafilatura for content extraction, stripping nav, ads, and boilerplate
- **DOCX:** python-docx for Word documents preserving heading hierarchy

Use Unstructured.io's `hi_res` strategy to classify every element into its type (`Title`, `NarrativeText`, `Table`, `Image`, `ListItem`, `Header`, `Footer`). Strip footers and headers. Pass `Table` elements through a table-to-markdown converter.

| Format | Parser | Special handling |
|--------|--------|-----------------|
| PDF (digital) | PyMuPDF + Unstructured | Element classification, table extraction |
| PDF (scanned) | Tesseract / AWS Textract | OCR before text extraction |
| PPTX | python-pptx | Slide title + body + speaker notes combined |
| Images | GPT-4V / Claude Vision | Caption generation, OCR for text in images |
| Code | tree-sitter | Function/class level, class context prepended to methods |
| HTML | trafilatura | Main content extraction, nav/footer stripped |
| DOCX | python-docx | Heading hierarchy preserved as metadata |

---

### Production Point 2 — Advanced Chunking Strategies

Upgrade from flat recursive splitting to a hierarchical strategy that optimizes separately for retrieval precision and generation context.

#### Primary strategy: parent-child chunking

- **Child chunks:** 200–256 tokens. Embedded and indexed. Precise retrieval units.
- **Parent chunks:** 800–1024 tokens. Stored in document store. Returned to LLM after retrieval.
- Each child stores `parent_id`. After retrieving children, fetch parents and deduplicate.
- Implementation: LangChain `ParentDocumentRetriever` with Redis or a persistent key-value store for parent storage

#### Structure-aware chunking for well-formatted documents

- **Markdown:** `MarkdownHeaderTextSplitter` — split on H1/H2/H3 boundaries, carry header path as metadata
- **HTML documentation:** `HTMLHeaderTextSplitter`
- **PDFs with clear sections:** Unstructured `chunk_by_title` groups content between `Title` elements

#### Semantic chunking for unstructured text

- Embed every sentence using the embedding model
- Compute cosine distance between adjacent sentences
- Split where distance exceeds 95th percentile threshold — these are genuine topic boundaries
- Apply secondary size cap to prevent oversized chunks

#### Proposition chunking for precision-critical corpora

- Use a fast LLM (Claude Haiku, GPT-4o-mini) to decompose each passage into atomic factual statements
- Each proposition becomes its own chunk with its own precise embedding
- Store original passage as parent for context during generation
- Best for: scientific literature, legal documents, technical specifications

#### RAPTOR for synthesis queries

- Build a recursive tree of summaries over your chunk collection
- Cluster leaf-level chunks by semantic similarity using Gaussian Mixture Models
- Summarize each cluster into a parent node. Repeat until single root.
- Index all levels. Specific queries match leaves; broad synthesis queries match high-level summaries.

#### Chunking strategy selection matrix

| Document type | Primary strategy | Fallback |
|---------------|-----------------|----------|
| Structured markdown/HTML docs | Structure-aware (header-based) | Recursive splitter for oversized sections |
| Messy PDFs, reports | Unstructured chunk_by_title + parent-child | Recursive splitter |
| Scientific / legal text | Proposition chunking | Semantic chunking |
| Source code | AST-aware (tree-sitter) | Recursive splitter by newlines |
| Large corpora, book-length | RAPTOR multi-level tree | Parent-child |
| Slides / presentations | Slide-level (one chunk per slide) | Add speaker notes as child |

---

### Production Point 3 — Embedding Model Upgrade

**Upgrades from MVP:**
- Switch to `text-embedding-3-large` (3072 dims) for higher retrieval precision, or `BGE-large-en-v1.5` for self-hosted data privacy
- **Domain-specific fine-tuning:** if your corpus is highly specialized (legal, medical, code), fine-tune a base embedding model on in-domain pairs using `sentence-transformers`
- **Embedding cache:** content-hash keyed cache (Redis) so unchanged documents are never re-embedded on re-ingestion
- **Batch size optimization:** tune batch_size to maximize GPU utilization for self-hosted models
- Separate query vs document encoders if using asymmetric models (E5-instruct requires different prompts for query vs document)

---

### Production Point 4 — Vector Store (Weaviate)

Migrate from ChromaDB to Weaviate for production reliability, hybrid search support, and operational features.

**Weaviate setup:**
- Deploy via Docker Compose (single node) or Weaviate Cloud for managed hosting
- Define `DocumentChunk` class with all metadata properties typed correctly
- Configure HNSW index: `M=16`, `ef_construction=256`, `ef_search=128` as starting point
- Enable BM25 module for hybrid search

**Schema design:**

| Property | Type | Purpose |
|----------|------|---------|
| `text` | text | Chunk content, used for BM25 indexing |
| `chunk_id` | text | UUID for deduplication |
| `parent_id` | text | Link to parent chunk in docstore |
| `source_file` | text | Filename for citation and filtering |
| `page_number` | int | Page for citation |
| `section_title` | text | Section heading for context |
| `doc_type` | text | pdf / pptx / code / image |
| `department` | text | For access control filtering |
| `created_at` | date | For time-based filtering |
| `embedding` | number[] | Dense vector |

---

### Production Point 5 — BM25 Sparse Retrieval

Add a parallel lexical retrieval path. BM25 catches exact matches that dense retrieval misses: product codes, proper nouns, technical identifiers, legal citations.

**What to build:**
- Enable Weaviate's built-in BM25 search (keyword search on the `text` property) — no separate Elasticsearch needed at first
- For scale or advanced features: run Elasticsearch alongside Weaviate, index the same chunks, query both in parallel
- Tokenization: use NLTK or spaCy for query tokenization. Remove stopwords. Apply stemming (PorterStemmer) for better recall.
- Return top-100 candidates from BM25 (same as dense) before fusion

**When BM25 outperforms dense retrieval:**
- Exact product/model numbers: `SKU-XJ-2234`, `CVE-2024-4877`
- Proper names: person names, company names, location names
- Technical identifiers: function names, API endpoints, error codes
- Legal and regulatory references: section numbers, statute codes

---

### Production Point 6 — Hybrid Retrieval + RRF

Combine dense and sparse retrieval results using Reciprocal Rank Fusion. This is the single most impactful upgrade from a naive RAG system.

**Architecture:**
- Run dense retrieval (Weaviate HNSW) and sparse retrieval (BM25) in parallel using `asyncio`
- Each returns top-100 candidates
- Apply RRF to merge: `score(d) = Σ 1 / (60 + rank(d))`
- RRF ignores raw score values — only uses rank positions. No normalization needed.
- Take top-25 from RRF-fused list for reranking

> **Why RRF works:** Dense scores (cosine similarity 0.7–1.0) and BM25 scores (unbounded positive numbers) are incomparable. Adding them directly is meaningless. RRF normalizes via rank position: a document ranked 1st by both retrievers scores `2/(60+1) ≈ 0.033`. A document only one retriever found gets at most `1/(60+1) ≈ 0.016`. Documents appearing in both lists are naturally boosted.

**Implementation note:**
- Weaviate's hybrid search API handles RRF internally if you use their hybrid query endpoint
- For cross-system fusion (Weaviate dense + Elasticsearch sparse): implement RRF manually in Python

---

### Production Point 7 — Metadata Enrichment

Expand metadata from the MVP's basic fields to a full set that enables rich filtering, proper citations, and access control.

**Additional fields for production:**
- `section_title` — the heading of the section this chunk came from
- `department` / `team` — for access control
- `author` — document author extracted from file metadata
- `language` — detected using `langdetect`, for multilingual corpora
- `last_modified` — for time-decay scoring and freshness filtering
- `total_chunks` — total number of chunks from this document
- `content_hash` — SHA-256 of chunk text for deduplication and cache keys

**Contextual enrichment (Anthropic method):**

Before embedding, prepend a 1–2 sentence LLM-generated context summary to each chunk.

- Send: full document (truncated to 8000 tokens) + chunk text to Claude Haiku / GPT-4o-mini
- Prompt: *"In 1–2 sentences, situate this chunk within the overall document for search retrieval purposes."*
- Prepend the generated context to the chunk before embedding
- Store: original chunk text for display, contextualized chunk text for embedding
- Cost: one fast LLM call per chunk at ingestion — done once, never repeated unless document changes
- Impact: Anthropic's experiments showed **49% reduction in retrieval failures**

---

### Production Point 8 — Query Transformation

Transform the user's raw query before retrieval to improve recall. These techniques address the fundamental vocabulary mismatch between how users ask questions and how documents are written.

#### Query rewriting (always on)
- Rephrase the query to be more search-friendly: expand abbreviations, make implicit concepts explicit, remove filler words
- One LLM call per query using a fast model

#### Multi-query expansion (high recall)
- Generate 3 distinct phrasings of the same question, each approaching from a different angle
- Run retrieval for each phrasing independently
- Union all result sets before RRF fusion
- Particularly effective for ambiguous or complex queries

#### HyDE — Hypothetical Document Embeddings (precision)
- Ask the LLM to generate a hypothetical document passage that would answer the query
- Embed the hypothetical passage (not the original query) and use that vector for dense retrieval
- Why it works: a hypothetical answer looks much more like a real document than a question does
- Adds ~500ms latency (one additional LLM call) — acceptable for most applications

#### Step-back prompting (for specific queries)
- For highly specific queries, first generate a more abstract version of the question
- Retrieve for both specific and abstract queries
- The abstract retrieval surfaces relevant background context

#### Query routing (advanced)
- Classify the query type: factual lookup / synthesis / comparison / procedural
- Route to different retrieval strategies per type: factual → BM25-weighted, synthesis → RAPTOR upper levels, procedural → code chunks first

---

### Production Point 9 — Cross-encoder Reranking

After RRF fusion, the top-25 candidates are in a reasonable order but not optimal. A cross-encoder reranker produces the final, accurate ranking before the top-5 go to the LLM.

**Why reranking matters:**

Bi-encoders (your embedding model) encode query and document independently — they cannot model interaction between them. Cross-encoders take the query and document together as input and output a relevance score. This captures nuanced relevance that bi-encoders miss, at the cost of speed. This is why reranking is applied only to the top-25 candidates, not the full index.

**Options:**

| Option | Quality | Latency | Cost | Recommendation |
|--------|---------|---------|------|----------------|
| Cohere Rerank v3 | Excellent | ~200ms | Pay per use | Best for production |
| `BAAI/bge-reranker-large` | Very good | ~300ms (GPU) | Free (self-hosted) | Best for privacy/cost |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Good | ~100ms | Free (self-hosted) | Good for latency-sensitive |

**Implementation:**
- Input: `(query, candidate_chunk_text)` pairs for all 25 candidates
- Output: relevance score per pair (cross-encoder reads them together)
- Take top-5 by reranker score and pass to generation
- Log reranker score deltas — large gaps between score 5 and 6 indicate clean retrieval; small gaps indicate ambiguity

---

### Production Point 10 — Generation Prompt Engineering

Upgrade the MVP generation prompt with faithfulness constraints, structured citations, context window management, and streaming.

**Upgrades from MVP:**
- Structured citation format: every claim must include `[Source: filename, page X]` inline, not just at the end
- Post-process generated answer to verify all cited sources exist in the retrieved chunks — catch hallucinated citations
- Context window tracking: count tokens before sending, truncate or summarize if approaching model limit
- Streaming: stream tokens as they are generated for better perceived latency
- Multi-turn conversation: maintain message history, re-retrieve context on each turn
- Confidence expression: prompt the model to express uncertainty when context is thin or contradictory

**Advanced prompt patterns:**
- **Chain-of-thought** for complex reasoning: ask the model to reason step-by-step before answering
- **Structured output:** for certain query types, ask for JSON output `{summary, key_facts, sources}` for downstream processing
- **Anti-hallucination reinforcement:** include explicit examples of correct behavior (citing sources) and incorrect behavior (adding outside knowledge) in the prompt

---

### Production Point 11 — RAGAS Evaluation Suite

Build a systematic, automated evaluation pipeline. This is what separates a system you built from a system you trust.

**The five RAGAS metrics:**

| Metric | Measures | Target | How computed |
|--------|----------|--------|--------------|
| Faithfulness | Are answer claims supported by context? | ≥ 0.85 | Extract claims, verify each against context via LLM judge |
| Context precision | What fraction of retrieved chunks were relevant? | ≥ 0.80 | LLM grades each retrieved chunk for relevance |
| Context recall | Was all needed information retrieved? | ≥ 0.75 | Compare retrieved content against ground truth answer |
| Answer relevancy | Does the answer address the question? | ≥ 0.82 | Generate reverse questions from answer, compare to original |
| Answer correctness | Is the answer factually correct? | ≥ 0.78 | Compare against labeled ground truth answer |

**Building the golden dataset:**
- **Manual:** domain experts write 100–200 question-answer pairs. Highest quality, most effort.
- **Synthetic:** use RAGAS `TestsetGenerator` to generate questions from your documents automatically. Mix simple (50%), reasoning (30%), multi-context (20%) question types.
- **Hybrid:** generate synthetically, then have experts review and correct. Best practical approach.
- Target: minimum 100 Q&A pairs covering the document types and query patterns you expect in production

**Running evaluation:**
- Run the full golden dataset through the RAG pipeline
- Collect: question, generated answer, retrieved chunks, ground truth answer
- Compute all 5 RAGAS metrics using the `ragas` Python library
- Store results with timestamp, commit hash, and system configuration
- Track metric trends over time to detect gradual degradation

---

### Production Point 12 — LangFuse Observability

Trace every query through the full pipeline. This is how you debug production failures and optimize performance.

**What LangFuse captures:**
- Full query trace: input query → rewritten query → retrieval results → reranked results → generation prompt → output
- Latency breakdown per step: how long did query rewriting take vs retrieval vs reranking vs generation?
- Token usage and cost per query
- Retrieved chunk IDs and their relevance scores at each stage
- User feedback if collected (thumbs up/down on answers)

**Integration:**
- Wrap each pipeline step with `@observe()` decorator
- Tag traces with `user_id`, `session_id`, `doc_type` filter, and any A/B test variant
- Set up dashboards: P50/P95/P99 latency, cost per query per day, error rate
- Alert on: latency spike (P95 > 5s), error rate > 1%, cost anomaly

**Debugging workflow:**
1. User reports bad answer → find trace in LangFuse by session ID
2. Inspect: what chunks were retrieved? Were they relevant? Did the reranker score them correctly?
3. Trace back: was it a parsing failure? A chunking boundary issue? A retrieval gap?
4. Fix the root cause, add a test case to the golden dataset, verify the fix

---

### Production Point 13 — CI Evaluation Gate

Automate quality enforcement. Every pull request runs the evaluation suite and is blocked from merging if any metric falls below its threshold.

**CI pipeline steps:**
1. Checkout code and install dependencies
2. Run the golden dataset through the updated RAG pipeline
3. Compute RAGAS metrics
4. Compare against threshold table
5. **Pass:** all metrics above threshold → merge allowed
6. **Fail:** any metric below threshold → block merge, post metric delta as PR comment

**Threshold table:**

| Metric | Block threshold | Warning threshold | Target |
|--------|----------------|-------------------|--------|
| Faithfulness | < 0.82 | < 0.85 | ≥ 0.88 |
| Context precision | < 0.77 | < 0.80 | ≥ 0.83 |
| Context recall | < 0.72 | < 0.75 | ≥ 0.78 |
| Answer relevancy | < 0.79 | < 0.82 | ≥ 0.85 |
| P95 latency | > 5s | > 3s | < 2s |

**Running a subset for fast feedback:**
- **Full golden dataset:** run nightly or on release branches (10–30 minutes for 200 questions)
- **Critical subset (20–30 questions):** run on every PR for fast feedback within 5 minutes

---

### Production Point 14 — Index Monitoring

Monitor the index for staleness, drift, and coverage gaps. A production RAG system is only as good as its index.

**Freshness monitoring:**
- Track `last_modified` of source documents vs ingestion timestamp
- Alert when source documents are modified but not re-ingested within 24 hours
- Build a document registry tracking: filename, source hash, last ingested, chunk count
- Nightly job: scan source directories, compare hashes, queue changed documents for re-ingestion

**Coverage monitoring:**
- Track which file types and sources are indexed vs exist in the source system
- Alert on parsing failures (documents that could not be processed)
- Monitor chunk count per document — a sudden drop suggests a parsing regression

**Embedding drift:**
- When you upgrade embedding models, old and new vectors are in different spaces — incompatible
- Track embedding model version as metadata on every chunk
- On model upgrade: schedule a full re-ingestion pass. Use blue-green index strategy: build new index in parallel, swap when complete.

---

### Production Point 15 — Production Infrastructure

Operational infrastructure for running the system reliably at scale.

**Ingestion pipeline:**
- Async job queue (Celery + Redis or AWS SQS) for background document processing
- Document status tracking: `pending` / `processing` / `indexed` / `failed`
- Retry logic with exponential backoff for failed ingestion jobs
- Rate limiting on LLM calls during contextual enrichment to avoid API throttling

**Serving layer:**
- Async FastAPI with connection pooling to Weaviate
- Request-level timeout (15 seconds max) with graceful degradation — if reranking times out, return RRF results
- Response caching (Redis) for identical queries within a time window
- Rate limiting per user/tenant

**Production technology stack:**

| Layer | Technology | Notes |
|-------|-----------|-------|
| Document parsing | Unstructured.io hi_res + tree-sitter | Multi-format, element classification, AST for code |
| Chunking | Parent-child + semantic + structure-aware | Strategy selected per document type |
| Embedding | `text-embedding-3-large` or `BGE-large` | Content-hash cache in Redis |
| Vector store | Weaviate (HNSW) | Hybrid search enabled, HNSW tuned |
| Sparse index | Weaviate BM25 or Elasticsearch | Parallel retrieval with dense |
| Query transformation | GPT-4o-mini / Claude Haiku | Rewriting + HyDE + multi-query |
| Reranking | Cohere Rerank v3 or `BGE-reranker-large` | Top-25 → top-5 |
| Generation | Claude Sonnet / GPT-4o | Streaming, structured citations |
| Evaluation | RAGAS + golden dataset | Automated on every PR |
| Tracing | LangFuse | Full pipeline traces, cost tracking |
| Job queue | Celery + Redis | Background ingestion |
| API | FastAPI + async | With rate limiting and caching |
| Monitoring | Prometheus + Grafana or Datadog | Latency, cost, error rate dashboards |

---

## 5. Implementation Roadmap

### Phase 1 — MVP (Weeks 1–4)

| Week | Focus | Deliverable |
|------|-------|-------------|
| Week 1 | Parsing + chunking + embedding pipeline | Can ingest PDFs and Markdown, chunks stored in ChromaDB |
| Week 2 | Dense retrieval + generation prompt | Can answer questions from indexed documents with citations |
| Week 3 | FastAPI endpoints + Streamlit UI | Demostable system with file upload and chat interface |
| Week 4 | Manual evaluation + bug fixes | 20–30 test questions evaluated, major issues fixed |

### Phase 2 — Production Upgrades (Weeks 5–16)

| Week(s) | Upgrade | Impact |
|---------|---------|--------|
| 5–6 | Migrate to Weaviate + add BM25 sparse retrieval + RRF fusion | Highest single retrieval quality improvement |
| 7 | Add contextual enrichment (Anthropic method) | Significant recall improvement with modest ingestion cost |
| 8 | Add query rewriting + HyDE | Better handling of ambiguous and specific queries |
| 9 | Add cross-encoder reranking (Cohere or BGE) | Higher precision in final context selection |
| 10–11 | Upgrade to parent-child chunking + structure-aware for docs | Better context for generation, cleaner chunk boundaries |
| 12 | Build golden dataset + RAGAS eval pipeline | First objective quality measurement |
| 13 | LangFuse tracing integration | Full observability, latency breakdown, cost tracking |
| 14 | CI eval gate + threshold enforcement | Automated regression prevention |
| 15 | Index monitoring + freshness pipeline | Production reliability |
| 16 | Load testing + latency optimization + documentation | Production readiness sign-off |

---

## 6. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Document parsing failures on unusual PDF layouts | High | Medium | Log all parse failures. Build a failure queue for manual review. Use `hi_res` strategy for problem documents. |
| Retrieval quality poor for domain-specific terminology | Medium | High | Add BM25 early (Week 5). Consider domain-specific embedding fine-tuning if BM25 doesn't solve it. |
| LLM hallucination even with good context | Medium | High | Enforce strict faithfulness prompt. Post-process to verify citations. Track faithfulness via RAGAS. |
| Embedding model upgrade breaks existing index | Low | High | Track embedding model version per chunk. Blue-green re-index on upgrade. |
| Context window exceeded for long documents | Medium | Medium | Token-count context before sending. Truncate or summarize if approaching limit. Log truncation events. |
| Ingestion pipeline too slow for large document loads | Medium | Low | Async job queue from Week 1. Rate limit LLM calls during enrichment. Batch embedding API calls. |

---

## 7. Success Metrics Summary

### MVP success criteria
- System ingests PDF and Markdown documents without errors
- System returns answers with source citations for 70%+ of test questions
- Zero crashes or timeouts on documents up to 100 pages
- Manual evaluation of 20–30 questions completed and findings documented

### Production success criteria

| Metric | Target | Measurement method |
|--------|--------|-------------------|
| Faithfulness | ≥ 0.85 | RAGAS on golden dataset, automated in CI |
| Context precision | ≥ 0.80 | RAGAS on golden dataset |
| Context recall | ≥ 0.75 | RAGAS on golden dataset |
| Answer relevancy | ≥ 0.82 | RAGAS on golden dataset |
| P95 query latency | < 3 seconds | LangFuse latency tracking |
| Index freshness | < 24hr lag | Index monitor freshness alerts |
| CI eval gate | Zero regressions merged | Automated PR check |
| Parse success rate | ≥ 95% | Ingestion failure tracking |

---

## 8. Appendix: Key Technical Concepts

**Embeddings and vector search**
An embedding model converts text into a fixed-length vector of floats. Semantically similar texts produce vectors that are close in cosine similarity. HNSW (Hierarchical Navigable Small World) indexes these vectors for approximate nearest neighbor search in O(log n) time, making search over millions of vectors feasible for interactive queries.

**BM25**
Best Match 25. A classical information retrieval algorithm that scores documents based on term frequency (how often query terms appear in a chunk) and inverse document frequency (how rare those terms are across the corpus). Excels at exact lexical matching where dense retrieval fails.

**RRF — Reciprocal Rank Fusion**
`score(d) = Σ_r 1/(k + rank_r(d))`. A rank-based score fusion method that combines results from multiple retrievers without requiring score normalization. Documents appearing high in multiple retriever result lists are boosted. `k=60` is the standard constant.

**Cross-encoder reranker**
A model that takes `(query, document)` as a joint input and outputs a relevance score. Unlike bi-encoders (embedding models) that encode query and document independently, cross-encoders model query-document interaction directly, producing much more accurate relevance scores at the cost of speed.

**RAGAS**
Retrieval Augmented Generation Assessment. An evaluation framework measuring faithfulness, context precision, context recall, answer relevancy, and answer correctness. Requires a golden dataset of labeled question-answer pairs.

**HyDE — Hypothetical Document Embeddings**
Generate a hypothetical document that would answer the query, then embed that hypothetical document for retrieval instead of the raw query. Works because a hypothetical answer is geometrically closer to real answer documents in embedding space than a question is.

**Parent-child chunking**
Store documents at two granularities: small child chunks (200 tokens) for precise embedding and retrieval, large parent chunks (1000 tokens) for context-rich generation. Retrieve with children, generate with parents. Balances retrieval precision with generation context.

**Contextual enrichment**
Before embedding each chunk, prepend a short LLM-generated summary that situates the chunk within its source document. Proposed by Anthropic. Reduces retrieval failures by ~49% by giving the embedding model document-level context for each chunk.

**RAPTOR**
Recursive Abstractive Processing for Tree-Organized Retrieval. Builds a multi-level summarization tree over the chunk collection. Enables retrieval at multiple granularities — from specific facts (leaf level) to broad synthesis (root level) — from a single index.

---

*End of Document — Multi-modal Enterprise Knowledge Base PRD v1.0*
*Confidential*
