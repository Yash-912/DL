import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Load env before importing our modules
load_dotenv()

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.storage.vector_store import VectorStore
from src.retrieval.retriever import DocumentRetriever
from src.generation.generator import AnswerGenerator

app = FastAPI(title="Multi-modal RAG API", version="1.0")

# Initialize pipeline components
parser = DocumentParser()
chunker = DocumentChunker()

vector_store = VectorStore()

retriever = DocumentRetriever(vector_store)
generator = AnswerGenerator()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    transform_mode: str = "none"
    alpha: float = 0.5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a document: Parse -> Chunk -> Embed -> Store
    """
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Parse
        parsed_data = parser.parse(temp_file_path)
        if not parsed_data:
            raise HTTPException(status_code=400, detail="Failed to parse document or document is empty.")
            
        # Chunk
        chunks = chunker.chunk(parsed_data)
        
        # Store
        vector_store.add_documents(chunks)
        
        return {"message": f"Successfully ingested {file.filename}", "chunks_indexed": len(chunks)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base: Retrieve -> Generate
    """
    try:
        # Retrieve context
        context_chunks = retriever.retrieve(
            request.query,
            top_k=request.top_k,
            transform_mode=request.transform_mode,
            alpha=request.alpha,
        )
        
        # Generate answer
        answer, sources = generator.generate(request.query, context_chunks)
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
