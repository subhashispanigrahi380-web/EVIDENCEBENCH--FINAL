"""
FastAPI Server for EvidenceBench Research Workspace.
Provides REST endpoints for querying, ingestion, document lineage, benchmark execution, and trace inspection.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from evidencebench.config import config
from evidencebench.pipeline import EvidenceBenchPipeline
from evidencebench.eval.runner import BenchmarkRunner
from evidencebench.models import (
    GenerationResult,
    ExecutionTrace,
    BenchmarkRunMetrics,
)

app = FastAPI(
    title="EvidenceBench Research Workspace",
    version="1.0.0",
    description="High-precision verifiable RAG with passage citations, hybrid fusion, reranking, and abstention gatekeeping.",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pipeline Singleton
pipeline = EvidenceBenchPipeline(
    storage_dir=config.storage_dir,
    chunking_strategy="structure_aware",
    index_version="v1.0",
)

# Ingest corpus on startup if store is empty
if len(pipeline.doc_store.documents) == 0:
    pipeline.doc_store.ingest_directory(config.corpus_dir)

if len(pipeline.indexed_chunks) == 0:
    pipeline.reindex()

# Recent traces cache
trace_store: Dict[str, ExecutionTrace] = {}


class QueryRequest(BaseModel):
    query: str
    pipeline_mode: str = Field(default="hybrid_rerank")  # dense, bm25, hybrid, hybrid_rerank
    chunking_strategy: str = Field(default="structure_aware")  # structure_aware, fixed_recursive
    top_k: int = Field(default=5)


class QueryResponse(BaseModel):
    query: str
    answer: str
    decision: Dict[str, Any]
    citations: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    trace_id: str
    total_latency_ms: float


class BenchmarkRequest(BaseModel):
    pipeline_mode: str = "hybrid_rerank"
    chunking_strategy: str = "structure_aware"


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "index_version": pipeline.index_version,
        "total_documents": len(pipeline.doc_store.documents),
        "total_chunks": len(pipeline.indexed_chunks),
        "chunking_strategy": pipeline.chunking_strategy_name,
        "storage_dir": str(pipeline.storage_dir),
    }


@app.post("/api/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    if req.chunking_strategy != pipeline.chunking_strategy_name:
        pipeline.set_chunking_strategy(req.chunking_strategy)

    gen_res, trace = pipeline.query(
        query_text=req.query,
        pipeline_mode=req.pipeline_mode,
        top_k=req.top_k,
    )

    trace_store[trace.trace_id] = trace

    # Build candidates representation
    candidates_data = []
    for c in gen_res.top_chunks:
        candidates_data.append({
            "chunk_id": c.chunk.metadata.chunk_id,
            "doc_filename": c.chunk.metadata.doc_filename,
            "page_number": c.chunk.metadata.page_number,
            "section_title": c.chunk.metadata.section_title,
            "rank": c.rank,
            "score": round(c.score, 4),
            "method_scores": c.method_scores,
            "method_ranks": c.method_ranks,
            "text": c.chunk.text,
            "context_prefix": c.chunk.context_prefix,
        })

    citations_data = [cit.model_dump() for cit in gen_res.citations]

    return QueryResponse(
        query=req.query,
        answer=gen_res.answer,
        decision=gen_res.decision.model_dump(),
        citations=citations_data,
        candidates=candidates_data,
        trace_id=trace.trace_id,
        total_latency_ms=trace.total_latency_ms,
    )


@app.get("/api/trace/{trace_id}")
def get_trace(trace_id: str):
    if trace_id not in trace_store:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace_store[trace_id].model_dump()


@app.get("/api/documents")
def list_documents():
    docs = pipeline.doc_store.list_documents(active_only=False)
    return {
        "documents": [d.model_dump() for d in docs],
        "index_version": pipeline.index_version,
        "total_active_chunks": len(pipeline.indexed_chunks),
    }


@app.post("/api/reindex")
def trigger_reindex(chunking_strategy: Optional[str] = None):
    if chunking_strategy:
        pipeline.set_chunking_strategy(chunking_strategy)
    else:
        pipeline.reindex()
    return {
        "message": "Reindexed successfully",
        "total_chunks": len(pipeline.indexed_chunks),
        "chunking_strategy": pipeline.chunking_strategy_name,
    }


@app.post("/api/benchmark/run")
def run_benchmark(req: BenchmarkRequest):
    if req.chunking_strategy != pipeline.chunking_strategy_name:
        pipeline.set_chunking_strategy(req.chunking_strategy)

    runner = BenchmarkRunner(pipeline=pipeline, dataset_path=config.benchmark_file)
    metrics, detailed, failures = runner.run_evaluation(pipeline_mode=req.pipeline_mode)

    return {
        "metrics": metrics.model_dump(),
        "total_failures": len(failures),
        "failures": failures[:15],  # return top 15 failures
        "detailed_sample": detailed[:5],
    }


# Static web UI
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(web_dir / "index.html"))
