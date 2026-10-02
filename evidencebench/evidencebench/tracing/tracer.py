"""
Structured Pipeline Execution Tracer.
Captures microsecond-level latency, candidate ranking transitions, score breakdowns,
abstention evaluations, and token consumption across every stage of the RAG pipeline.
"""

import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from evidencebench.models import (
    ExecutionTrace,
    StageTrace,
    AbstentionDecision,
    RetrievalCandidate,
    GenerationResult,
)


class PipelineTracer:
    """
    Records and structures the execution trace of every query through the RAG pipeline.
    """

    def __init__(
        self,
        query: str,
        index_version: str = "v1.0",
        chunking_strategy: str = "structure_aware",
        pipeline_mode: str = "hybrid_rerank",
    ):
        self.trace = ExecutionTrace(
            query=query,
            index_version=index_version,
            chunking_strategy=chunking_strategy,
            pipeline_mode=pipeline_mode,
        )
        self._start_time = time.perf_counter()

    def record_stage(
        self,
        stage_name: str,
        latency_ms: float,
        input_summary: Optional[Dict[str, Any]] = None,
        output_summary: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        stage = StageTrace(
            stage_name=stage_name,
            latency_ms=round(latency_ms, 2),
            input_summary=input_summary or {},
            output_summary=output_summary or {},
            metadata=metadata or {},
        )
        self.trace.stages.append(stage)

    def record_candidates(self, candidates: List[RetrievalCandidate]) -> None:
        scores_table = []
        for c in candidates:
            entry = {
                "chunk_id": c.chunk.metadata.chunk_id,
                "doc_filename": c.chunk.metadata.doc_filename,
                "page": c.chunk.metadata.page_number,
                "section": c.chunk.metadata.section_title,
                "rank": c.rank,
                "final_score": round(c.score, 4),
                "method_scores": {k: round(v, 4) if isinstance(v, float) else v for k, v in c.method_scores.items()},
                "method_ranks": c.method_ranks,
                "snippet": c.chunk.text[:100] + "...",
            }
            scores_table.append(entry)
        self.trace.candidate_scores = scores_table

    def finalize(self, gen_result: GenerationResult) -> ExecutionTrace:
        self.trace.total_latency_ms = round((time.perf_counter() - self._start_time) * 1000.0, 2)
        self.trace.abstention_decision = gen_result.decision
        self.trace.final_output = gen_result.answer

        cit_data = []
        for cit in gen_result.citations:
            cit_data.append({
                "citation_id": cit.citation_id,
                "doc_filename": cit.doc_filename,
                "page": cit.page_number,
                "section": cit.section_title,
                "chunk_id": cit.chunk_id,
                "alignment_score": cit.alignment_score,
                "is_faithful": cit.is_faithful,
                "claim": cit.claim_text[:120],
            })
        self.trace.citations = cit_data

        return self.trace

    def to_json(self) -> str:
        return self.trace.model_dump_json(indent=2)
