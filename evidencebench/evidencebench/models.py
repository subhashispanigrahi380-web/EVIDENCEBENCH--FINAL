"""
Domain models for EvidenceBench.
Defines strongly-typed schemas for documents, chunks, retrieval results,
citations, traces, and benchmark evaluation.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid


class DocumentType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class DocumentMetadata(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_type: DocumentType
    content_hash: str  # SHA-256 hash of raw file content
    version: int = 1
    index_version: str = "v1"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_pages: int = 1
    total_sections: int = 0
    total_characters: int = 0
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    section_id: str
    title: str
    level: int = 1
    hierarchy_path: List[str] = Field(default_factory=list)  # e.g., ["Chapter 1", "Section 1.2"]
    start_char: int = 0
    end_char: int = 0
    page_number: int = 1


class DocumentPage(BaseModel):
    page_number: int
    text: str
    char_count: int
    sections: List[str] = Field(default_factory=list)


class Document(BaseModel):
    metadata: DocumentMetadata
    raw_text: str
    pages: List[DocumentPage] = Field(default_factory=list)
    sections: List[DocumentSection] = Field(default_factory=list)


class ChunkMetadata(BaseModel):
    chunk_id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:8]}")
    doc_id: str
    doc_filename: str
    file_type: DocumentType
    page_number: int = 1
    page_end: int = 1
    section_title: str = "General"
    section_hierarchy: List[str] = Field(default_factory=list)
    start_char: int = 0
    end_char: int = 0
    chunking_strategy: str = "fixed_recursive"  # or "structure_aware"
    chunk_index: int = 0
    token_count: int = 0
    content_hash: str = ""


class Chunk(BaseModel):
    metadata: ChunkMetadata
    text: str
    context_prefix: str = ""  # e.g. "[Document: doc.pdf | Section: Overview]"

    @property
    def full_representation(self) -> str:
        if self.context_prefix:
            return f"{self.context_prefix}\n{self.text}"
        return self.text


class RetrievalCandidate(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    retrieval_method: str  # "bm25", "dense", "hybrid_rrf", "reranked"
    method_scores: Dict[str, float] = Field(default_factory=dict)
    method_ranks: Dict[str, int] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    method: str
    candidates: List[RetrievalCandidate] = Field(default_factory=list)
    latency_ms: float = 0.0
    total_indexed_chunks: int = 0


class CitationSpan(BaseModel):
    citation_id: str = Field(default_factory=lambda: f"cit_{uuid.uuid4().hex[:6]}")
    claim_text: str
    chunk_id: str
    doc_filename: str
    page_number: int
    section_title: str
    quoted_snippet: str
    alignment_score: float = 0.0  # Precision score of evidence support
    is_faithful: bool = True


class AbstentionReason(str, Enum):
    SUFFICIENT = "sufficient_evidence"
    WEAK_EVIDENCE = "weak_evidence"
    MISSING_EVIDENCE = "missing_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    OUT_OF_DOMAIN = "out_of_domain"


class AbstentionDecision(BaseModel):
    decision: str  # "ANSWER" or "ABSTAIN"
    reason: AbstentionReason
    confidence_score: float  # 0.0 to 1.0
    rationale: str
    conflicting_chunk_ids: List[str] = Field(default_factory=list)
    evidence_score: float = 0.0


class GenerationResult(BaseModel):
    query: str
    answer: str
    decision: AbstentionDecision
    citations: List[CitationSpan] = Field(default_factory=list)
    top_chunks: List[RetrievalCandidate] = Field(default_factory=list)
    citation_precision: float = 1.0
    citation_coverage: float = 1.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StageTrace(BaseModel):
    stage_name: str
    latency_ms: float
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str
    index_version: str
    chunking_strategy: str
    pipeline_mode: str  # "dense", "bm25", "hybrid", "hybrid_rerank"
    stages: List[StageTrace] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    abstention_decision: Optional[AbstentionDecision] = None
    candidate_scores: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    final_output: str = ""


class QuestionCategory(str, Enum):
    DIRECT = "direct"
    PARAPHRASED = "paraphrased"
    MULTI_DOCUMENT = "multi_document"
    CONFLICTING = "conflicting"
    UNANSWERABLE = "unanswerable"


class BenchmarkQuestion(BaseModel):
    question_id: str
    category: QuestionCategory
    question: str
    expected_answer: Optional[str] = None
    should_abstain: bool = False
    expected_abstention_reason: Optional[AbstentionReason] = None
    golden_doc_filenames: List[str] = Field(default_factory=list)
    golden_passage_keywords: List[str] = Field(default_factory=list)
    conflicting_source_pairs: Optional[List[List[str]]] = None


class BenchmarkRunMetrics(BaseModel):
    total_questions: int
    category_counts: Dict[str, int] = Field(default_factory=dict)
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    citation_precision: float = 0.0
    citation_coverage: float = 0.0
    abstention_accuracy: float = 0.0
    abstention_precision: float = 0.0
    abstention_recall: float = 0.0
    abstention_f1: float = 0.0
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0
