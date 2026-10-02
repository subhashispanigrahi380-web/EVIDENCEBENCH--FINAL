"""
EvidenceBench RAG Pipeline Orchestrator.
Assembles ingestion, chunking, keyword/dense retrieval, hybrid fusion, reranking,
abstention evaluation, answer generation, and execution tracing.
"""

import time
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from evidencebench.config import config
from evidencebench.models import (
    Chunk,
    RetrievalCandidate,
    GenerationResult,
    ExecutionTrace,
)
from evidencebench.ingestion.document_store import DocumentStore
from evidencebench.chunking.fixed_recursive import FixedRecursiveChunker
from evidencebench.chunking.structural import StructureAwareChunker
from evidencebench.chunking.base import BaseChunker
from evidencebench.retrieval.bm25 import BM25Retriever
from evidencebench.retrieval.dense import DenseRetriever
from evidencebench.retrieval.fusion import CalibratedHybridFusion
from evidencebench.retrieval.reranker import CrossEncoderReranker
from evidencebench.generator.abstention import AbstentionEngine
from evidencebench.generator.engine import GroundedGenerator
from evidencebench.tracing.tracer import PipelineTracer


class EvidenceBenchPipeline:
    def __init__(
        self,
        storage_dir: Optional[Path | str] = None,
        chunking_strategy: str = "structure_aware",
        index_version: str = "v1.0",
    ):
        self.storage_dir = Path(storage_dir or config.storage_dir)
        self.index_version = index_version
        self.chunking_strategy_name = chunking_strategy

        # Ingestion & Store
        self.doc_store = DocumentStore(self.storage_dir, index_version=self.index_version)

        # Chunkers
        self.fixed_chunker = FixedRecursiveChunker(
            chunk_size=config.fixed_chunk_size,
            chunk_overlap=config.fixed_chunk_overlap,
        )
        self.structural_chunker = StructureAwareChunker(
            max_chunk_size=config.structural_max_chunk_size,
            min_chunk_size=config.structural_min_chunk_size,
        )

        # Active chunker
        self.active_chunker: BaseChunker = (
            self.structural_chunker if chunking_strategy == "structure_aware" else self.fixed_chunker
        )

        # Retrieval components
        self.bm25_retriever = BM25Retriever(k1=config.bm25_k1, b=config.bm25_b)
        self.dense_retriever = DenseRetriever(
            model_name=config.dense_model_name,
            batch_size=config.dense_batch_size,
        )
        self.hybrid_fusion = CalibratedHybridFusion(
            rrf_k=config.rrf_k,
            dense_weight=config.dense_weight,
            keyword_weight=config.keyword_weight,
        )
        self.reranker = CrossEncoderReranker(
            model_name=config.reranker_model_name,
            batch_size=config.reranker_batch_size,
        )

        # Abstention and Generation
        self.abstention_engine = AbstentionEngine(
            weak_threshold=config.weak_evidence_threshold,
            missing_overlap_threshold=config.missing_evidence_overlap_threshold,
        )
        self.generator = GroundedGenerator(abstention_engine=self.abstention_engine)

        self.indexed_chunks: List[Chunk] = []

    def set_chunking_strategy(self, strategy_name: str) -> None:
        """Dynamically switches chunking strategy and re-indexes."""
        if strategy_name == "structure_aware":
            self.active_chunker = self.structural_chunker
            self.chunking_strategy_name = "structure_aware"
        elif strategy_name == "fixed_recursive":
            self.active_chunker = self.fixed_chunker
            self.chunking_strategy_name = "fixed_recursive"
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy_name}")

        self.reindex()

    def reindex(self) -> int:
        """Re-chunks all active documents and rebuilds BM25 and Dense indexes."""
        active_docs = [
            self.doc_store.get_document(meta.doc_id)
            for meta in self.doc_store.list_documents(active_only=True)
        ]
        active_docs = [d for d in active_docs if d is not None]

        self.indexed_chunks = self.active_chunker.chunk_documents(active_docs)
        self.bm25_retriever.index(self.indexed_chunks)
        self.dense_retriever.index(self.indexed_chunks)

        return len(self.indexed_chunks)

    def query(
        self,
        query_text: str,
        pipeline_mode: str = "hybrid_rerank",  # "dense", "bm25", "hybrid", "hybrid_rerank"
        top_k: int = 5,
        target_doc_ids: Optional[List[str]] = None,
    ) -> Tuple[GenerationResult, ExecutionTrace]:
        """
        Runs query through the RAG pipeline according to pipeline_mode.
        Returns the generation result and the complete execution trace.
        """
        tracer = PipelineTracer(
            query=query_text,
            index_version=self.index_version,
            chunking_strategy=self.chunking_strategy_name,
            pipeline_mode=pipeline_mode,
        )

        # 1. Retrieval Stage
        candidates: List[RetrievalCandidate] = []

        if pipeline_mode == "bm25":
            t0 = time.perf_counter()
            candidates = self.bm25_retriever.retrieve(query_text, top_k=top_k * 4 if target_doc_ids else top_k)
            if target_doc_ids:
                candidates = [
                    c for c in candidates
                    if c.chunk.metadata.doc_id in target_doc_ids or c.chunk.metadata.doc_filename in target_doc_ids
                ][:top_k]
            t_ms = (time.perf_counter() - t0) * 1000.0
            tracer.record_stage(
                "BM25 Retrieval",
                t_ms,
                {"query": query_text, "top_k": top_k},
                {"candidates_retrieved": len(candidates)},
            )

        elif pipeline_mode == "dense":
            t0 = time.perf_counter()
            candidates = self.dense_retriever.retrieve(query_text, top_k=top_k * 4 if target_doc_ids else top_k)
            if target_doc_ids:
                candidates = [
                    c for c in candidates
                    if c.chunk.metadata.doc_id in target_doc_ids or c.chunk.metadata.doc_filename in target_doc_ids
                ][:top_k]
            t_ms = (time.perf_counter() - t0) * 1000.0
            tracer.record_stage(
                "Dense Retrieval",
                t_ms,
                {"query": query_text, "top_k": top_k},
                {"candidates_retrieved": len(candidates)},
            )

        elif pipeline_mode in ["hybrid", "hybrid_rerank"]:
            # Retrieve wider candidate sets
            t0 = time.perf_counter()
            pool_size = config.hybrid_top_k
            dense_cand = self.dense_retriever.retrieve(query_text, top_k=pool_size)
            bm25_cand = self.bm25_retriever.retrieve(query_text, top_k=pool_size)
            t_retrieval_ms = (time.perf_counter() - t0) * 1000.0
            tracer.record_stage(
                "Dual Retrieval (Dense + BM25)",
                t_retrieval_ms,
                {"query": query_text, "pool_size": pool_size},
                {"dense_count": len(dense_cand), "bm25_count": len(bm25_cand)},
            )

            # Fuse via Calibrated RRF
            t0 = time.perf_counter()
            fused_cand = self.hybrid_fusion.fuse(dense_cand, bm25_cand, top_k=pool_size)
            t_fuse_ms = (time.perf_counter() - t0) * 1000.0
            tracer.record_stage(
                "Calibrated Hybrid Fusion",
                t_fuse_ms,
                {"rrf_k": self.hybrid_fusion.rrf_k, "dense_w": self.hybrid_fusion.dense_weight},
                {"fused_candidates": len(fused_cand)},
            )

            # If target_doc_ids provided, filter candidates before reranking
            if target_doc_ids:
                fused_cand = [
                    c for c in fused_cand
                    if c.chunk.metadata.doc_id in target_doc_ids or c.chunk.metadata.doc_filename in target_doc_ids
                ]

            if pipeline_mode == "hybrid":
                candidates = fused_cand[:top_k]
            else:
                # Stage 2: Rerank top pool
                t0 = time.perf_counter()
                reranked_cand = self.reranker.rerank(query_text, fused_cand, top_k=top_k)
                t_rerank_ms = (time.perf_counter() - t0) * 1000.0
                tracer.record_stage(
                    "Cross-Encoder Reranking",
                    t_rerank_ms,
                    {"model": config.reranker_model_name, "rerank_input_size": len(fused_cand)},
                    {"top_score": reranked_cand[0].score if reranked_cand else 0.0},
                )
                candidates = reranked_cand

        # Record candidate table into trace
        tracer.record_candidates(candidates)

        # 2. Generation & Abstention Stage
        t0 = time.perf_counter()
        gen_result = self.generator.generate(query_text, candidates)
        t_gen_ms = (time.perf_counter() - t0) * 1000.0
        tracer.record_stage(
            "Generation & Abstention Gatekeeper",
            t_gen_ms,
            {"candidates_evaluated": len(candidates)},
            {
                "decision": gen_result.decision.decision,
                "reason": gen_result.decision.reason.value,
                "citation_count": len(gen_result.citations),
                "citation_precision": gen_result.citation_precision,
            },
        )

        trace = tracer.finalize(gen_result)
        return gen_result, trace
