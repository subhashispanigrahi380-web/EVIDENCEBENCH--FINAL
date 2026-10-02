# -*- coding: utf-8 -*-
"""
Tests for Citation Alignment and Verification Engine.
"""

from evidencebench.models import Chunk, ChunkMetadata, DocumentType, RetrievalCandidate
from evidencebench.citations.aligner import CitationAligner


def test_citation_verification():
    aligner = CitationAligner(min_alignment_score=0.30)

    c1 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_cloud_001",
            doc_id="d1",
            doc_filename="cloud_arch.pdf",
            file_type=DocumentType.PDF,
            page_number=1,
            section_title="SLA",
        ),
        text="The NextGen Cloud Platform provides 99.999% availability and 15ms latency.",
    )
    candidates = [RetrievalCandidate(chunk=c1, score=0.9, rank=1, retrieval_method="dense")]

    answer_with_valid_citation = (
        "The system guarantees 99.999% availability across all zones "
        "[Doc: cloud_arch.pdf, p. 1, SLA | chunk: chk_cloud_001]."
    )

    citations, prec, cov = aligner.verify_citations(answer_with_valid_citation, candidates)

    assert len(citations) == 1
    assert citations[0].is_faithful is True
    assert citations[0].alignment_score >= 0.3
    assert prec == 1.0
    assert cov == 1.0


def test_unfaithful_citation_detection():
    aligner = CitationAligner(min_alignment_score=0.35)

    c1 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_cloud_001",
            doc_id="d1",
            doc_filename="cloud_arch.pdf",
            file_type=DocumentType.PDF,
            page_number=1,
            section_title="SLA",
        ),
        text="The NextGen Cloud Platform provides 99.999% availability.",
    )
    candidates = [RetrievalCandidate(chunk=c1, score=0.9, rank=1, retrieval_method="dense")]

    # Completely unrelated hallucinated claim citing the chunk
    hallucinated_answer = (
        "The system costs $500 per month for unlimited GPU cluster access "
        "[Doc: cloud_arch.pdf, p. 1, Pricing | chunk: chk_cloud_001]."
    )

    citations, prec, cov = aligner.verify_citations(hallucinated_answer, candidates)

    assert len(citations) == 1
    assert citations[0].is_faithful is False
    assert prec == 0.0
