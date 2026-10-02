"""
Tests for Abstention Engine across weak, missing, and conflicting evidence.
"""

from evidencebench.models import Chunk, ChunkMetadata, DocumentType, RetrievalCandidate, AbstentionReason
from evidencebench.generator.abstention import AbstentionEngine


def test_missing_evidence_abstention():
    engine = AbstentionEngine()
    c1 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_1",
            doc_id="d1",
            doc_filename="cloud.pdf",
            file_type=DocumentType.PDF,
        ),
        text="NextGen Cloud Platform guarantees 99.999% availability.",
    )
    cand = [RetrievalCandidate(chunk=c1, score=0.2, rank=1, retrieval_method="dense")]

    # Query has completely missing concepts
    decision = engine.evaluate("What was the retail price of iPhone in Tokyo?", cand)
    assert decision.decision == "ABSTAIN"
    assert decision.reason == AbstentionReason.MISSING_EVIDENCE


def test_weak_evidence_abstention():
    engine = AbstentionEngine(weak_threshold=0.40)
    c1 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_1",
            doc_id="d1",
            doc_filename="cloud.pdf",
            file_type=DocumentType.PDF,
        ),
        text="Some marginal cloud computing notes.",
    )
    # Low score
    cand = [RetrievalCandidate(
        chunk=c1,
        score=0.25,
        rank=1,
        retrieval_method="reranked",
        method_scores={"reranker_prob": 0.25},
    )]

    decision = engine.evaluate("cloud computing notes", cand)
    assert decision.decision == "ABSTAIN"
    assert decision.reason == AbstentionReason.WEAK_EVIDENCE


def test_conflicting_evidence_abstention():
    engine = AbstentionEngine()

    c1 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_1",
            doc_id="d1",
            doc_filename="climate_2024.md",
            file_type=DocumentType.MARKDOWN,
        ),
        text="The European Union mandates a 55% emissions reduction target by 2030.",
    )
    c2 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_2",
            doc_id="d2",
            doc_filename="climate_2025_revision.md",
            file_type=DocumentType.MARKDOWN,
        ),
        text="Under the 2025 revision, the 2030 emissions reduction target was changed to 40% reduction.",
    )

    candidates = [
        RetrievalCandidate(
            chunk=c1,
            score=0.85,
            rank=1,
            retrieval_method="reranked",
            method_scores={"reranker_prob": 0.85},
        ),
        RetrievalCandidate(
            chunk=c2,
            score=0.82,
            rank=2,
            retrieval_method="reranked",
            method_scores={"reranker_prob": 0.82},
        ),
    ]

    decision = engine.evaluate("What is the EU 2030 emissions reduction target?", candidates)
    assert decision.decision == "ABSTAIN"
    assert decision.reason == AbstentionReason.CONFLICTING_EVIDENCE
    assert len(decision.conflicting_chunk_ids) >= 2
