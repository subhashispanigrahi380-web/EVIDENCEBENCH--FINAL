"""
Tests for Calibrated Hybrid Fusion (C-RRF).
"""

from evidencebench.models import Chunk, ChunkMetadata, DocumentType, RetrievalCandidate
from evidencebench.retrieval.fusion import CalibratedHybridFusion


def test_hybrid_fusion_ranking():
    c1 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_1",
            doc_id="d1",
            doc_filename="doc1.pdf",
            file_type=DocumentType.PDF,
        ),
        text="Chunk one content",
    )
    c2 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_2",
            doc_id="d2",
            doc_filename="doc2.pdf",
            file_type=DocumentType.PDF,
        ),
        text="Chunk two content",
    )

    # Dense ranked: c1=1, c2=2
    dense_cand = [
        RetrievalCandidate(chunk=c1, score=0.9, rank=1, retrieval_method="dense"),
        RetrievalCandidate(chunk=c2, score=0.4, rank=2, retrieval_method="dense"),
    ]

    # BM25 ranked: c2=1, c1=2
    bm25_cand = [
        RetrievalCandidate(chunk=c2, score=4.5, rank=1, retrieval_method="bm25"),
        RetrievalCandidate(chunk=c1, score=2.1, rank=2, retrieval_method="bm25"),
    ]

    fusion = CalibratedHybridFusion(rrf_k=60, dense_weight=0.6, keyword_weight=0.4)
    fused = fusion.fuse(dense_cand, bm25_cand, top_k=2)

    assert len(fused) == 2
    # Verify both methods contributed
    for fc in fused:
        assert "dense" in fc.method_scores
        assert "bm25" in fc.method_scores
        assert "rrf" in fc.method_scores
        assert fc.score > 0.0
