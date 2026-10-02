"""
Tests for BM25 Keyword Retrieval and Dense Vector Retrieval.
"""

from evidencebench.models import Chunk, ChunkMetadata, DocumentType
from evidencebench.retrieval.bm25 import BM25Retriever
from evidencebench.retrieval.dense import DenseRetriever


def create_sample_chunks():
    c1 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_1",
            doc_id="d1",
            doc_filename="cloud.pdf",
            file_type=DocumentType.PDF,
            page_number=1,
            section_title="Uptime SLA",
        ),
        text="NextGen Cloud Platform guarantees 99.999% availability and 15ms latency.",
        context_prefix="[cloud.pdf | § Uptime SLA]",
    )
    c2 = Chunk(
        metadata=ChunkMetadata(
            chunk_id="chk_2",
            doc_id="d2",
            doc_filename="climate.md",
            file_type=DocumentType.MARKDOWN,
            page_number=1,
            section_title="Emissions",
        ),
        text="The European Union mandates a 55% emissions reduction target by 2030.",
        context_prefix="[climate.md | § Emissions]",
    )
    return [c1, c2]


def test_bm25_retrieval():
    chunks = create_sample_chunks()
    bm25 = BM25Retriever()
    bm25.index(chunks)

    # Keyword match for cloud
    results = bm25.retrieve("99.999% availability", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.metadata.chunk_id == "chk_1"
    assert results[0].score > 0.0

    # Keyword match for climate
    results2 = bm25.retrieve("emissions reduction 2030", top_k=2)
    assert len(results2) >= 1
    assert results2[0].chunk.metadata.chunk_id == "chk_2"


def test_dense_retrieval():
    chunks = create_sample_chunks()
    dense = DenseRetriever()
    dense.index(chunks)

    # Paraphrased conceptual query (semantic matching)
    results = dense.retrieve("service uptime resilience and low delay limits", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.metadata.chunk_id == "chk_1"
    assert results[0].score > 0.3
