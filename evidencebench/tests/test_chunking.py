"""
Tests and comparisons for chunking strategies: FixedRecursive vs StructureAware.
"""

from pathlib import Path
from evidencebench.ingestion.parser import UnifiedDocumentParser
from evidencebench.chunking.fixed_recursive import FixedRecursiveChunker
from evidencebench.chunking.structural import StructureAwareChunker


def test_chunking_comparison():
    corpus_file = Path(__file__).parent.parent / "data" / "corpus" / "climate_policy_eu_2024.md"
    assert corpus_file.exists()

    parser = UnifiedDocumentParser()
    doc = parser.parse(corpus_file)

    fixed_chunker = FixedRecursiveChunker(chunk_size=300, chunk_overlap=50)
    struct_chunker = StructureAwareChunker(max_chunk_size=500, min_chunk_size=100)

    fixed_chunks = fixed_chunker.chunk_document(doc)
    struct_chunks = struct_chunker.chunk_document(doc)

    assert len(fixed_chunks) > 0
    assert len(struct_chunks) > 0

    # Verify StructureAwareChunker adds contextual prefix
    for sc in struct_chunks:
        assert "[" in sc.context_prefix
        assert doc.metadata.filename in sc.context_prefix
        assert "§" in sc.context_prefix

    # Fixed recursive chunks do not have contextual prefixes by design
    for fc in fixed_chunks:
        assert fc.context_prefix == ""

    # Check structural boundaries align with sections
    sec_titles = {s.title for s in doc.sections}
    struct_sections = {sc.metadata.section_title for sc in struct_chunks}
    assert struct_sections.intersection(sec_titles)
