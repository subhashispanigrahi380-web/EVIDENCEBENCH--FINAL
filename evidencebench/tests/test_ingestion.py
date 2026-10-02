"""
Tests for Document Ingestion, Unified Parsing, and Version Lineage Store.
"""

import tempfile
import shutil
from pathlib import Path
import pytest

from evidencebench.ingestion.parser import UnifiedDocumentParser
from evidencebench.ingestion.document_store import DocumentStore
from evidencebench.models import DocumentType


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_markdown_parsing(temp_dir):
    md_file = temp_dir / "test_doc.md"
    md_file.write_text(
        "# Heading 1\n\nIntro paragraph.\n\n## Subheading 1.1\n\nDetailed content here.",
        encoding="utf-8",
    )

    parser = UnifiedDocumentParser()
    doc = parser.parse(md_file)

    assert doc.metadata.filename == "test_doc.md"
    assert doc.metadata.file_type == DocumentType.MARKDOWN
    assert len(doc.sections) == 2
    assert doc.sections[0].title == "Heading 1"
    assert doc.sections[1].title == "Subheading 1.1"
    assert doc.sections[1].hierarchy_path == ["Heading 1", "Subheading 1.1"]
    assert len(doc.metadata.content_hash) == 64


def test_text_parsing(temp_dir):
    txt_file = temp_dir / "test_doc.txt"
    txt_file.write_text(
        "=== Section Alpha ===\nContent for alpha.\n\n=== Section Beta ===\nContent for beta.",
        encoding="utf-8",
    )

    parser = UnifiedDocumentParser()
    doc = parser.parse(txt_file)

    assert doc.metadata.file_type == DocumentType.TEXT
    assert len(doc.sections) == 2
    assert doc.sections[0].title == "Section Alpha"
    assert doc.sections[1].title == "Section Beta"


def test_document_store_deduplication_and_versioning(temp_dir):
    store = DocumentStore(storage_dir=temp_dir, index_version="v1.0")

    file_a = temp_dir / "report.md"
    file_a.write_text("# Initial Report\n\nOriginal text content.", encoding="utf-8")

    # 1. Ingest brand new file
    s1 = store.ingest_file(file_a)
    assert s1.action == "CREATED"
    assert s1.version == 1
    assert len(store.documents) == 1

    # 2. Duplicate ingestion (same content)
    s2 = store.ingest_file(file_a)
    assert s2.action == "DUPLICATE_IGNORED"
    assert s2.doc_id == s1.doc_id
    assert len(store.documents) == 1

    # 3. Update document (same filename, different content)
    file_a.write_text("# Updated Report\n\nSubstantially modified content.", encoding="utf-8")
    s3 = store.ingest_file(file_a)
    assert s3.action == "UPDATED_NEW_VERSION"
    assert s3.version == 2
    assert len(store.documents) == 2

    # Active documents should only return latest version (v2)
    active = store.list_documents(active_only=True)
    assert len(active) == 1
    assert active[0].version == 2
