"""
Document store with SHA-256 deduplication and index version lineage.
Tracks documents, detects identical content, and handles document updates across index versions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from evidencebench.models import Document, DocumentMetadata
from evidencebench.ingestion.parser import UnifiedDocumentParser


class IngestionStatus(BaseModel):
    action: str  # "CREATED", "UPDATED_NEW_VERSION", "DUPLICATE_IGNORED"
    doc_id: str
    filename: str
    version: int
    index_version: str
    message: str


class DocumentStore:
    def __init__(self, storage_dir: Path | str, index_version: str = "v1.0"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_version = index_version
        self.manifest_file = self.storage_dir / "document_manifest.json"

        # Internal registries
        self.documents: Dict[str, Document] = {}  # doc_id -> Document
        self.hash_to_doc_id: Dict[str, str] = {}  # content_hash -> doc_id
        self.filename_to_versions: Dict[str, List[str]] = {}  # filename -> [doc_id_v1, doc_id_v2, ...]
        self.parser = UnifiedDocumentParser()

        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self.manifest_file.exists():
            return
        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.index_version = data.get("index_version", self.index_version)
            for doc_dict in data.get("documents", []):
                doc = Document.model_validate(doc_dict)
                self.documents[doc.metadata.doc_id] = doc
                self.hash_to_doc_id[doc.metadata.content_hash] = doc.metadata.doc_id
                fn = doc.metadata.filename
                if fn not in self.filename_to_versions:
                    self.filename_to_versions[fn] = []
                self.filename_to_versions[fn].append(doc.metadata.doc_id)
        except Exception as e:
            print(f"Warning: Failed to load manifest from {self.manifest_file}: {e}")

    def save_manifest(self) -> None:
        data = {
            "index_version": self.index_version,
            "document_count": len(self.documents),
            "documents": [doc.model_dump() for doc in self.documents.values()],
        }
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def ingest_file(self, file_path: Path | str) -> IngestionStatus:
        path = Path(file_path)
        doc = self.parser.parse(path)
        content_hash = doc.metadata.content_hash
        filename = doc.metadata.filename

        # 1. Exact Duplicate Check (same content hash)
        if content_hash in self.hash_to_doc_id:
            existing_doc_id = self.hash_to_doc_id[content_hash]
            existing_doc = self.documents[existing_doc_id]
            return IngestionStatus(
                action="DUPLICATE_IGNORED",
                doc_id=existing_doc_id,
                filename=filename,
                version=existing_doc.metadata.version,
                index_version=existing_doc.metadata.index_version,
                message=f"Duplicate content detected (hash: {content_hash[:10]}...). Reusing existing document ID {existing_doc_id}.",
            )

        # 2. File Update Check (same filename, different content hash)
        if filename in self.filename_to_versions and len(self.filename_to_versions[filename]) > 0:
            previous_doc_ids = self.filename_to_versions[filename]
            latest_prev_doc = self.documents[previous_doc_ids[-1]]
            new_version = latest_prev_doc.metadata.version + 1

            doc.metadata.version = new_version
            doc.metadata.index_version = self.index_version

            # Register document
            self.documents[doc.metadata.doc_id] = doc
            self.hash_to_doc_id[content_hash] = doc.metadata.doc_id
            self.filename_to_versions[filename].append(doc.metadata.doc_id)
            self.save_manifest()

            return IngestionStatus(
                action="UPDATED_NEW_VERSION",
                doc_id=doc.metadata.doc_id,
                filename=filename,
                version=new_version,
                index_version=self.index_version,
                message=f"Document '{filename}' updated. Created new version {new_version} under index {self.index_version}.",
            )

        # 3. Brand New Document
        doc.metadata.version = 1
        doc.metadata.index_version = self.index_version
        self.documents[doc.metadata.doc_id] = doc
        self.hash_to_doc_id[content_hash] = doc.metadata.doc_id
        self.filename_to_versions[filename] = [doc.metadata.doc_id]
        self.save_manifest()

        return IngestionStatus(
            action="CREATED",
            doc_id=doc.metadata.doc_id,
            filename=filename,
            version=1,
            index_version=self.index_version,
            message=f"Ingested new document '{filename}' (version 1, index {self.index_version}).",
        )

    def ingest_directory(self, dir_path: Path | str) -> List[IngestionStatus]:
        directory = Path(dir_path)
        statuses: List[IngestionStatus] = []
        if not directory.exists() or not directory.is_dir():
            return statuses

        for file_path in sorted(directory.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".md", ".markdown", ".txt"]:
                status = self.ingest_file(file_path)
                statuses.append(status)
        return statuses

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def list_documents(self, active_only: bool = True) -> List[DocumentMetadata]:
        """Returns list of documents. If active_only, returns only the latest version per filename."""
        if not active_only:
            return [doc.metadata for doc in self.documents.values()]

        active_docs: List[DocumentMetadata] = []
        for filename, doc_ids in self.filename_to_versions.items():
            if doc_ids:
                latest_doc_id = doc_ids[-1]
                active_docs.append(self.documents[latest_doc_id].metadata)
        return active_docs

    def bump_index_version(self, new_version_tag: str) -> None:
        self.index_version = new_version_tag
        self.save_manifest()
