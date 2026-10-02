"""
Unified document parser for PDFs, Markdown, and Text files.
Preserves document, page, and section hierarchy metadata along with character offsets.
"""

import re
import hashlib
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader

from evidencebench.models import (
    Document,
    DocumentMetadata,
    DocumentPage,
    DocumentSection,
    DocumentType,
)


class UnifiedDocumentParser:
    """Parses PDF, Markdown, and TXT documents preserving structural metadata."""

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes SHA-256 fingerprint for document content deduplication."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def parse(self, file_path: str | Path) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        content_hash = self.compute_sha256(path)

        if suffix == ".pdf":
            return self._parse_pdf(path, content_hash)
        elif suffix in [".md", ".markdown"]:
            return self._parse_markdown(path, content_hash)
        elif suffix in [".txt", ".text"]:
            return self._parse_text(path, content_hash)
        else:
            # Default to text parser
            return self._parse_text(path, content_hash)

    def _parse_pdf(self, path: Path, content_hash: str) -> Document:
        reader = PdfReader(str(path))
        pages: List[DocumentPage] = []
        full_text_parts: List[str] = []
        page_char_ranges: List[Tuple[int, int, int]] = []  # (page_num, start, end)
        current_offset = 0

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            page_text = page.extract_text() or ""
            page_text = page_text.replace("\x00", "").replace("&amp;", "&").strip()
            char_count = len(page_text)

            start_char = current_offset
            end_char = current_offset + char_count
            page_char_ranges.append((page_num, start_char, end_char))
            current_offset = end_char + 2  # account for \n\n separator

            full_text_parts.append(page_text)
            pages.append(
                DocumentPage(
                    page_number=page_num,
                    text=page_text,
                    char_count=char_count,
                    sections=[],
                )
            )

        full_text = "\n\n".join(full_text_parts)

        # Detect section headings in the combined document text
        heading_pattern = re.compile(
            r"^(?:(?:[0-9]+\.){1,3}\s+[A-Z][\w\s\-,\&;]+|[A-Z][A-Z0-9\s\-,\&;]{3,50}:?)$",
            re.MULTILINE,
        )

        matches = list(heading_pattern.finditer(full_text))
        sections: List[DocumentSection] = []

        if matches:
            for idx, match in enumerate(matches):
                sec_title = match.group(0).strip()
                start_char = match.start()
                end_char = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)

                # Determine page number
                page_num = 1
                for p_num, p_start, p_end in page_char_ranges:
                    if p_start <= start_char <= p_end:
                        page_num = p_num
                        break

                sections.append(
                    DocumentSection(
                        section_id=f"sec_p{page_num}_{idx}",
                        title=sec_title,
                        level=1,
                        hierarchy_path=[sec_title],
                        start_char=start_char,
                        end_char=end_char,
                        page_number=page_num,
                    )
                )
        else:
            sections.append(
                DocumentSection(
                    section_id="sec_root",
                    title="General",
                    level=1,
                    hierarchy_path=["General"],
                    start_char=0,
                    end_char=len(full_text),
                    page_number=1,
                )
            )

        metadata = DocumentMetadata(
            filename=path.name,
            file_type=DocumentType.PDF,
            content_hash=content_hash,
            total_pages=len(pages),
            total_sections=len(sections),
            total_characters=len(full_text),
        )

        return Document(
            metadata=metadata,
            raw_text=full_text,
            pages=pages,
            sections=sections,
        )

    def _parse_markdown(self, path: Path, content_hash: str) -> Document:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        pages: List[DocumentPage] = []
        sections: List[DocumentSection] = []

        # Parse Markdown headers (# H1, ## H2, ### H3)
        header_regex = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        hierarchy_stack: List[Tuple[int, str]] = []
        matches = list(header_regex.finditer(raw_text))

        if matches:
            for idx, match in enumerate(matches):
                level = len(match.group(1))
                title = match.group(2).strip()
                start_char = match.start()
                end_char = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)

                # Maintain hierarchy stack
                while hierarchy_stack and hierarchy_stack[-1][0] >= level:
                    hierarchy_stack.pop()
                hierarchy_stack.append((level, title))

                hierarchy_path = [h[1] for h in hierarchy_stack]

                sections.append(
                    DocumentSection(
                        section_id=f"sec_md_{idx}",
                        title=title,
                        level=level,
                        hierarchy_path=hierarchy_path,
                        start_char=start_char,
                        end_char=end_char,
                        page_number=1,
                    )
                )
        else:
            sections.append(
                DocumentSection(
                    section_id="sec_root",
                    title="General",
                    level=1,
                    hierarchy_path=["General"],
                    start_char=0,
                    end_char=len(raw_text),
                    page_number=1,
                )
            )

        # For markdown, we treat the entire document as page 1 (or logical pages if page breaks present)
        pages.append(
            DocumentPage(
                page_number=1,
                text=raw_text,
                char_count=len(raw_text),
                sections=[s.title for s in sections],
            )
        )

        metadata = DocumentMetadata(
            filename=path.name,
            file_type=DocumentType.MARKDOWN,
            content_hash=content_hash,
            total_pages=1,
            total_sections=len(sections),
            total_characters=len(raw_text),
        )

        return Document(
            metadata=metadata,
            raw_text=raw_text,
            pages=pages,
            sections=sections,
        )

    def _parse_text(self, path: Path, content_hash: str) -> Document:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        sections: List[DocumentSection] = []
        pages: List[DocumentPage] = []

        # Split text by explicit section dividers like "=== Section Name ===" or "--- Section Name ---"
        divider_pattern = re.compile(r"^(?:={3,}|-{3,})\s*([A-Za-z0-9\s\-]+)\s*(?:={3,}|-{3,})$", re.MULTILINE)
        matches = list(divider_pattern.finditer(raw_text))

        if matches:
            for idx, match in enumerate(matches):
                title = match.group(1).strip()
                start_char = match.start()
                end_char = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)

                sections.append(
                    DocumentSection(
                        section_id=f"sec_txt_{idx}",
                        title=title,
                        level=1,
                        hierarchy_path=[title],
                        start_char=start_char,
                        end_char=end_char,
                        page_number=1,
                    )
                )
        else:
            sections.append(
                DocumentSection(
                    section_id="sec_root",
                    title="Main",
                    level=1,
                    hierarchy_path=["Main"],
                    start_char=0,
                    end_char=len(raw_text),
                    page_number=1,
                )
            )

        pages.append(
            DocumentPage(
                page_number=1,
                text=raw_text,
                char_count=len(raw_text),
                sections=[s.title for s in sections],
            )
        )

        metadata = DocumentMetadata(
            filename=path.name,
            file_type=DocumentType.TEXT,
            content_hash=content_hash,
            total_pages=1,
            total_sections=len(sections),
            total_characters=len(raw_text),
        )

        return Document(
            metadata=metadata,
            raw_text=raw_text,
            pages=pages,
            sections=sections,
        )
