"""
Fixed-size recursive character sliding window chunker.
Splits text recursively using delimiters while tracking character offsets, page boundaries, and sections.
"""

from typing import List, Tuple
from evidencebench.chunking.base import BaseChunker
from evidencebench.models import Document, Chunk, ChunkMetadata


class FixedRecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    @property
    def name(self) -> str:
        return "fixed_recursive"

    def _split_text(self, text: str, start_offset: int) -> List[Tuple[str, int, int]]:
        """
        Recursively splits text into chunks of at most chunk_size with chunk_overlap.
        Returns list of (chunk_text, start_char, end_char).
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [(text.strip(), start_offset, start_offset + len(text))]

        chunks: List[Tuple[str, int, int]] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                # Try to break at a natural delimiter
                split_at = -1
                chunk_candidate = text[start:end]
                for sep in self.separators:
                    last_sep_idx = chunk_candidate.rfind(sep)
                    if last_sep_idx != -1 and last_sep_idx > (self.chunk_size // 3):
                        split_at = start + last_sep_idx + len(sep)
                        break
                if split_at != -1:
                    end = split_at

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, start_offset + start, start_offset + end))

            if end >= text_len:
                break

            # Slide window forward taking overlap into account
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _find_page_and_section(self, doc: Document, start_char: int, end_char: int) -> Tuple[int, int, str, List[str]]:
        """Maps character offsets to page numbers and section hierarchy."""
        page_start = 1
        page_end = 1
        curr_offset = 0

        for page in doc.pages:
            p_end = curr_offset + page.char_count
            if curr_offset <= start_char <= p_end:
                page_start = page.page_number
            if curr_offset <= end_char <= p_end:
                page_end = page.page_number
            curr_offset = p_end + 1

        # Match section
        section_title = "General"
        section_hierarchy = ["General"]
        for sec in doc.sections:
            if sec.start_char <= start_char < sec.end_char or (sec.start_char <= end_char <= sec.end_char):
                section_title = sec.title
                section_hierarchy = sec.hierarchy_path
                break

        return page_start, page_end, section_title, section_hierarchy

    def chunk_document(self, document: Document) -> List[Chunk]:
        raw_chunks = self._split_text(document.raw_text, start_offset=0)
        chunks: List[Chunk] = []

        for idx, (text_content, s_char, e_char) in enumerate(raw_chunks):
            p_start, p_end, sec_title, sec_path = self._find_page_and_section(
                document, s_char, e_char
            )

            meta = ChunkMetadata(
                chunk_id=f"chk_fix_{document.metadata.filename[:6]}_{idx:03d}",
                doc_id=document.metadata.doc_id,
                doc_filename=document.metadata.filename,
                file_type=document.metadata.file_type,
                page_number=p_start,
                page_end=p_end,
                section_title=sec_title,
                section_hierarchy=sec_path,
                start_char=s_char,
                end_char=e_char,
                chunking_strategy=self.name,
                chunk_index=idx,
                token_count=len(text_content.split()),
            )

            chunk = Chunk(
                metadata=meta,
                text=text_content,
                context_prefix="",  # Baseline fixed recursive uses plain text without context prefix
            )
            chunks.append(chunk)

        return chunks
