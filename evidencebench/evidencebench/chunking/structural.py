"""
Structure-aware and section-hierarchical chunker.
Preserves semantic headings, tables, and section boundaries while prepending rich structural context.
"""

from typing import List, Tuple
from evidencebench.chunking.base import BaseChunker
from evidencebench.models import Document, Chunk, ChunkMetadata, DocumentSection


class StructureAwareChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 150):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    @property
    def name(self) -> str:
        return "structure_aware"

    def _split_section_text(self, text: str, max_size: int) -> List[str]:
        """Splits section content by paragraphs without cutting through table rows or sentences."""
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_buffer: List[str] = []
        current_len = 0

        for para in paragraphs:
            para_clean = para.strip()
            if not para_clean:
                continue

            para_len = len(para_clean)

            if current_len + para_len + 2 <= max_size:
                current_buffer.append(para_clean)
                current_len += para_len + 2
            else:
                if current_buffer:
                    chunks.append("\n\n".join(current_buffer))
                    current_buffer = []
                    current_len = 0

                # If single paragraph exceeds max_size, split by sentences
                if para_len > max_size:
                    sentences = para_clean.replace(".\n", ". \n").split(". ")
                    sub_buf: List[str] = []
                    sub_len = 0
                    for sent in sentences:
                        sent_clean = sent.strip()
                        if not sent_clean:
                            continue
                        if not sent_clean.endswith("."):
                            sent_clean += "."
                        if sub_len + len(sent_clean) + 1 <= max_size:
                            sub_buf.append(sent_clean)
                            sub_len += len(sent_clean) + 1
                        else:
                            if sub_buf:
                                chunks.append(" ".join(sub_buf))
                            sub_buf = [sent_clean]
                            sub_len = len(sent_clean)
                    if sub_buf:
                        chunks.append(" ".join(sub_buf))
                else:
                    current_buffer.append(para_clean)
                    current_len = para_len

        if current_buffer:
            chunks.append("\n\n".join(current_buffer))

        return chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_idx = 0

        # Group by sections
        sections = document.sections if document.sections else [
            DocumentSection(
                section_id="sec_root",
                title="General",
                level=1,
                hierarchy_path=["General"],
                start_char=0,
                end_char=len(document.raw_text),
                page_number=1,
            )
        ]

        for sec in sections:
            sec_text = document.raw_text[sec.start_char:sec.end_char].strip()
            if not sec_text:
                continue

            # Subdivide section content
            sub_chunks = self._split_section_text(sec_text, self.max_chunk_size)

            hierarchy_str = " > ".join(sec.hierarchy_path) if sec.hierarchy_path else sec.title
            context_prefix = f"[{document.metadata.filename} | § {hierarchy_str} | Page {sec.page_number}]"

            for sub_text in sub_chunks:
                # Find start and end char
                sub_start = document.raw_text.find(sub_text, sec.start_char)
                if sub_start == -1:
                    sub_start = sec.start_char
                sub_end = sub_start + len(sub_text)

                meta = ChunkMetadata(
                    chunk_id=f"chk_struct_{document.metadata.filename[:6]}_{chunk_idx:03d}",
                    doc_id=document.metadata.doc_id,
                    doc_filename=document.metadata.filename,
                    file_type=document.metadata.file_type,
                    page_number=sec.page_number,
                    page_end=sec.page_number,
                    section_title=sec.title,
                    section_hierarchy=sec.hierarchy_path,
                    start_char=sub_start,
                    end_char=sub_end,
                    chunking_strategy=self.name,
                    chunk_index=chunk_idx,
                    token_count=len(sub_text.split()),
                )

                chunk = Chunk(
                    metadata=meta,
                    text=sub_text,
                    context_prefix=context_prefix,
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks
