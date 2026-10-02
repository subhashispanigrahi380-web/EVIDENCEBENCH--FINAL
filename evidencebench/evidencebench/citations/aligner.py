# -*- coding: utf-8 -*-
"""
Citation Alignment and Verification Engine.
Extracts passage-level citations, verifies grounding fidelity against source chunks,
and calculates Citation Precision and Citation Coverage metrics.
"""

import re
from typing import List, Dict, Tuple, Optional
from evidencebench.models import (
    Chunk,
    CitationSpan,
    RetrievalCandidate,
)
from evidencebench.retrieval.bm25 import STOPWORDS


class CitationAligner:
    """
    Verifies that claims made in the generated answer are grounded in the cited chunks.
    Calculates token containment, entity fidelity, and citation coverage.
    """

    CITATION_PATTERN = re.compile(
        r"\[Doc:\s*([^,\]]+?)(?:,\s*p\.\s*([0-9]+))?(?:,\s*(?:[\u00a7]|sec|section)?\s*([^\|\]]+?))?\s*\|\s*(?:chunk:\s*)?([a-zA-Z0-9_\-]+)\]"
    )

    def __init__(self, min_alignment_score: float = 0.30):
        self.min_alignment_score = min_alignment_score

    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Parses passage-level citation tags from the text."""
        citations = []
        for match in self.CITATION_PATTERN.finditer(text):
            citations.append({
                "raw_match": match.group(0),
                "doc_filename": match.group(1).strip(),
                "page": match.group(2).strip() if match.group(2) else "1",
                "section": match.group(3).strip() if match.group(3) else "General",
                "chunk_id": match.group(4).strip(),
                "start": match.start(),
                "end": match.end(),
            })
        return citations

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits answer text into claim units while preserving brackets and citation tags.
        Avoids naive period splitting which breaks on abbreviations like 'p. 1'.
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        claims = []
        for line in lines:
            if line.startswith("Based on"):
                continue
            clean_line = re.sub(r"^[•\-\*]\s*", "", line).strip()
            if clean_line:
                claims.append(clean_line)
        return claims if claims else [text.strip()]

    def _extract_keywords(self, sentence: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", sentence.lower())
        return [w for w in words if w not in STOPWORDS]

    def verify_citations(
        self,
        answer_text: str,
        retrieved_candidates: List[RetrievalCandidate],
    ) -> Tuple[List[CitationSpan], float, float]:
        """
        Verifies citations in answer text against candidate chunk text.
        Returns:
            - List of verified CitationSpan objects
            - Citation Precision (ratio of citations that are faithful)
            - Citation Coverage (ratio of sentences with valid citations)
        """
        chunk_map: Dict[str, Chunk] = {
            c.chunk.metadata.chunk_id: c.chunk for c in retrieved_candidates
        }

        sentences = self._split_into_sentences(answer_text)
        if not sentences:
            return [], 1.0, 1.0

        all_citations: List[CitationSpan] = []
        sentences_with_citation = 0

        for sentence in sentences:
            cit_matches = self.extract_citations(sentence)
            clean_claim = self.CITATION_PATTERN.sub("", sentence).strip()

            if cit_matches:
                sentences_with_citation += 1

            for cit in cit_matches:
                chunk_id = cit["chunk_id"]
                target_chunk = chunk_map.get(chunk_id)

                if target_chunk is None:
                    # Try matching by filename
                    for c in retrieved_candidates:
                        if c.chunk.metadata.doc_filename.lower() == cit["doc_filename"].lower():
                            target_chunk = c.chunk
                            break

                if target_chunk is None:
                    span = CitationSpan(
                        claim_text=clean_claim,
                        chunk_id=chunk_id,
                        doc_filename=cit["doc_filename"],
                        page_number=int(cit["page"]) if cit["page"].isdigit() else 1,
                        section_title=cit["section"],
                        quoted_snippet="",
                        alignment_score=0.0,
                        is_faithful=False,
                    )
                    all_citations.append(span)
                    continue

                # Measure overlap between claim keywords and chunk text
                claim_words = self._extract_keywords(clean_claim)
                chunk_text_lower = target_chunk.text.lower()

                if not claim_words:
                    score = 1.0
                else:
                    matched = [w for w in claim_words if w in chunk_text_lower]
                    score = len(matched) / len(claim_words)

                snippet = target_chunk.text[:120].strip() + ("..." if len(target_chunk.text) > 120 else "")
                is_faithful = score >= self.min_alignment_score

                span = CitationSpan(
                    claim_text=clean_claim,
                    chunk_id=target_chunk.metadata.chunk_id,
                    doc_filename=target_chunk.metadata.doc_filename,
                    page_number=target_chunk.metadata.page_number,
                    section_title=target_chunk.metadata.section_title,
                    quoted_snippet=snippet,
                    alignment_score=round(score, 3),
                    is_faithful=is_faithful,
                )
                all_citations.append(span)

        # Calculate Precision: faithful citations / total citations
        if not all_citations:
            citation_precision = 0.0 if len(sentences) > 0 else 1.0
        else:
            faithful_count = sum(1 for c in all_citations if c.is_faithful)
            citation_precision = round(faithful_count / len(all_citations), 3)

        citation_coverage = round(sentences_with_citation / len(sentences), 3) if sentences else 1.0

        return all_citations, citation_precision, citation_coverage
