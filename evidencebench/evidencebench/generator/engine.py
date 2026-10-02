"""
Answer Generation Engine with Grounded Citations and Abstention Integration.
Synthesizes readable, concise answers tied directly to retrieved passages.
"""

import time
import re
from typing import List, Optional
from evidencebench.models import (
    GenerationResult,
    RetrievalCandidate,
    AbstentionDecision,
    CitationSpan,
)
from evidencebench.generator.abstention import AbstentionEngine
from evidencebench.citations.aligner import CitationAligner


class GroundedGenerator:
    def __init__(self, abstention_engine: Optional[AbstentionEngine] = None):
        self.abstention_engine = abstention_engine or AbstentionEngine()
        self.citation_aligner = CitationAligner()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip emoji, PDF table artifacts, and non-printable characters."""
        # Remove emoji and symbols (keep ASCII + common punctuation)
        text = re.sub(r"[^\x00-\x7F▸►•→←↑↓]", " ", text)
        text = re.sub(r"[▸►→←↑↓]", " ", text)
        # Collapse multiple spaces/newlines
        text = re.sub(r"\s{2,}", " ", text)
        # Remove lines that are just symbols, numbers, or table borders
        lines = [l.strip() for l in text.split("\n")]
        lines = [l for l in lines if len(re.sub(r"[^a-zA-Z]", "", l)) > 5]
        return " ".join(lines).strip()

    def _extract_key_sentences(self, text: str, query: str, max_sentences: int = 3) -> str:
        """Pick the most query-relevant sentences from a chunk."""
        cleaned = self._clean_text(text)
        query_tokens = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        stopwords = {"what","is","the","a","an","of","in","on","at","to","for",
                     "with","and","or","are","was","were","be","been","by","from","how"}
        query_tokens -= stopwords

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 25]

        if not sentences:
            return cleaned[:300]

        # Score by query token overlap
        def score(s):
            s_tokens = set(re.sub(r"[^\w\s]", "", s.lower()).split())
            return len(query_tokens & s_tokens)

        ranked = sorted(sentences, key=score, reverse=True)
        top = ranked[:max_sentences]

        # Return in original order
        ordered = [s for s in sentences if s in top]
        return " ".join(ordered[:max_sentences])

    def _build_answer(self, query: str, candidates: List[RetrievalCandidate]) -> str:
        """Build a concise, readable answer from top retrieved chunks."""
        top = candidates[:3]
        parts = []

        for i, cand in enumerate(top):
            c = cand.chunk
            key_text = self._extract_key_sentences(c.text, query, max_sentences=2)
            if not key_text:
                continue

            citation_tag = (
                f"[Doc: {c.metadata.doc_filename}, p. {c.metadata.page_number}"
                f", § {c.metadata.section_title} | chunk: {c.metadata.chunk_id}]"
            )
            parts.append(f"{key_text} {citation_tag}")

        if not parts:
            return "No relevant content could be extracted from the retrieved passages."

        # Return synthesized paragraphs joined cleanly
        return "\n\n".join(parts)

    def generate(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
    ) -> GenerationResult:
        start_time = time.perf_counter()

        # 1. Evaluate Abstention Gatekeeper
        decision: AbstentionDecision = self.abstention_engine.evaluate(query, candidates)

        # If Abstention Gatekeeper refuses:
        if decision.decision == "ABSTAIN":
            latency = (time.perf_counter() - start_time) * 1000.0
            abstain_text = (
                f"Cannot answer: {decision.rationale}"
            )
            return GenerationResult(
                query=query,
                answer=abstain_text,
                decision=decision,
                citations=[],
                top_chunks=candidates[:3],
                citation_precision=1.0,
                citation_coverage=1.0,
                latency_ms=latency,
                prompt_tokens=len(query.split()) + 50,
                completion_tokens=len(abstain_text.split()),
            )

        # 2. Grounded Answer Synthesis
        synthesized_answer = self._build_answer(query, candidates)

        # 3. Citation Verification and Metrics
        top_candidates = candidates[:3]
        citations, prec, cov = self.citation_aligner.verify_citations(
            synthesized_answer, top_candidates
        )

        latency = (time.perf_counter() - start_time) * 1000.0
        prompt_tokens = sum(len(c.chunk.text.split()) for c in top_candidates) + len(query.split())
        completion_tokens = len(synthesized_answer.split())

        return GenerationResult(
            query=query,
            answer=synthesized_answer,
            decision=decision,
            citations=citations,
            top_chunks=candidates,
            citation_precision=prec,
            citation_coverage=cov,
            latency_ms=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
