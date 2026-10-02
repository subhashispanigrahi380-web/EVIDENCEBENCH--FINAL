"""
Abstention Engine (Gatekeeper).
Evaluates retrieved evidence and refuses to answer when evidence is weak, missing, or conflicting.
"""

import re
from typing import List, Tuple, Optional
from evidencebench.models import (
    RetrievalCandidate,
    AbstentionDecision,
    AbstentionReason,
)
from evidencebench.retrieval.bm25 import STOPWORDS


class AbstentionEngine:
    """
    Evaluates evidence quality across three dimensions:
    1. Missing Evidence: Query terms/entities have no support in the retrieved pool.
    2. Weak Evidence: Reranker confidence scores fall below threshold.
    3. Conflicting Evidence: Top candidates from different documents present contradictory statements.
    """

    def __init__(
        self,
        weak_threshold: float = 0.05,
        missing_overlap_threshold: float = 0.15,
        top_k_check: int = 3,
    ):
        self.weak_threshold = weak_threshold
        self.missing_overlap_threshold = missing_overlap_threshold
        self.top_k_check = top_k_check

    def _extract_query_keywords(self, query: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", query.lower())
        return [w for w in words if w not in STOPWORDS]

    def _check_missing_evidence(
        self, query: str, candidates: List[RetrievalCandidate]
    ) -> Tuple[bool, float, str]:
        """Checks if the query concepts exist in the candidate pool."""
        if not candidates:
            return True, 0.0, "No candidates retrieved for query."

        query_keywords = self._extract_query_keywords(query)
        if not query_keywords:
            return False, 1.0, ""

        combined_text = " ".join([c.chunk.text.lower() for c in candidates[:self.top_k_check]])

        matched_terms = [w for w in query_keywords if w in combined_text]
        overlap_ratio = len(matched_terms) / len(query_keywords)

        if overlap_ratio < self.missing_overlap_threshold:
            missing_terms = [w for w in query_keywords if w not in combined_text]
            return (
                True,
                overlap_ratio,
                f"Missing evidence in retrieved corpus. Missing key query terms: {', '.join(missing_terms[:4])}",
            )

        return False, overlap_ratio, ""

    def _check_weak_evidence(
        self, candidates: List[RetrievalCandidate]
    ) -> Tuple[bool, float, str]:
        """Checks if reranker confidence score is too weak to guarantee grounding."""
        if not candidates:
            return True, 0.0, "Candidate pool is empty."

        top_cand = candidates[0]
        # Check reranker probability or fallback to raw score
        top_score = top_cand.method_scores.get("reranker_prob", top_cand.score)

        if top_score < self.weak_threshold:
            return (
                True,
                top_score,
                f"Weak evidence: Top candidate confidence ({top_score:.3f}) is below threshold ({self.weak_threshold:.2f}).",
            )

        return False, top_score, ""

    def _detect_conflicts(
        self, candidates: List[RetrievalCandidate]
    ) -> Tuple[bool, List[str], str]:
        """
        Detects factual contradictions between top candidates:
        - Numerical discrepancies regarding the same entities/metrics.
        - Polarity reversals (e.g., 'approved' vs 'rejected', 'increase' vs 'decrease').
        """
        top_pool = candidates[: min(4, len(candidates))]
        if len(top_pool) < 2:
            return False, [], ""

        # Polarity antonym pairs
        polarity_pairs = [
            ("approved", "rejected"),
            ("success", "failed"),
            ("compliant", "non-compliant"),
            ("decreased", "increased"),
            ("growth", "decline"),
            ("mandatory", "optional"),
            ("supported", "unsupported"),
            ("deprecated", "active"),
            ("true", "false"),
            ("enabled", "disabled"),
        ]

        conflicting_ids: List[str] = []
        conflict_reasons: List[str] = []

        # Check pairs from different documents
        for i in range(len(top_pool)):
            for j in range(i + 1, len(top_pool)):
                c1 = top_pool[i]
                c2 = top_pool[j]

                # Only check conflict if candidates originate from different documents or versions
                if c1.chunk.metadata.doc_filename == c2.chunk.metadata.doc_filename and \
                   c1.chunk.metadata.doc_id == c2.chunk.metadata.doc_id:
                    continue

                t1 = c1.chunk.text.lower()
                t2 = c2.chunk.text.lower()

                # 1. Polarity contradiction check
                for pos, neg in polarity_pairs:
                    if (pos in t1 and neg in t2) or (neg in t1 and pos in t2):
                        # Verify topical overlap
                        w1 = set(self._extract_query_keywords(t1))
                        w2 = set(self._extract_query_keywords(t2))
                        overlap = len(w1.intersection(w2))
                        if overlap >= 3:
                            conflicting_ids.extend([c1.chunk.metadata.chunk_id, c2.chunk.metadata.chunk_id])
                            conflict_reasons.append(
                                f"Contradiction between '{c1.chunk.metadata.doc_filename}' and '{c2.chunk.metadata.doc_filename}' regarding '{pos}' vs '{neg}'."
                            )
                            break

                # 2. Numerical metric conflict check
                # Find percentages or monetary values
                num_pat = re.compile(r"(\$?[0-9]+(?:\.[0-9]+)?\s*(?:%|billion|million|bn|mn|ms|percent)?)", re.IGNORECASE)
                nums_1 = set(num_pat.findall(t1))
                nums_2 = set(num_pat.findall(t2))

                # If both mention the same metric keyword but report differing numbers
                metric_keywords = ["target", "reduction", "emissions", "budget", "revenue", "ebitda", "latency", "sla", "sla limit"]
                for mk in metric_keywords:
                    if mk in t1 and mk in t2:
                        diff_nums = nums_1.symmetric_difference(nums_2)
                        # If both have numbers and they differ
                        if nums_1 and nums_2 and diff_nums and (nums_1 != nums_2):
                            conflicting_ids.extend([c1.chunk.metadata.chunk_id, c2.chunk.metadata.chunk_id])
                            conflict_reasons.append(
                                f"Conflicting metric numbers reported for '{mk}' between {c1.chunk.metadata.doc_filename} ({list(nums_1)[:2]}) and {c2.chunk.metadata.doc_filename} ({list(nums_2)[:2]})."
                            )
                            break

                if conflicting_ids:
                    break
            if conflicting_ids:
                break

        if conflicting_ids:
            return True, list(set(conflicting_ids)), "; ".join(conflict_reasons)

        return False, [], ""

    def evaluate(
        self, query: str, candidates: List[RetrievalCandidate]
    ) -> AbstentionDecision:
        """
        Runs comprehensive abstention evaluation across missing, weak, and conflicting evidence.
        """
        if not candidates:
            return AbstentionDecision(
                decision="ABSTAIN",
                reason=AbstentionReason.MISSING_EVIDENCE,
                confidence_score=0.0,
                rationale="No relevant documents or evidence found in the index.",
                conflicting_chunk_ids=[],
                evidence_score=0.0,
            )

        # 1. Missing Evidence Check
        is_missing, overlap_score, missing_reason = self._check_missing_evidence(query, candidates)
        if is_missing:
            return AbstentionDecision(
                decision="ABSTAIN",
                reason=AbstentionReason.MISSING_EVIDENCE,
                confidence_score=overlap_score,
                rationale=missing_reason,
                conflicting_chunk_ids=[],
                evidence_score=overlap_score,
            )

        # 2. Conflicting Evidence Check
        is_conflict, conflict_ids, conflict_reason = self._detect_conflicts(candidates)
        if is_conflict:
            return AbstentionDecision(
                decision="ABSTAIN",
                reason=AbstentionReason.CONFLICTING_EVIDENCE,
                confidence_score=0.45,
                rationale=f"Refusing to answer due to contradictory sources: {conflict_reason}",
                conflicting_chunk_ids=conflict_ids,
                evidence_score=0.45,
            )

        # 3. Weak Evidence Check
        is_weak, score_val, weak_reason = self._check_weak_evidence(candidates)
        if is_weak:
            return AbstentionDecision(
                decision="ABSTAIN",
                reason=AbstentionReason.WEAK_EVIDENCE,
                confidence_score=score_val,
                rationale=weak_reason,
                conflicting_chunk_ids=[],
                evidence_score=score_val,
            )

        # If all checks pass, sufficient evidence exists to generate answer
        top_score = candidates[0].method_scores.get("reranker_prob", candidates[0].score)
        return AbstentionDecision(
            decision="ANSWER",
            reason=AbstentionReason.SUFFICIENT,
            confidence_score=top_score,
            rationale=f"Sufficient evidence retrieved (confidence {top_score:.3f} >= threshold).",
            conflicting_chunk_ids=[],
            evidence_score=top_score,
        )
