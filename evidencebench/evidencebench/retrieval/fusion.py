"""
Calibrated Reciprocal Rank Fusion (C-RRF) Engine.
Combines dense vector retrieval and BM25 keyword retrieval using rank reciprocals
augmented by min-max score intensity calibration.
"""

from typing import List, Dict, Optional
from evidencebench.models import Chunk, RetrievalCandidate


class CalibratedHybridFusion:
    """
    Combines dense and keyword retrieval candidates using:
    1. Reciprocal Rank Fusion (RRF) with channel weights (w_dense, w_sparse).
    2. Min-Max score intensity normalization.
    3. Composite calibration blending rank robustness with score magnitude.
    """

    def __init__(
        self,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        keyword_weight: float = 0.4,
        rank_score_blend: float = 0.7,
    ):
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
        self.rank_score_blend = rank_score_blend  # beta: weight on RRF vs normalized score

    def _normalize_scores(self, candidates: List[RetrievalCandidate]) -> Dict[str, float]:
        """Normalizes candidate scores into [0, 1] range via min-max scaling."""
        if not candidates:
            return {}
        raw_scores = [c.score for c in candidates]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        denom = max_s - min_s
        if denom <= 1e-9:
            return {c.chunk.metadata.chunk_id: 1.0 for c in candidates}
        return {c.chunk.metadata.chunk_id: (c.score - min_s) / denom for c in candidates}

    def fuse(
        self,
        dense_candidates: List[RetrievalCandidate],
        bm25_candidates: List[RetrievalCandidate],
        top_k: int = 20,
    ) -> List[RetrievalCandidate]:
        """
        Fuses dense and BM25 candidate lists into a single ranked list.
        """
        all_chunks: Dict[str, Chunk] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}
        dense_raw_scores: Dict[str, float] = {}
        bm25_raw_scores: Dict[str, float] = {}

        for c in dense_candidates:
            cid = c.chunk.metadata.chunk_id
            all_chunks[cid] = c.chunk
            dense_ranks[cid] = c.rank
            dense_raw_scores[cid] = c.score

        for c in bm25_candidates:
            cid = c.chunk.metadata.chunk_id
            all_chunks[cid] = c.chunk
            bm25_ranks[cid] = c.rank
            bm25_raw_scores[cid] = c.score

        # Min-max normalized scores
        dense_norm = self._normalize_scores(dense_candidates)
        bm25_norm = self._normalize_scores(bm25_candidates)

        default_unretrieved_rank = max(len(dense_candidates), len(bm25_candidates)) + 50

        fused_items = []
        for cid, chunk in all_chunks.items():
            r_dense = dense_ranks.get(cid, default_unretrieved_rank)
            r_bm25 = bm25_ranks.get(cid, default_unretrieved_rank)

            # 1. Reciprocal Rank Fusion component
            rrf_dense = self.dense_weight / (self.rrf_k + r_dense)
            rrf_bm25 = self.keyword_weight / (self.rrf_k + r_bm25)
            rrf_score = rrf_dense + rrf_bm25

            # 2. Normalized Score component
            norm_dense_val = dense_norm.get(cid, 0.0)
            norm_bm25_val = bm25_norm.get(cid, 0.0)
            score_norm = (
                self.dense_weight * norm_dense_val + self.keyword_weight * norm_bm25_val
            )

            # 3. Composite Calibrated Score
            # Scale RRF score to roughly [0, 1] for harmonious blend
            max_possible_rrf = (self.dense_weight + self.keyword_weight) / (self.rrf_k + 1)
            rrf_normalized = rrf_score / max_possible_rrf if max_possible_rrf > 0 else 0.0

            composite_score = (
                self.rank_score_blend * rrf_normalized
                + (1.0 - self.rank_score_blend) * score_norm
            )

            fused_items.append({
                "chunk": chunk,
                "composite_score": composite_score,
                "rrf_score": rrf_score,
                "score_norm": score_norm,
                "dense_rank": r_dense,
                "bm25_rank": r_bm25,
                "dense_score": dense_raw_scores.get(cid, 0.0),
                "bm25_score": bm25_raw_scores.get(cid, 0.0),
            })

        # Sort descending by composite score
        fused_items.sort(key=lambda x: x["composite_score"], reverse=True)

        final_candidates: List[RetrievalCandidate] = []
        for rank, item in enumerate(fused_items[:top_k], start=1):
            cand = RetrievalCandidate(
                chunk=item["chunk"],
                score=float(item["composite_score"]),
                rank=rank,
                retrieval_method="hybrid_rrf",
                method_scores={
                    "dense": item["dense_score"],
                    "bm25": item["bm25_score"],
                    "rrf": item["rrf_score"],
                    "normalized_blend": item["score_norm"],
                },
                method_ranks={
                    "dense": item["dense_rank"],
                    "bm25": item["bm25_rank"],
                },
            )
            final_candidates.append(cand)

        return final_candidates
