"""
Cross-encoder reranking engine.
Re-scores wider candidate pools using deep query-document cross-attention,
exposing raw logits, normalized probabilities, and margin metrics for debug views.
"""

import math
from typing import List, Optional
from sentence_transformers import CrossEncoder
from evidencebench.models import RetrievalCandidate


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        if not candidates:
            return []

        # Prepare (query, chunk_representation) pairs
        pairs = [(query, cand.chunk.full_representation) for cand in candidates]

        # Predict cross-encoder logits
        logits = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        scored_candidates = []
        for cand, raw_logit in zip(candidates, logits):
            raw_val = float(raw_logit)
            norm_prob = self._sigmoid(raw_val)

            # Preserve stage 1 history and record stage 2 reranker score
            method_scores = dict(cand.method_scores)
            method_scores["reranker_logit"] = raw_val
            method_scores["reranker_prob"] = norm_prob

            method_ranks = dict(cand.method_ranks)
            method_ranks["pre_rerank_rank"] = cand.rank

            scored_candidates.append({
                "candidate": cand,
                "raw_logit": raw_val,
                "norm_prob": norm_prob,
                "method_scores": method_scores,
                "method_ranks": method_ranks,
            })

        # Sort descending by reranker normalized score
        scored_candidates.sort(key=lambda x: x["norm_prob"], reverse=True)

        reranked_results: List[RetrievalCandidate] = []
        for new_rank, item in enumerate(scored_candidates[:top_k], start=1):
            original_cand = item["candidate"]
            item["method_ranks"]["reranked_rank"] = new_rank

            reranked_cand = RetrievalCandidate(
                chunk=original_cand.chunk,
                score=float(item["norm_prob"]),
                rank=new_rank,
                retrieval_method="reranked",
                method_scores=item["method_scores"],
                method_ranks=item["method_ranks"],
            )
            reranked_results.append(reranked_cand)

        return reranked_results
