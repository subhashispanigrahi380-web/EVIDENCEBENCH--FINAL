"""
Dense vector retrieval engine using Sentence Transformers.
Computes L2-normalized embeddings and efficient cosine similarity matrix operations.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from evidencebench.models import Chunk, RetrievalCandidate


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None  # shape (N, dim)
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return

        texts = [chunk.full_representation for chunk in chunks]
        # Generate embeddings with L2 normalization for cosine similarity
        raw_embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = raw_embs

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalCandidate]:
        if not self.chunks or self.embeddings is None:
            return []

        # Encode query
        query_emb = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]  # shape (dim,)

        # Cosine similarity via dot product against normalized vectors
        scores = np.dot(self.embeddings, query_emb)  # shape (N,)

        # Get top-k indices
        top_k = min(top_k, len(self.chunks))
        if top_k <= 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]

        candidates: List[RetrievalCandidate] = []
        for rank, idx in enumerate(top_indices, start=1):
            score_val = float(scores[idx])
            chunk = self.chunks[idx]
            cand = RetrievalCandidate(
                chunk=chunk,
                score=score_val,
                rank=rank,
                retrieval_method="dense",
                method_scores={"dense": score_val},
                method_ranks={"dense": rank},
            )
            candidates.append(cand)

        return candidates
