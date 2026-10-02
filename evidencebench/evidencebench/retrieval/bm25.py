"""
First-principles Okapi BM25 keyword retrieval engine.
Implements tokenization, term-frequency inverted index, Robertson-Spärck Jones IDF,
and length-normalized BM25 scoring with parameters k1 and b.
"""

import math
import re
from typing import List, Dict, Set
from evidencebench.models import Chunk, RetrievalCandidate


# Standard English stopwords
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        self.inverted_index: Dict[str, Dict[int, int]] = {}  # token -> {chunk_idx: freq}
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases, strips punctuation, and removes common stopwords."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        return [w for w in words if w not in STOPWORDS]

    def index(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_lengths = []
        self.inverted_index = {}
        self.idf = {}

        if not chunks:
            self.avg_doc_len = 0.0
            return

        total_length = 0
        doc_frequencies: Dict[str, int] = {}

        for idx, chunk in enumerate(chunks):
            # Index chunk representation (including context prefix if present)
            tokens = self.tokenize(chunk.full_representation)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            term_counts: Dict[str, int] = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for term, count in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][idx] = count
                doc_frequencies[term] = doc_frequencies.get(term, 0) + 1

        self.avg_doc_len = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate Robertson-Spärck Jones IDF
        for term, df in doc_frequencies.items():
            # Add 1 inside log to keep IDF strictly non-negative
            idf_val = math.log(((self.corpus_size - df + 0.5) / (df + 0.5)) + 1.0)
            self.idf[term] = max(0.01, idf_val)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalCandidate]:
        if not self.chunks or self.corpus_size == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[int, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf[token]
            postings = self.inverted_index[token]

            for doc_idx, freq in postings.items():
                doc_len = self.doc_lengths[doc_idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * (numerator / denominator)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        # Sort by score descending
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        candidates: List[RetrievalCandidate] = []
        for rank, (doc_idx, score) in enumerate(sorted_indices, start=1):
            chunk = self.chunks[doc_idx]
            cand = RetrievalCandidate(
                chunk=chunk,
                score=float(score),
                rank=rank,
                retrieval_method="bm25",
                method_scores={"bm25": float(score)},
                method_ranks={"bm25": rank},
            )
            candidates.append(cand)

        return candidates
