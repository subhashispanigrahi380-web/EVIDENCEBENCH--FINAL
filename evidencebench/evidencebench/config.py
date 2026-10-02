"""
Configuration module for EvidenceBench RAG system.
Contains tunable hyperparameters for ingestion, chunking, retrieval, reranking, and abstention.
"""

from pathlib import Path
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
BENCHMARK_DIR = DATA_DIR / "benchmark"
INDICES_DIR = DATA_DIR / "indices"


class EvidenceBenchConfig(BaseModel):
    # Ingestion & Document Store
    storage_dir: Path = INDICES_DIR
    corpus_dir: Path = CORPUS_DIR
    benchmark_file: Path = BENCHMARK_DIR / "dataset.json"

    # Chunking Hyperparameters
    fixed_chunk_size: int = 500  # characters
    fixed_chunk_overlap: int = 100  # characters
    structural_max_chunk_size: int = 800  # characters
    structural_min_chunk_size: int = 150  # characters

    # Dense Embedding
    dense_model_name: str = "all-MiniLM-L6-v2"
    dense_batch_size: int = 32

    # Keyword (BM25)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Hybrid Fusion (Reciprocal Rank Fusion + Normalized Score Calibration)
    rrf_k: int = 60
    dense_weight: float = 0.6
    keyword_weight: float = 0.4
    hybrid_top_k: int = 20  # Stage 1 candidate pool size

    # Reranker
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5  # Final candidate set for synthesis
    reranker_batch_size: int = 16

    # Abstention Gatekeeper Thresholds
    weak_evidence_threshold: float = 0.35  # Normalized rerank score below this triggers weak evidence abstention
    missing_evidence_overlap_threshold: float = 0.15  # Minimum lexical/semantic overlap required
    conflict_similarity_threshold: float = 0.70  # Chunks must be on same topic to trigger conflict check
    contradiction_score_threshold: float = 0.60  # Polarity/numerical discrepancy score threshold

    # Generation & Citations
    citation_overlap_min: float = 0.40  # Minimum token/subword overlap to verify citation alignment
    max_context_tokens: int = 2048


# Default singleton instance
config = EvidenceBenchConfig()
