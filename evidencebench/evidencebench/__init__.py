"""
EvidenceBench: High-precision retrieval-augmented generation with verifiable citations and abstention gatekeeping.
"""

from evidencebench.pipeline import EvidenceBenchPipeline
from evidencebench.config import config
from evidencebench.models import (
    Document,
    Chunk,
    RetrievalCandidate,
    GenerationResult,
    ExecutionTrace,
    BenchmarkQuestion,
    BenchmarkRunMetrics,
)

__version__ = "1.0.0"
__all__ = [
    "EvidenceBenchPipeline",
    "config",
    "Document",
    "Chunk",
    "RetrievalCandidate",
    "GenerationResult",
    "ExecutionTrace",
    "BenchmarkQuestion",
    "BenchmarkRunMetrics",
]
