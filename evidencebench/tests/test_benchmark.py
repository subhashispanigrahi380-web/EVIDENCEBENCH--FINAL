"""
Tests for Benchmark Dataset Runner and Metrics computation.
"""

from pathlib import Path
from evidencebench.config import config
from evidencebench.pipeline import EvidenceBenchPipeline
from evidencebench.eval.runner import BenchmarkRunner


def test_benchmark_runner_execution():
    pipeline = EvidenceBenchPipeline(storage_dir=config.storage_dir)
    # Ensure corpus ingested
    pipeline.doc_store.ingest_directory(config.corpus_dir)
    pipeline.reindex()

    runner = BenchmarkRunner(pipeline=pipeline, dataset_path=config.benchmark_file)
    assert len(runner.questions) == 50

    # Test runner on first 5 questions for rapid unit test verification
    runner.questions = runner.questions[:5]
    metrics, detailed, failures = runner.run_evaluation(pipeline_mode="hybrid_rerank")

    assert metrics.total_questions == 5
    assert metrics.citation_precision >= 0.8
    assert metrics.abstention_accuracy >= 0.8
