"""
Automated benchmark and ablation study execution script.
Runs all 50 questions across:
1. Dense-Only
2. Keyword-Only (BM25)
3. Calibrated Hybrid Fusion (C-RRF)
4. Calibrated Hybrid Fusion + Cross-Encoder Reranker (Structure-Aware)
5. Calibrated Hybrid Fusion + Cross-Encoder Reranker (Fixed Recursive)

Outputs detailed metrics and records all failure cases to support EXPERIMENT_REPORT.md and FAILURE_LOG.md.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from evidencebench.config import config
from evidencebench.pipeline import EvidenceBenchPipeline
from evidencebench.eval.runner import BenchmarkRunner

RESULTS_DIR = Path(__file__).parent.parent / "data" / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("EVIDENCEBENCH 30-DAY ABLATION EXPERIMENT RUNNER")
    print("=" * 80)

    pipeline = EvidenceBenchPipeline(storage_dir=config.storage_dir)
    pipeline.doc_store.ingest_directory(config.corpus_dir)
    pipeline.reindex()

    runner = BenchmarkRunner(pipeline=pipeline, dataset_path=config.benchmark_file)
    print(f"Loaded {len(runner.questions)} benchmark questions.")

    configurations = [
        ("Dense-Only", "dense", "structure_aware"),
        ("BM25-Only", "bm25", "structure_aware"),
        ("Calibrated Hybrid (C-RRF)", "hybrid", "structure_aware"),
        ("Hybrid + Reranker (Structure-Aware)", "hybrid_rerank", "structure_aware"),
        ("Hybrid + Reranker (Fixed-Recursive)", "hybrid_rerank", "fixed_recursive"),
    ]

    all_metrics = {}
    all_failures = []

    for name, mode, chunk_strat in configurations:
        print(f"\n---> Running configuration: {name} (mode={mode}, chunker={chunk_strat})...")
        pipeline.set_chunking_strategy(chunk_strat)
        metrics, detailed, failures = runner.run_evaluation(pipeline_mode=mode)
        all_metrics[name] = metrics.model_dump()

        for f in failures:
            f["configuration"] = name
            all_failures.append(f)

        print(f"     Recall@1: {metrics.recall_at_1*100:.1f}% | Recall@5: {metrics.recall_at_5*100:.1f}% | MRR@5: {metrics.mrr_at_5:.3f}")
        print(f"     Citation Precision: {metrics.citation_precision*100:.1f}% | Abstention F1: {metrics.abstention_f1*100:.1f}% | Latency: {metrics.avg_latency_ms:.1f}ms")

    # Save summary
    summary_path = RESULTS_DIR / "ablation_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    failures_path = RESULTS_DIR / "all_failures.json"
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(all_failures, f, indent=2)

    print(f"\n[SUCCESS] Saved ablation metrics to {summary_path}")
    print(f"[SUCCESS] Saved {len(all_failures)} failure cases to {failures_path}")

    # Generate Markdown Table
    print("\n### ABLATION COMPARISON TABLE ###\n")
    print("| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 | Citation Prec | Citation Cov | Abstain F1 | Abstain Acc | Avg Latency |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, m in all_metrics.items():
        print(
            f"| **{name}** | {m['recall_at_1']*100:.1f}% | {m['recall_at_3']*100:.1f}% | "
            f"{m['recall_at_5']*100:.1f}% | {m['mrr_at_5']:.3f} | {m['ndcg_at_5']:.3f} | "
            f"{m['citation_precision']*100:.1f}% | {m['citation_coverage']*100:.1f}% | "
            f"{m['abstention_f1']*100:.1f}% | {m['abstention_accuracy']*100:.1f}% | {m['avg_latency_ms']:.1f} ms |"
        )


if __name__ == "__main__":
    main()
