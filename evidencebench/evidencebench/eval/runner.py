"""
Question-Dataset Benchmark Runner and Ablation Evaluator.
Executes test suites across pipeline configurations and chunking strategies,
logging performance metrics, ranking transitions, and failure cases.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from evidencebench.models import (
    BenchmarkQuestion,
    BenchmarkRunMetrics,
    GenerationResult,
)
from evidencebench.pipeline import EvidenceBenchPipeline
from evidencebench.eval.metrics import BenchmarkEvaluator


class BenchmarkRunner:
    def __init__(self, pipeline: EvidenceBenchPipeline, dataset_path: Path | str):
        self.pipeline = pipeline
        self.dataset_path = Path(dataset_path)
        self.questions: List[BenchmarkQuestion] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.questions = [BenchmarkQuestion.model_validate(q) for q in data]

    def run_evaluation(
        self,
        pipeline_mode: str = "hybrid_rerank",
        top_k: int = 5,
        verbose: bool = False,
    ) -> Tuple[BenchmarkRunMetrics, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Runs full evaluation for a given pipeline mode.
        Returns:
            - BenchmarkRunMetrics
            - List of evaluation results per question
            - List of failure cases (retrieval misses, false abstentions, failed abstentions)
        """
        eval_pairs = []
        detailed_results = []
        failures = []

        for q in self.questions:
            gen_res, trace = self.pipeline.query(
                query_text=q.question,
                pipeline_mode=pipeline_mode,
                top_k=top_k,
            )

            pair = {"question": q, "result": gen_res}
            eval_pairs.append(pair)

            is_abstain = gen_res.decision.decision == "ABSTAIN"
            is_relevant_hit = False

            if not q.should_abstain:
                for cand in gen_res.top_chunks[:top_k]:
                    if BenchmarkEvaluator.is_chunk_relevant(
                        cand.chunk.metadata.doc_filename,
                        cand.chunk.text,
                        q.golden_doc_filenames,
                        q.golden_passage_keywords,
                    ):
                        is_relevant_hit = True
                        break

            # Failure Detection
            failure_reason = None
            if q.should_abstain and not is_abstain:
                failure_reason = f"FAILED_TO_ABSTAIN: System answered despite expected abstention ({q.expected_abstention_reason.value if q.expected_abstention_reason else 'unspecified'})."
            elif not q.should_abstain and is_abstain:
                failure_reason = f"FALSE_ABSTAIN: System refused to answer an answerable question ({gen_res.decision.rationale})."
            elif not q.should_abstain and not is_relevant_hit:
                failure_reason = f"RETRIEVAL_MISS: Ground-truth document/keywords not found in top-{top_k} candidates."

            if failure_reason:
                failures.append({
                    "question_id": q.question_id,
                    "category": q.category.value,
                    "question": q.question,
                    "failure_type": failure_reason.split(":")[0],
                    "details": failure_reason,
                    "pipeline_mode": pipeline_mode,
                    "abstention_decision": gen_res.decision.model_dump(),
                    "top_retrieved": [
                        {
                            "chunk_id": c.chunk.metadata.chunk_id,
                            "doc": c.chunk.metadata.doc_filename,
                            "score": c.score,
                        }
                        for c in gen_res.top_chunks[:3]
                    ],
                })

            detailed_results.append({
                "question_id": q.question_id,
                "category": q.category.value,
                "question": q.question,
                "should_abstain": q.should_abstain,
                "system_decision": gen_res.decision.decision,
                "abstention_reason": gen_res.decision.reason.value,
                "confidence_score": gen_res.decision.confidence_score,
                "relevant_hit": is_relevant_hit if not q.should_abstain else None,
                "citations_count": len(gen_res.citations),
                "citation_precision": gen_res.citation_precision,
                "latency_ms": gen_res.latency_ms,
            })

        metrics = BenchmarkEvaluator.compute_metrics(eval_pairs)
        return metrics, detailed_results, failures

    def run_full_ablation(self) -> Dict[str, BenchmarkRunMetrics]:
        """
        Runs comprehensive 4-way ablation comparison across:
        1. Dense-only
        2. Keyword-only (BM25)
        3. Hybrid Fusion (C-RRF)
        4. Hybrid Fusion + Cross-Encoder Reranker
        """
        modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]
        results: Dict[str, BenchmarkRunMetrics] = {}

        for mode in modes:
            metrics, _, _ = self.run_evaluation(pipeline_mode=mode)
            results[mode] = metrics

        return results
