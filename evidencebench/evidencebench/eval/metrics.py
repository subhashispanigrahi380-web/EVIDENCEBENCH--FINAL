"""
Evaluation metrics calculation module.
Computes Recall@K, MRR@K, nDCG@K, Citation Precision, Citation Coverage,
and Abstention Precision/Recall/F1/Accuracy.
"""

import math
from typing import List, Dict, Any, Set
from evidencebench.models import (
    BenchmarkQuestion,
    GenerationResult,
    BenchmarkRunMetrics,
)


class BenchmarkEvaluator:
    @staticmethod
    def is_chunk_relevant(
        doc_filename: str,
        chunk_text: str,
        golden_docs: List[str],
        golden_keywords: List[str],
    ) -> bool:
        """Determines if a retrieved chunk matches golden criteria."""
        # 1. Document filename match
        if golden_docs:
            matched_doc = any(gd.lower() in doc_filename.lower() for gd in golden_docs)
            if not matched_doc:
                return False

        # 2. Golden keyword match (at least 50% of keywords present in chunk)
        if golden_keywords:
            text_lower = chunk_text.lower()
            matched_kw = sum(1 for kw in golden_keywords if kw.lower() in text_lower)
            if matched_kw < max(1, len(golden_keywords) * 0.4):
                return False

        return True

    @classmethod
    def compute_metrics(
        cls,
        eval_pairs: List[Dict[str, Any]],  # list of {"question": BenchmarkQuestion, "result": GenerationResult}
    ) -> BenchmarkRunMetrics:
        total = len(eval_pairs)
        if total == 0:
            return BenchmarkRunMetrics(total_questions=0)

        category_counts: Dict[str, int] = {}
        answerable_count = 0

        # Retrieval metric accumulators
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        rr_sum_5 = 0.0
        dcg_sum_5 = 0.0

        # Citation metric accumulators
        citation_prec_sum = 0.0
        citation_cov_sum = 0.0
        answered_eval_count = 0

        # Abstention metric accumulators
        # Classes: Positive = Should Abstain, Negative = Should Answer
        tp = 0  # Abstain when should abstain
        fp = 0  # Abstain when should answer
        tn = 0  # Answer when should answer
        fn = 0  # Answer when should abstain

        total_latency = 0.0
        total_tokens = 0.0

        for pair in eval_pairs:
            q: BenchmarkQuestion = pair["question"]
            res: GenerationResult = pair["result"]

            cat = q.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            total_latency += res.latency_ms
            total_tokens += res.prompt_tokens + res.completion_tokens

            # Abstention Evaluation
            is_system_abstain = res.decision.decision == "ABSTAIN"
            if q.should_abstain:
                if is_system_abstain:
                    tp += 1
                else:
                    fn += 1
            else:
                if is_system_abstain:
                    fp += 1
                else:
                    tn += 1

            # If question is answerable, evaluate retrieval & citations
            if not q.should_abstain:
                answerable_count += 1

                # Retrieval evaluations across top-5
                ranks_hit = []
                for rank_idx, cand in enumerate(res.top_chunks[:5], start=1):
                    rel = cls.is_chunk_relevant(
                        cand.chunk.metadata.doc_filename,
                        cand.chunk.text,
                        q.golden_doc_filenames,
                        q.golden_passage_keywords,
                    )
                    if rel:
                        ranks_hit.append(rank_idx)

                # Recall@K
                if any(r <= 1 for r in ranks_hit):
                    hits_at_1 += 1
                if any(r <= 3 for r in ranks_hit):
                    hits_at_3 += 1
                if any(r <= 5 for r in ranks_hit):
                    hits_at_5 += 1

                # MRR@5
                if ranks_hit:
                    rr_sum_5 += 1.0 / ranks_hit[0]

                # nDCG@5 (binary relevance)
                dcg = 0.0
                for r in ranks_hit:
                    dcg += 1.0 / math.log2(r + 1)
                # Ideal DCG for 1 relevant item = 1.0 / log2(2) = 1.0
                idcg = 1.0
                dcg_sum_5 += min(1.0, dcg / idcg)

                # Citation metrics if system generated an answer
                if not is_system_abstain:
                    answered_eval_count += 1
                    citation_prec_sum += res.citation_precision
                    citation_cov_sum += res.citation_coverage

        # Averages
        recall_1 = round(hits_at_1 / answerable_count, 4) if answerable_count > 0 else 0.0
        recall_3 = round(hits_at_3 / answerable_count, 4) if answerable_count > 0 else 0.0
        recall_5 = round(hits_at_5 / answerable_count, 4) if answerable_count > 0 else 0.0
        mrr_5 = round(rr_sum_5 / answerable_count, 4) if answerable_count > 0 else 0.0
        ndcg_5 = round(dcg_sum_5 / answerable_count, 4) if answerable_count > 0 else 0.0

        cit_prec = round(citation_prec_sum / answered_eval_count, 4) if answered_eval_count > 0 else 1.0
        cit_cov = round(citation_cov_sum / answered_eval_count, 4) if answered_eval_count > 0 else 1.0

        # Abstention Stats
        abs_acc = round((tp + tn) / total, 4) if total > 0 else 0.0
        abs_prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        abs_rec = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        abs_f1 = (
            round((2 * abs_prec * abs_rec) / (abs_prec + abs_rec), 4)
            if (abs_prec + abs_rec) > 0
            else 0.0
        )

        return BenchmarkRunMetrics(
            total_questions=total,
            category_counts=category_counts,
            recall_at_1=recall_1,
            recall_at_3=recall_3,
            recall_at_5=recall_5,
            mrr_at_5=mrr_5,
            ndcg_at_5=ndcg_5,
            citation_precision=cit_prec,
            citation_coverage=cit_cov,
            abstention_accuracy=abs_acc,
            abstention_precision=abs_prec,
            abstention_recall=abs_rec,
            abstention_f1=abs_f1,
            avg_latency_ms=round(total_latency / total, 2),
            avg_tokens=round(total_tokens / total, 1),
        )
