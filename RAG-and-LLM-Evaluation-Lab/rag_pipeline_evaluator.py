#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) & LLM Pipeline Evaluator
Author: Bonga Manzini <contact@bongamanzini.dev>
Domain: Enterprise AI / LLMOps & Evaluation Frameworks

Computes core RAG triad metrics:
1. Context Precision: Signal-to-noise ratio in retrieved context chunks.
2. Context Recall: Degree to which retrieved contexts cover ground-truth facts.
3. Faithfulness: Validates whether the generated answer is grounded in retrieved facts (hallucination detection).
4. Answer Relevance: Semantic alignment between user query and generated response.
"""

import sys
import json
import math
import re
from typing import List, Dict, Any

def tokenize(text: str) -> set:
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9_-]+\b", text)
    stopwords = {"the", "a", "an", "is", "in", "to", "of", "and", "or", "for", "with", "that", "this", "can", "be", "it"}
    return set(t for t in tokens if t not in stopwords)

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return round(intersection / union, 3)

class RAGEvaluator:
    def __init__(self):
        pass

    def evaluate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        q_tokens = tokenize(item["question"])
        gt_context_tokens = tokenize(item["ground_truth_context"])
        gt_ans_tokens = tokenize(item["ground_truth_answer"])
        retrieved_contexts = item.get("retrieved_contexts", [])
        gen_ans_tokens = tokenize(item["generated_answer"])

        all_retrieved_tokens = set()
        precision_scores = []

        for ctx in retrieved_contexts:
            ctx_tokens = tokenize(ctx)
            all_retrieved_tokens.update(ctx_tokens)
            # Relevance of this chunk to query
            relevance = len(ctx_tokens.intersection(q_tokens)) / max(1, len(q_tokens))
            precision_scores.append(min(1.0, relevance * 2.0))

        # 1. Context Precision
        avg_precision = sum(precision_scores) / max(1, len(precision_scores)) if precision_scores else 0.0

        # 2. Context Recall (Overlap with ground truth context)
        context_recall = len(all_retrieved_tokens.intersection(gt_context_tokens)) / max(1, len(gt_context_tokens))

        # 3. Faithfulness (Is answer derived from retrieved context?)
        grounded_tokens = gen_ans_tokens.intersection(all_retrieved_tokens)
        faithfulness = len(grounded_tokens) / max(1, len(gen_ans_tokens))

        # 4. Answer Relevance (Semantic overlap with ground truth answer)
        answer_relevance = jaccard_similarity(gen_ans_tokens, gt_ans_tokens)

        # Composite RAG Quality Score
        rag_score = (avg_precision * 0.25) + (context_recall * 0.25) + (faithfulness * 0.3) + (answer_relevance * 0.2)

        return {
            "question": item["question"],
            "metrics": {
                "context_precision": round(avg_precision, 3),
                "context_recall": round(context_recall, 3),
                "faithfulness": round(faithfulness, 3),
                "answer_relevance": round(answer_relevance, 3),
                "composite_rag_score": round(rag_score, 3)
            },
            "hallucination_detected": faithfulness < 0.65
        }

    def evaluate_benchmark(self, dataset_path: str = "benchmark_dataset.json") -> Dict[str, Any]:
        with open(dataset_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        results = [self.evaluate_item(item) for item in items]
        mean_precision = sum(r["metrics"]["context_precision"] for r in results) / len(results)
        mean_recall = sum(r["metrics"]["context_recall"] for r in results) / len(results)
        mean_faithfulness = sum(r["metrics"]["faithfulness"] for r in results) / len(results)
        mean_relevance = sum(r["metrics"]["answer_relevance"] for r in results) / len(results)
        mean_rag_score = sum(r["metrics"]["composite_rag_score"] for r in results) / len(results)

        return {
            "evaluation_engine": "RAGEvalFramework-v3.0",
            "samples_evaluated": len(results),
            "aggregate_metrics": {
                "mean_context_precision": round(mean_precision, 3),
                "mean_context_recall": round(mean_recall, 3),
                "mean_faithfulness": round(mean_faithfulness, 3),
                "mean_answer_relevance": round(mean_relevance, 3),
                "mean_composite_rag_score": round(mean_rag_score, 3)
            },
            "detailed_evaluations": results
        }

if __name__ == "__main__":
    dataset_file = "AI-and-Machine-Learning/Research-Labs-and-Benchmarks/RAG-and-LLM-Evaluation-Lab/benchmark_dataset.json"
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_benchmark(dataset_file)
    print(json.dumps(report, indent=2))
