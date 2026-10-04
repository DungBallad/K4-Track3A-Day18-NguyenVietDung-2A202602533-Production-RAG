from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]) if isinstance(row["contexts"], (list, tuple)) else [str(row["contexts"])],
                ground_truth=str(row["ground_truth"]),
                faithfulness=float(row.get("faithfulness", 0.0) if row.get("faithfulness") is not None and not (isinstance(row.get("faithfulness"), float) and row.get("faithfulness") != row.get("faithfulness")) else 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) if row.get("answer_relevancy") is not None and not (isinstance(row.get("answer_relevancy"), float) and row.get("answer_relevancy") != row.get("answer_relevancy")) else 0.0),
                context_precision=float(row.get("context_precision", 0.0) if row.get("context_precision") is not None and not (isinstance(row.get("context_precision"), float) and row.get("context_precision") != row.get("context_precision")) else 0.0),
                context_recall=float(row.get("context_recall", 0.0) if row.get("context_recall") is not None and not (isinstance(row.get("context_recall"), float) and row.get("context_recall") != row.get("context_recall")) else 0.0),
            )
            for _, row in df.iterrows()
        ]

        def _safe_mean(col_name):
            val = result.get(col_name)
            if val is not None:
                return float(val)
            vals = [getattr(p, col_name) for p in per_question]
            return float(sum(vals) / len(vals)) if vals else 0.0

        return {
            "faithfulness": _safe_mean("faithfulness"),
            "answer_relevancy": _safe_mean("answer_relevancy"),
            "context_precision": _safe_mean("context_precision"),
            "context_recall": _safe_mean("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}", flush=True)
        per_question = [
            EvalResult(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=gt,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    if not eval_results:
        return []

    scored_items = []
    for item in eval_results:
        scores = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg_score = sum(scores.values()) / len(scores)
        worst_metric = min(scores.keys(), key=lambda k: scores[k])
        diagnosis, fix = diagnostic_tree.get(worst_metric, ("Unknown issue", "Review pipeline logs"))
        scored_items.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "worst_metric": worst_metric,
            "score": round(scores[worst_metric], 4),
            "avg_score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored_items.sort(key=lambda x: (x["avg_score"], x["score"]))
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
