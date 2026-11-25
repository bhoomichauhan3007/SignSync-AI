"""Measure retrieval and refusal behaviour on a labelled question set.

Metrics (computed exactly from the run, nothing estimated):
- source_hit_rate: for answerable questions, share where the expected file is among the retrieved sources.
- refusal_accuracy: share of all questions where the tutor refused exactly when it should (expect_source null).

Usage (from backend/):
    python -m rag.scripts.evaluate                      # uses the built index and rag/evaluation/questions.json
    python -m rag.scripts.evaluate --output rag/evaluation/results.json
A small set is only a smoke test. Use 20 to 30 questions written against real verified material for real conclusions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..assistant import get_retriever

DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent / "evaluation" / "questions.json"


def evaluate(questions: list[dict], retriever) -> dict:
    rows = []
    answerable = hits = correct_refusals = 0
    for item in questions:
        expected = item.get("expect_source")
        retrieved = retriever.retrieve(item["question"])
        sources = [c.source for c in retrieved]
        refused = not retrieved
        if expected is not None:
            answerable += 1
            hit = expected in sources
            hits += int(hit)
        else:
            hit = None
        should_refuse = expected is None
        correct_refusals += int(refused == should_refuse)
        rows.append({
            "question": item["question"],
            "expected_source": expected,
            "retrieved_sources": sources,
            "top_score": round(retrieved[0].score, 3) if retrieved else None,
            "refused": refused,
            "source_hit": hit,
            "refusal_correct": refused == should_refuse,
        })
    total = len(questions)
    return {
        "questions": total,
        "answerable": answerable,
        "source_hit_rate": round(hits / answerable, 3) if answerable else None,
        "refusal_accuracy": round(correct_refusals / total, 3) if total else None,
        "min_score": retriever.min_score,
        "top_k": retriever.top_k,
        "embedder": retriever.store.embedder.kind,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and refusal behaviour.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    args = parser.parse_args(argv)

    retriever, error = get_retriever()
    if error:
        print(error)
        return 1
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    report = evaluate(questions, retriever)

    print(f"Embedder: {report['embedder']} | top_k={report['top_k']} | min_score={report['min_score']}")
    print(f"Questions: {report['questions']} (answerable: {report['answerable']})")
    print(f"source_hit_rate:   {report['source_hit_rate']}")
    print(f"refusal_accuracy:  {report['refusal_accuracy']}")
    for row in report["rows"]:
        status = "OK " if row["refusal_correct"] and row["source_hit"] in (None, True) else "MISS"
        print(f"  [{status}] {row['question']}  refused={row['refused']}  top_score={row['top_score']}  "
              f"hit={row['source_hit']}")
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
