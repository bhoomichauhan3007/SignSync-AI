"""Ask the tutor from the terminal, or test retrieval only.

Usage (run from the backend folder):
    python -m rag.scripts.query_cli "How can I learn Indian Sign Language?"
    python -m rag.scripts.query_cli --retrieve-only --show-scores "What is an emergency number?"
    python -m rag.scripts.query_cli            # interactive mode, type 'quit' to stop
"""
from __future__ import annotations

import argparse
import sys

from .. import assistant


def _print_sources(sources: list[dict]) -> None:
    if not sources:
        print("Sources: none")
        return
    print("Sources:")
    for i, src in enumerate(sources, start=1):
        page = f", page {src['page']}" if src.get("page") else ""
        print(f"  [S{i}] {src['title']}{page} | status={src['status']} | score={src['score']}")


def _retrieve_only(question: str, top_k: int | None) -> int:
    retriever, error = assistant.get_retriever()
    if error:
        print(error)
        return 1
    hits = retriever.retrieve(question, top_k=top_k)
    print(f"Retrieved {len(hits)} passage(s) (min_score={retriever.min_score}):")
    for i, hit in enumerate(hits, start=1):
        excerpt = " ".join(hit.text.split())[:240]
        print(f"\n[{i}] {hit.title} | page={hit.page} | status={hit.status} | score={hit.score:.3f}\n    {excerpt}")
    if not hits:
        print("No passage passed the relevance threshold.")
    return 0


def _answer(question: str, retrieve_only: bool, top_k: int | None) -> int:
    if retrieve_only:
        return _retrieve_only(question, top_k)
    result = assistant.ask(question)
    print(result["answer"])
    print()
    _print_sources(result["sources"])
    print(f"(mode: {result['mode']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SignSync AI tutor - terminal client.")
    parser.add_argument("question", nargs="*", help="question text (omit for interactive mode)")
    parser.add_argument("--retrieve-only", action="store_true", help="show retrieved passages and scores, no LLM")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    if args.question:
        return _answer(" ".join(args.question), args.retrieve_only, args.top_k)

    print("SignSync AI Tutor (type 'quit' to exit)")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in ("quit", "exit", ""):
            return 0
        _answer(question, args.retrieve_only, args.top_k)
        print()


if __name__ == "__main__":
    sys.exit(main())
