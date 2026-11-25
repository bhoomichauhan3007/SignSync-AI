"""Checks which SignSync AI features are ready on this computer.

Run from the project root, with the .venv active:
    python scripts/check_features.py          (fast: packages, ffmpeg, key present, index type)
    python scripts/check_features.py --live   (also makes one real Gemini call and loads the semantic model)

The API key is never printed. Nothing is changed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

rows = []


def row(feature, ok, detail):
    rows.append((feature, ok, detail))


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="make real calls (Gemini, semantic model)")
    args = parser.parse_args()

    from rag.config import RagConfig
    cfg = RagConfig.from_env()

    # Speech (Member 3)
    row("Speech to text (Whisper)", has_module("whisper"),
        "installed" if has_module("whisper") else "pip install openai-whisper")
    ffmpeg = shutil.which("ffmpeg")
    row("ffmpeg (needed by Whisper)", bool(ffmpeg), ffmpeg or "not on PATH: install it and open a new terminal")
    row("Server voice (gTTS)", has_module("gtts"), "installed" if has_module("gtts") else "pip install gTTS")

    # Sentences and text to sign (Member 3, Gemini)
    row("Gemini client package (google-genai)", has_module("google.genai"),
        "installed" if has_module("google.genai") else "pip install google-genai")
    key = cfg.gemini_api_key
    row("Gemini API key set (backend/.env)", bool(key),
        f"set ({len(key)} characters, not shown)" if key else "add GEMINI_API_KEY=... to backend/.env")

    # Tutor (Member 4)
    row("Tutor answers with Gemini", cfg.llm_provider == "gemini" and bool(key),
        f"RAG_LLM_PROVIDER={cfg.llm_provider}" + ("" if (cfg.llm_provider == "gemini" and key) else " (set RAG_LLM_PROVIDER=gemini and a key)"))
    row("Semantic search package (sentence-transformers)", has_module("sentence_transformers"),
        "installed" if has_module("sentence_transformers") else "pip install -r rag/requirements-rag.txt")
    row("Vector index package (faiss-cpu)", has_module("faiss"),
        "installed" if has_module("faiss") else "pip install faiss-cpu (optional; numpy search works)")

    manifest_path = cfg.index_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kind = manifest.get("embedder", {}).get("kind", "?")
        semantic = kind == "sentence-transformers"
        row("Tutor index uses semantic embeddings", semantic,
            f"index built with {kind} ({manifest.get('chunk_count')} chunks, store {manifest.get('backend')})"
            + ("" if semantic else " - rebuild after installing sentence-transformers: python -m rag.scripts.build_index"))
    else:
        row("Tutor index built", False, "run: python -m rag.scripts.build_index (from backend/)")

    # Word recognition (Member 2)
    row("Word recognition (TensorFlow)", has_module("tensorflow"),
        "installed" if has_module("tensorflow") else "pip install tensorflow")

    if args.live:
        try:
            from rag.generation.llm import GeminiProvider
            reply = GeminiProvider(key, cfg.gemini_model, cfg.llm_timeout_seconds).generate(
                "Reply with exactly: Gemini connection successful.", "")
            row("LIVE Gemini call", "successful" in reply.lower(), reply[:80])
        except Exception as exc:  # report, do not crash
            row("LIVE Gemini call", False, str(exc)[:160])
        if has_module("sentence_transformers"):
            try:
                from rag.embeddings.embedder import SentenceTransformerEmbedder
                vec = SentenceTransformerEmbedder(cfg.embedding_model).encode(["hello"])
                row("LIVE semantic model", vec.shape[1] > 0, f"loaded, {vec.shape[1]} dimensions")
            except Exception as exc:
                row("LIVE semantic model", False, str(exc)[:160])

    width = max(len(r[0]) for r in rows)
    print("\nSignSync AI feature check\n")
    for feature, ok, detail in rows:
        print(f"  [{'OK' if ok else '--'}] {feature.ljust(width)}  {detail}")
    ready = sum(1 for r in rows if r[1])
    print(f"\n{ready} of {len(rows)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
