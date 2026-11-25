"""Checks which optional parts of SignSync AI are ready on this computer. Changes nothing.

Run from the backend folder:
    python -m scripts.check_setup          quick check (no internet calls)
    python -m scripts.check_setup --live   also makes one small Gemini request to test the key
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PROJECT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from rag.config import RagConfig  # noqa: E402  (also loads backend/.env and rag/.env)


def has_package(name: str) -> bool:
    """True if the package (or a dotted sub-package such as google.genai) can be found. Never raises."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ImportError, ValueError):
        return False


def row(name: str, ok: bool, detail: str, fix: str = "") -> dict:
    return {"item": name, "status": "READY" if ok else "NOT READY", "detail": detail, "fix": "" if ok else fix}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="make one small Gemini request")
    args = parser.parse_args()
    cfg = RagConfig.from_env()
    rows = []

    rows.append(row("Python", sys.version_info >= (3, 11), f"Python {sys.version.split()[0]}", "Use Python 3.11 or 3.12"))
    rows.append(row("Whisper (speech to text)", has_package("whisper"),
                    "openai-whisper installed" if has_package("whisper") else "not installed",
                    "pip install openai-whisper"))
    ffmpeg = shutil.which("ffmpeg")
    rows.append(row("ffmpeg (audio decoding)", bool(ffmpeg), ffmpeg or "not found on PATH",
                    "Install ffmpeg and reopen the terminal"))
    rows.append(row("gTTS (server voice)", has_package("gtts"),
                    "gTTS installed" if has_package("gtts") else "not installed", "pip install gTTS"))
    key = bool(cfg.gemini_api_key)
    rows.append(row("GEMINI_API_KEY in backend/.env", key,
                    "key is set" if key else "not set", "Add GEMINI_API_KEY=... to backend/.env"))
    rows.append(row("google-genai (Gemini client)", has_package("google.genai"),
                    "installed" if has_package("google.genai") else "not installed", "pip install google-genai python-dotenv"))
    rows.append(row("Tutor answers with Gemini", cfg.llm_provider in ("auto", "gemini") and key,
                    f"RAG_LLM_PROVIDER={cfg.llm_provider}", "Set GEMINI_API_KEY; keep RAG_LLM_PROVIDER=auto"))
    rows.append(row("sentence-transformers (semantic search)", has_package("sentence_transformers"),
                    "installed" if has_package("sentence_transformers") else "not installed",
                    "pip install sentence-transformers (large download)"))
    rows.append(row("faiss-cpu (vector index)", has_package("faiss"),
                    "installed" if has_package("faiss") else "not installed", "pip install faiss-cpu"))
    manifest = cfg.index_dir / "manifest.json"
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding="utf-8"))
        kind = info.get("embedder", {}).get("kind", "?")
        rows.append(row("Tutor index built", True,
                        f"{info.get('chunk_count')} chunks, embedder={kind}, store={info.get('backend')}"))
        want_semantic = has_package("sentence_transformers")
        rows.append(row("Index uses semantic search", kind == "sentence-transformers" or not want_semantic,
                        f"index embedder is {kind}",
                        "Rebuild: python -m rag.scripts.build_index"))
    else:
        rows.append(row("Tutor index built", False, "no index yet", "python -m rag.scripts.build_index"))
    rows.append(row("TensorFlow (word recognition)", has_package("tensorflow"),
                    "installed" if has_package("tensorflow") else "not installed", "pip install tensorflow"))
    for name, path in [
        ("Letter training data (data/exports)", PROJECT / "data/exports/alphabet_dataset.npz"),
        ("Letter model (server)", PROJECT / "sign_recognition/models/alphabet_mlp.joblib"),
        ("Letter model (browser)", PROJECT / "frontend/public/models/alphabet.json"),
        ("Word model (Member 2 GRU)", PROJECT / "sign_recognition/word/artifacts/word_gru.keras"),
        ("Word labels and normalisation", PROJECT / "sign_recognition/word/artifacts/label_encoder.json"),
    ]:
        rows.append(row(name, path.exists(), str(path.relative_to(PROJECT))))

    if args.live and key:
        try:
            sys.path.insert(0, str(PROJECT / "nlp_translation"))
            from gemini_client import generate_text
            reply = generate_text("Reply with exactly: connection ok", max_tokens=20)
            rows.append(row("Gemini live test", True, f"reply: {reply[:60]}"))
        except Exception as exc:  # show the real reason
            rows.append(row("Gemini live test", False, str(exc)[:160], "Check the key, the model name and internet"))

    width = max(len(r["item"]) for r in rows)
    print(f"{'ITEM'.ljust(width)}  STATUS      DETAIL")
    for r in rows:
        print(f"{r['item'].ljust(width)}  {r['status']:<10}  {r['detail']}")
        if r["fix"]:
            print(f"{' ' * width}              -> {r['fix']}")
    ready = sum(r["status"] == "READY" for r in rows)
    print(f"\n{ready} of {len(rows)} items ready. Items marked NOT READY are optional unless you need that feature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
