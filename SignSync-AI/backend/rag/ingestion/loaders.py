"""Document loaders: PDF, TXT and Markdown files inside the knowledge base folder.

Each loaded Document keeps its file name (source), a human title, the PDF page number
(None for text files) and a status taken from the folder it lives in:
    knowledge_base/verified/    -> status "verified"   (checked ISL reference material)
    knowledge_base/sample/      -> status "sample"     (test/demo material, not ISL truth)
    anything else               -> status "unverified" (default, treated as provisional)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("signsync.rag.ingestion")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
SKIPPED_FILE_NAMES = {"sources.md", "readme.md"}  # documentation files, not knowledge
STATUS_FOLDERS = ("verified", "sample", "unverified")


@dataclass(frozen=True)
class Document:
    text: str
    source: str          # path relative to the knowledge base, POSIX style
    title: str
    page: int | None     # 1-based PDF page, None for text files
    status: str


def status_for(relative: Path) -> str:
    parts = relative.parts
    if len(parts) > 1 and parts[0] in STATUS_FOLDERS:
        return parts[0]
    return "unverified"


def _markdown_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def load_pdf(path: Path, relative: Path) -> list[Document]:
    """One Document per page that has extractable text. Scanned pages are skipped and logged."""
    from pypdf import PdfReader  # imported lazily so other code does not need pypdf

    reader = PdfReader(str(path))
    title = path.stem.replace("_", " ").strip()
    documents: list[Document] = []
    empty_pages = 0
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            empty_pages += 1
            continue
        documents.append(Document(
            text=text,
            source=relative.as_posix(),
            title=title,
            page=number,
            status=status_for(relative),
        ))
    if empty_pages:
        logger.warning("%s: %d page(s) had no extractable text (scanned images need OCR).",
                       relative.as_posix(), empty_pages)
    return documents


def load_text(path: Path, relative: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    fallback = path.stem.replace("_", " ").strip()
    title = _markdown_title(text, fallback) if path.suffix.lower() == ".md" else fallback
    return [Document(
        text=text,
        source=relative.as_posix(),
        title=title,
        page=None,
        status=status_for(relative),
    )]


def load_document(path: Path, kb_root: Path) -> list[Document]:
    relative = path.relative_to(kb_root)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path, relative)
    return load_text(path, relative)


def iter_documents(kb_root: Path | str) -> Iterator[Document]:
    """Yield every supported document under kb_root (recursively, sorted, hidden files ignored)."""
    kb_root = Path(kb_root)
    if not kb_root.is_dir():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_root}")
    for path in sorted(kb_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(kb_root).parts):
            continue
        if path.name.lower() in SKIPPED_FILE_NAMES:
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        yield from load_document(path, kb_root)
