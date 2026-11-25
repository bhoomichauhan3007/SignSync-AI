"""Split documents into overlapping chunks that keep their metadata.

Why chunk? Embedding a whole 50-page PDF gives one blurry vector. Small chunks (about
900 characters) each describe one topic, so a question can match the right paragraph.
Overlap (about 150 characters carried into the next chunk) stops an answer being cut
in half at a chunk boundary.

Algorithm:
1. Clean the text and split it into paragraphs (blank lines).
2. Break paragraphs longer than max_chars into sentences (and hard-split very long ones).
3. Pack the units into chunks up to max_chars; start the next chunk with the tail of the last.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .cleaning import clean_text
from .loaders import Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    title: str
    page: int | None
    status: str
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        while len(sentence) > max_chars:  # a single sentence longer than a chunk
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def _tail(text: str, n: int) -> str:
    """The last n characters, starting at a word boundary."""
    if n <= 0 or len(text) <= n:
        return "" if n <= 0 else text
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def _chunk_id(source: str, page: int | None, index: int, text: str) -> str:
    raw = f"{source}|{page}|{index}|{text}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def chunk_document(doc: Document, max_chars: int = 900, overlap_chars: int = 150) -> list[Chunk]:
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200 so chunks stay meaningful")
    overlap = min(max(overlap_chars, 0), max_chars // 3)

    text = clean_text(doc.text)
    paragraphs = [" ".join(p.split("\n")).strip() for p in re.split(r"\n\s*\n", text)]
    units: list[str] = []
    for paragraph in paragraphs:
        if paragraph:
            units.extend(_split_long_paragraph(paragraph, max_chars))

    pieces: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > max_chars:
            pieces.append(current)
            tail = _tail(current, overlap)
            if tail and len(tail) + 2 + len(unit) <= max_chars:
                current = f"{tail}\n\n{unit}"
            else:
                current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        pieces.append(current)

    return [
        Chunk(
            id=_chunk_id(doc.source, doc.page, index, piece),
            text=piece,
            source=doc.source,
            title=doc.title,
            page=doc.page,
            status=doc.status,
            chunk_index=index,
        )
        for index, piece in enumerate(pieces)
    ]


def chunk_documents(docs: list[Document], max_chars: int = 900, overlap_chars: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, max_chars, overlap_chars))
    return chunks
