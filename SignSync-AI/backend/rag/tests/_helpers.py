"""Shared test helpers: build tiny PDFs and temporary knowledge bases without any external files."""
from __future__ import annotations

import tempfile
from pathlib import Path


def make_pdf(pages: list[str]) -> bytes:
    """Create a valid multi-page PDF (Helvetica text) entirely in memory."""
    count = len(pages)
    bodies: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = []
    for i, text in enumerate(pages):
        page_no = 4 + 2 * i
        content_no = page_no + 1
        kids.append(f"{page_no} 0 R")
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("latin-1")
        bodies[content_no] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        bodies[page_no] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            f"/Contents {content_no} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode()
    bodies[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {count} >>".encode()

    total = max(bodies)
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for num in range(1, total + 1):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + bodies[num] + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {total + 1}\n".encode() + b"0000000000 65535 f \n"
    for num in range(1, total + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {total + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)


def make_kb(files: dict[str, str | bytes]) -> Path:
    """Write files into a fresh temporary knowledge-base folder and return its path."""
    root = Path(tempfile.mkdtemp(prefix="rag_kb_"))
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return root


def make_index(kb: Path, index_dir: Path | None = None):
    """Build a hashing+numpy index from kb and return (store, index_dir)."""
    from ..embeddings.embedder import HashingEmbedder
    from ..ingestion.chunking import chunk_documents
    from ..ingestion.loaders import iter_documents
    from ..vector_store.store import VectorStore

    chunks = chunk_documents(list(iter_documents(kb)))
    store = VectorStore.build(HashingEmbedder(), [c.to_dict() for c in chunks], backend="numpy")
    index_dir = index_dir or Path(tempfile.mkdtemp(prefix="rag_index_"))
    store.save(index_dir)
    return store, index_dir
