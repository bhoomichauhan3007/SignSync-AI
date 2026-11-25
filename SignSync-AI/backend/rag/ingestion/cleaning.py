"""Text normalisation applied to every document before chunking.

Steps: Unicode NFKC normalisation, line-ending fix, removal of soft hyphens, joining words
split across lines ("tech-\\nnology" -> "technology"), removal of lines that are only a
page marker ("Page 3 of 10"), collapsing spaces and extra blank lines.
"""
from __future__ import annotations

import re
import unicodedata

_PAGE_MARKER_LINE = re.compile(r"^\s*(page\s+\d+(\s+of\s+\d+)?|\d+\s+of\s+\d+)\s*$", re.IGNORECASE)
_HYPHEN_LINE_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = _HYPHEN_LINE_BREAK.sub(r"\1\2", text)
    kept_lines = []
    for line in text.split("\n"):
        if _PAGE_MARKER_LINE.match(line):
            continue
        kept_lines.append(re.sub(r"[ \t\f\v]+", " ", line).strip())
    text = "\n".join(kept_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
