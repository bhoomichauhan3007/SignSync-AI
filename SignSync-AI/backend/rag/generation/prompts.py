"""Prompt templates. The rules here are how the tutor avoids unsupported claims.

The LLM is only shown numbered passages ([S1], [S2] ...) from the knowledge base. It must cite
them, say when the passages do not answer the question, and never invent sign meanings.
Prompt rules reduce mistakes but cannot guarantee them, so the assistant also refuses without
calling the LLM when retrieval finds nothing relevant.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are the SignSync AI Tutor, an assistant for learning Indian Sign Language (ISL).

Rules you must follow:
1. Answer ONLY from the numbered passages in the user message. Do not add facts from outside them.
2. Cite the passages you rely on using their labels, for example [S1] or [S2].
3. If the passages do not contain the answer, say: "The knowledge base does not contain this information." Do not guess.
4. Never invent the meaning, handshape, movement, or location of any ISL sign. Describe a sign only when a passage describes it, and name that passage.
5. Do not add notes about verification or disclaimers at the end. The app shows each source's status itself.
6. You cannot see the user's hands or video. Never say whether the user's signing is correct.
7. Keep the answer short and practical, at most about 150 words.
"""

REFUSAL_TEXT = ("I could not find this in the ISL knowledge base, so I cannot answer it reliably. "
                "Try rephrasing the question, or ask a qualified ISL teacher.")

EMERGENCY_NOTE = ("If this is a real emergency, contact local emergency services immediately. "
                  "In India, the emergency number is 112.")

VIDEO_LIMITATION_NOTE = ("This tutor cannot see videos or images, so it cannot identify a sign you perform. "
                         "Sign recognition is handled by the separate recognition module.")

UNVERIFIED_NOTE = "Note: this material is not yet checked by a qualified ISL teacher."


def _format_history(history: list[dict] | None, max_turns: int = 4, max_chars: int = 300) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-max_turns:]:
        role = "User" if turn.get("role") == "user" else "Tutor"
        content = str(turn.get("content", "")).strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        if content:
            lines.append(f"{role}: {content}")
    return "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n" if lines else ""


def build_user_prompt(question: str, passages: list, history: list[dict] | None,
                      max_chars: int) -> tuple[str, list]:
    """Return the user prompt and the passages actually included (so citations match the sources list)."""
    included = []
    blocks = []
    used = 0
    for passage in passages:
        label = f"[S{len(included) + 1}]"
        page = f", page {passage.page}" if passage.page else ""
        block = f"{label} {passage.title}{page} (status: {passage.status})\n{passage.text.strip()}"
        if included and used + len(block) > max_chars:
            break
        included.append(passage)
        blocks.append(block)
        used += len(block)
    prompt = (
        "Numbered passages from the knowledge base:\n\n"
        + "\n\n".join(blocks)
        + "\n\n"
        + _format_history(history)
        + f"Question: {question}\n\n"
        + "Answer using only the passages above and cite labels such as [S1]."
    )
    return prompt, included


_SENTENCE_SPLIT = None


def _stem(word: str) -> str:
    """Very simple word endings so 'learning' matches 'learn' and 'shapes' matches 'shape'."""
    for suffix, minimum in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if word.endswith(suffix) and len(word) >= minimum and not word.endswith("ss"):
            return word[: -len(suffix)]
    return word


def _words(text: str) -> set[str]:
    import re
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())}


def _query_terms(question: str) -> set[str]:
    stop = {"how", "do", "i", "the", "a", "an", "is", "are", "what", "can", "to", "of", "in", "on",
            "my", "me", "for", "and", "it", "this", "that", "with", "about", "you", "tell", "please"}
    return {_stem(w) for w in _words(question) if len(w) > 1 and w not in stop}


def _units(text: str) -> list[str]:
    """Split a passage into sentence-sized units: each line (bullets and headings) and each sentence in it."""
    import re
    units = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*# ").strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+", line):
            if len(part.strip()) > 3:
                units.append(part.strip())
    return units


def best_sentences(text: str, question: str, limit: int = 3) -> str:
    """The sentences of a passage that best match the question, kept in their original order."""
    terms = _query_terms(question)
    scored = []
    for index, unit in enumerate(_units(text)):
        scored.append((len(terms & _words(unit)), index, unit))
    if not scored:
        return ""
    top = sorted(scored, key=lambda x: (-x[0], x[1]))[:limit]
    chosen = sorted([x for x in top if x[0] > 0], key=lambda x: x[1]) or sorted(top, key=lambda x: x[1])[:1]
    excerpt = " ".join(unit for _, _, unit in chosen)
    return excerpt[:600].rsplit(" ", 1)[0] + " ..." if len(excerpt) > 600 else excerpt


def format_extractive(passages: list, question: str = "") -> str:
    """Answer used when no LLM is configured: the best-matching sentences from each passage, labelled.

    No AI rewriting happens here, so every sentence shown is copied from the knowledge base.
    """
    lines = ["From the knowledge base (no AI summary, so these are the document's own sentences):"]
    for i, passage in enumerate(passages, start=1):
        page = f", page {passage.page}" if passage.page else ""
        excerpt = best_sentences(passage.text, question) if question else " ".join(passage.text.split())[:350]
        lines.append(f"[S{i}] {passage.title}{page}: {excerpt}")
    return "\n\n".join(lines)
