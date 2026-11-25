
# Member 3: Gemini-powered NLP functions
# Existing isl_nlp.py remains unchanged.

import json
import re
from gemini_client import generate_text


def words_to_sentence(words, language="English"):
    """Convert ISL gloss words into a natural sentence."""

    if not words:
        return ""

    prompt = f"""
Convert these Indian Sign Language gloss words into one
short, natural {language} sentence.

Gloss: {" ".join(words)}

Rules:
- Preserve the original meaning.
- Add only necessary grammar words.
- Do not invent facts.
- IloveYou means I love you.
- Return only the sentence.
"""

    try:
        return generate_text(prompt, max_tokens=120)
    except Exception as e:
        print("Gemini sentence error:", e)
        sentence = " ".join(w.replace("_", " ").lower() for w in words)
        return sentence.capitalize() + "."


INTENTS = [
    "greeting", "request", "question",
    "emergency", "thanks", "farewell", "statement"
]


def extract_intent(sentence):
    """Identify intent and important entities."""

    prompt = f"""
Classify this sentence.

Allowed intents: {", ".join(INTENTS)}

Sentence: "{sentence}"

Return ONLY valid JSON in this format:
{{"intent": "greeting", "entities": []}}
"""

    try:
        result = generate_text(prompt, max_tokens=120)
        result = result.replace("```json", "").replace("```", "").strip()
        data = json.loads(result)

        if data.get("intent") not in INTENTS:
            raise ValueError("Invalid intent")

        return {
            "intent": data["intent"],
            "entities": data.get("entities", [])
        }

    except Exception as e:
        print("Gemini intent error:", e)

        s = sentence.lower()

        if any(w in s for w in ["help", "emergency", "doctor", "hospital", "police"]):
            intent = "emergency"
        elif any(w in s for w in ["hello", "namaste", "good morning"]):
            intent = "greeting"
        elif "thank" in s:
            intent = "thanks"
        elif any(w in s for w in ["bye", "goodbye"]):
            intent = "farewell"
        elif s.strip().endswith("?"):
            intent = "question"
        elif any(w in s for w in ["please", "want", "need"]):
            intent = "request"
        else:
            intent = "statement"

        return {"intent": intent, "entities": []}


STOPWORDS = {
    "a", "an", "the", "is", "am", "are",
    "was", "were", "be", "to", "of", "do", "does"
}


def text_to_gloss(text, vocab):
    """Convert ordinary text into ISL-style keywords."""

    known_signs = ", ".join(
        word for word in vocab if len(word) > 1
    )

    prompt = f"""
Convert this sentence into Indian Sign Language gloss.

Sentence: "{text}"

Known signs: {known_signs}

Rules:
- Use uppercase keywords.
- Remove articles and auxiliary verbs.
- Prefer subject-object-verb order.
- Put question words last.
- Use known signs only when their meaning matches.
- Return only keywords separated by spaces.
"""

    try:
        result = generate_text(prompt, max_tokens=120)
        return result.upper().split()

    except Exception as e:
        print("Gemini gloss error:", e)

        return [
            word.upper()
            for word in re.findall(r"[a-zA-Z']+", text.lower())
            if word not in STOPWORDS
        ]
