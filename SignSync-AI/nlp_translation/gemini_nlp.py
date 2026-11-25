
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
    """Convert text into gloss while preserving important words."""

    if not text or not text.strip():
        return []

    known_signs = [word for word in vocab if len(word) > 1]

    prompt = f"""
Convert this English sentence into simple ISL-style gloss keywords.

Sentence: {text}

Available whole-word signs:
{", ".join(known_signs)}

Rules:
- Return uppercase keywords separated by spaces.
- Prefer available signs when their meaning matches.
- Remove articles and unnecessary auxiliary verbs.
- Preserve all meaningful words, especially verbs, requests and names.
- Keep unavailable words intact for alphabet fingerspelling.
- Never invent or substitute unrelated signs.
- Return only keywords.
"""

    try:
        result = generate_text(prompt, max_tokens=120)
        generated = re.findall(r"[A-Z]+", result.upper())

        
        # Keep meaningful original words that Gemini omitted.
        original = re.findall(r"[a-zA-Z']+", text.upper())

        for word in original:
            if word.lower() in STOPWORDS:
                continue

            # Skip if the complete word already exists.
            if word in generated:
                continue

            # Skip if Gemini already returned its letters in sequence.
            letters = list(word)
            already_spelled = any(
                generated[i:i + len(letters)] == letters
                for i in range(len(generated) - len(letters) + 1)
            )

            if not already_spelled:
                generated.append(word)

        return generated


    except Exception as e:
        print("Gemini gloss error:", e)

        return [
            word.upper()
            for word in re.findall(r"[a-zA-Z']+", text)
            if word.lower() not in STOPWORDS
        ]
