import os, re, json, time
import numpy as np

LLM_MODEL = "claude-sonnet-5-5"
MIN_CONFIDENCE = 0.80
STABLE_FRAMES = 8
PAUSE_SECONDS = 2.0

try:
    import anthropic
    _client = anthropic.Anthropic() if os.getenv("ANTHROPIC_API_KEY") else None
except Exception:
    _client = None


def _llm(prompt, max_tokens=120):
    if _client is None:
        return None
    try:
        r = _client.messages.create(model=LLM_MODEL, max_tokens=max_tokens,
                                    messages=[{"role": "user", "content": prompt}])
        return r.content[0].text.strip()
    except Exception as e:
        print("LLM error:", e)
        return None


# ---------- 1. Prediction adapter (accepts any model output format) ----------
def normalize_prediction(output, labels):
    """Returns (LABEL, confidence) from: (label, conf) | label | index | probability vector."""
    if isinstance(output, tuple) and len(output) == 2:
        a, b = output
        return (a.upper(), float(b)) if isinstance(a, str) else (labels[int(a)].upper(), float(b))
    if isinstance(output, str):
        return output.upper(), 1.0
    arr = np.asarray(output)
    if arr.ndim == 0:
        return labels[int(arr)].upper(), 1.0
    idx = int(np.argmax(arr))
    return labels[idx].upper(), float(arr[idx])


# ---------- 2. Sign buffer: noisy predictions -> clean words ----------
class SignBuffer:
    """Feed predictions one at a time. Single letters (A-Z) are joined into a spelled word.
    Call add_none() when no hand is visible, so the same sign can be repeated later."""

    def __init__(self, min_conf=MIN_CONFIDENCE, stable=STABLE_FRAMES, pause=PAUSE_SECONDS):
        self.min_conf, self.stable, self.pause = min_conf, stable, pause
        self.reset()

    def reset(self):
        self.words, self.letters = [], []
        self.candidate, self.count, self.last_added = None, 0, None
        self.last_time = time.time()

    def _close_letters(self):
        if self.letters:
            self.words.append("".join(self.letters))
            self.letters = []

    def add_none(self):
        self.candidate, self.count, self.last_added = None, 0, None

    def add(self, label, confidence, now=None):
        now = time.time() if now is None else now
        if confidence < self.min_conf:
            return None
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = label, 1
        if self.count < self.stable or label == self.last_added:
            return None
        self.last_added, self.last_time = label, now
        if len(label) == 1 and label.isalpha():
            self.letters.append(label)
        else:
            self._close_letters()
            self.words.append(label)
        return label

    def is_done(self, now=None):
        now = time.time() if now is None else now
        return bool(self.words or self.letters) and (now - self.last_time) > self.pause

    def flush(self):
        self._close_letters()
        words = self.words[:]
        self.reset()
        return words


# ---------- 3. Words -> sentence ----------
def _rule_sentence(words):
    if not words:
        return ""
    s = " ".join(w.replace("_", " ").lower() for w in words)
    return s[0].upper() + s[1:] + "."


def words_to_sentence(words, language="English"):
    if not words:
        return ""
    prompt = f"""Convert this Indian Sign Language gloss into one short, natural {language} sentence.
Gloss (ISL order, keywords only, small vocabulary): {" ".join(words)}
Rules:
- Use the given words; add only small grammar words (articles, "is", "do", "please") when needed.
- Do NOT invent new facts or extra meaning. If it is ambiguous, choose the simplest reading.
- Use '?' only if the gloss is clearly a question.
- 'IloveYou' means "I love you". Spelled letters joined together (e.g. RAM) are a person's name.
- Output ONLY the sentence."""
    return _llm(prompt) or _rule_sentence(words)


# ---------- 4. Intent extraction ----------
INTENTS = ["greeting", "request", "question", "emergency", "thanks", "farewell", "statement"]


def _keyword_intent(sentence):
    s = sentence.lower()
    if any(w in s for w in ["help", "emergency", "doctor", "hospital", "police"]):
        return {"intent": "emergency", "entities": []}
    if any(w in s for w in ["hello", "namaste", "good morning", "good evening"]):
        return {"intent": "greeting", "entities": []}
    if any(w in s for w in ["bye", "goodbye"]):
        return {"intent": "farewell", "entities": []}
    if "thank" in s:
        return {"intent": "thanks", "entities": []}
    if s.strip().endswith("?"):
        return {"intent": "question", "entities": []}
    if "please" in s or "want" in s or "need" in s:
        return {"intent": "request", "entities": []}
    return {"intent": "statement", "entities": []}


def extract_intent(sentence):
    out = _llm(f'''Classify the sentence. Intents: {INTENTS}
Sentence: "{sentence}"
Return ONLY JSON: {{"intent": "<one intent>", "entities": ["<key nouns/objects>"]}}''', 100)
    if out:
        try:
            return json.loads(out.replace("```json", "").replace("```", "").strip())
        except Exception:
            pass
    return _keyword_intent(sentence)


# ---------- 5. Text -> signs ----------
STOPWORDS = {"a", "an", "the", "is", "am", "are", "was", "were", "be", "to", "of", "do", "does"}


def _rule_gloss(text):
    return [w.upper() for w in re.findall(r"[a-zA-Z']+", text.lower()) if w not in STOPWORDS]


def text_to_gloss(text, vocab):
    known = ", ".join(k for k in vocab if len(k) > 1)
    out = _llm(f'''Convert to Indian Sign Language gloss.
Sentence: "{text}"
Known signs: {known}
Rules: uppercase keywords only, no articles/auxiliary verbs; ISL order (subject-object-verb, question word last);
replace a word with a known sign ONLY when the meaning is the same (e.g. thank you -> THANKS, hi -> HELLO);
otherwise keep the original word. Return ONLY the words separated by spaces.''')
    return out.upper().split() if out else _rule_gloss(text)


def gloss_to_clips(gloss, vocab):
    seq = []
    for w in gloss:
        if w in vocab and len(w) > 1:
            seq.append({"type": "word", "gloss": w, "clip": vocab[w]})
        else:  # fingerspell unknown words
            seq += [{"type": "letter", "gloss": c, "clip": vocab.get(c, f"signs/alphabet/{c}.mp4")}
                    for c in w if c.isalpha()]
    return seq


def text_to_signs(text, vocab):
    gloss = text_to_gloss(text, vocab)
    return {"text": text, "gloss": gloss, "clips": gloss_to_clips(gloss, vocab)}


# ---------- 6. STT / TTS ----------
_whisper = None


def speech_to_text(audio_path, language=None):
    global _whisper
    import whisper
    if _whisper is None:
        _whisper = whisper.load_model("base")
    return _whisper.transcribe(audio_path, language=language)["text"].strip()


def text_to_audio_file(text, path="tts_out.mp3", lang="en"):
    from gtts import gTTS
    gTTS(text=text, lang=lang).save(path)
    return path


# ---------- 7. One-call helpers for other members ----------
def signs_to_speech_result(words, tts_lang="en"):
    sentence = words_to_sentence(words)
    return {"words": words, "sentence": sentence,
            "intent": extract_intent(sentence),
            "audio_path": text_to_audio_file(sentence, lang=tts_lang)}


def speech_to_signs_result(audio_path, vocab, language=None):
    return text_to_signs(speech_to_text(audio_path, language), vocab)
