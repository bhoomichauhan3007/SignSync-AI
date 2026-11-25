# SignSync AI - NLP Module (Member 3)

Setup: pip install anthropic gTTS openai-whisper numpy ; install ffmpeg ; set env var ANTHROPIC_API_KEY (optional, offline fallbacks exist)
Usage: from isl_nlp import *

| Function | Input | Output |
|---|---|---|
| normalize_prediction(model_output, labels) | label / index / probability vector | (LABEL, confidence) |
| SignBuffer().add(label, conf) / add_none() / is_done() / flush() | per-frame predictions | clean list of words |
| words_to_sentence(words) | ['HELLO','TEACHER'] | "Hello, teacher." |
| extract_intent(sentence) | sentence | {"intent":..., "entities":[...]} |
| text_to_signs(text, vocab) | text | {"gloss":[...], "clips":[...]} |
| speech_to_text(audio_path) | audio file | text |
| text_to_audio_file(text, path, lang) | text | mp3 path |
| signs_to_speech_result(words) | words | sentence + intent + audio path |
| speech_to_signs_result(audio, vocab) | audio file | text + gloss + clips |

For Member 4 (RAG/tutor): feed result['sentence'] and result['intent'] into the tutor;
pass the tutor's text answer to text_to_signs() and text_to_audio_file() to reply in signs/voice.
Vocabulary: 40 word classes + A-Z (see labels.json).
