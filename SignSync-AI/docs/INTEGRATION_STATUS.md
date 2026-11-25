# Integration status (21 November 2026)

| Hook / feature | Owner | Status | Notes |
|---|---|---|---|
| predict_sign, letters (server) | Member 2 | real | alphabet_mlp.joblib (90.46% on the old held-out set) |
| letters in the browser | Member 2 and 1 | real | frontend/public/models/alphabet.json, same math as Python (checked) |
| predict_sign, words | Member 2 | real when TensorFlow is installed, otherwise demo | word_gru.keras (default; 89.2% test accuracy), normalisation and labels from Member 2's artifacts |
| signs_to_sentence | Member 3 | real when Gemini is configured, otherwise rules | member3_adapter.py; falls back if google-genai or the key is missing |
| text_to_sign_sequence | Member 3 | real | gloss from Gemini or rules, clips from sign_vocab.json |
| speech_to_text | Member 3 | mock until Whisper is installed | needs openai-whisper and ffmpeg |
| text_to_speech (server) | Member 3 | mock until gTTS is installed; browser voice used otherwise | needs gTTS and internet |
| ask_assistant | Member 4 | real | works with no key; knowledge base includes unverified ISL basics |
| compare_with_reference (letters) | Member 5 (heuristic) | real | backend/cv/feedback.py, compared with 30 real examples per letter |
| compare_with_reference (words) | Member 5 | fixed score 70 with note | word correctness comes from the word model |
| My Progress page | Member 6 | real data | /progress/me |

Changes this round
- Member 3's new NLP files merged; gemini_client.py changed so it no longer stops the import when the key is missing.
- backend/nlp/pipeline.py now uses Member 3's adapter with a fallback to the rule-based code.
- Live Translator rewritten: browser letter model, MediaPipe Hands for letters, Holistic for words, sentence forming, speech.
- New Text & Voice page; navigation updated.
- Word picture cards redrawn; alphabet gallery uses the team's photos.
- ISL basics reference added to the tutor (unverified).
- Member 2 word package integrated: GRU/LSTM/Transformer models, normalisation, labels (see docs/WORD_MODEL.md).
- Member 2 review: requirement fixed (TensorFlow 2.20+), live script rewritten, checker added (docs/MEMBER2_REVIEW.md).
- Hand-shape feedback added for letters (backend/cv/feedback.py). My Progress page connected to real data.
