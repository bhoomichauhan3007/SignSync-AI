# SignSync AI: what is implemented, and what is still to do (by member, in order)

Status date: 21 November 2026. "Implemented" means the code is in the project and was tested as stated.
"Not tested here" means it needs your computer (browser, TensorFlow, Whisper, MediaPipe, API key).

## Member 1: Computer vision (camera, landmarks, datasets)

Update (fifth round): the external video word dataset ("new" folder) is integrated as code and documentation
(computer_vision/video_pipeline/) with a checker. The data itself is in a separate RESEARCH ONLY ZIP and is NOT merged into
the word model. Checker results: covers 9 of 40 SignSync words; 73% of clips still have pose points outside the picture after
masking low-visibility points; 38 of 98 test clips share a session with training. Details: docs/EXTERNAL_VIDEO_DATASET.md.
The project's own recorder data (Member 2's MP_Data) may have the same pose issue and should be checked the same way.

Update (third round, Member 1's signsync_cv_changes.zip, integrated and checked):
- One feature schema for every script (computer_vision/feature_schema.py, holistic-1662-v1).
- Recorder crash fixed: save_sample now gets its six arguments at every call (checked in the code).
- Left and right hands: recorder, live predictor and hand tracking now follow one setting (MIRROR_INPUT_FOR_HOLISTIC = False).
- Image extraction: Hands-model fallback for close-up hand images (code present, not run here: needs MediaPipe 0.10.21).
- New export cv_word_dataset.npz stores signer_ids and sample_ids, so future recordings can be split by person.
- Member 1's logic tests pass inside the project (8 of 8). Their live predictor was not run with a real model.

Implemented
- Landmark recorder and alphabet extractor (computer_vision/landmark_extraction.py, unchanged from the upload).
- Holistic feature layout, 1662 values per frame (computer_vision/holistic_features.py), used by the browser too.
- Export, preprocessing and audit scripts, updated (export_features.py, preprocessing.py, dataset_audit.py).
- Word export fixed so it reports classes without sequences instead of stopping (fix_word_export.py).
- Dataset cleaned: 19,784 alphabet images, no image in two splits (docs/DATASET_REPORT.md).
- Browser hand tracking for letters (MediaPipe Hands) and body tracking for words (MediaPipe Holistic) in the Live Translator.

Still to do
- Record new word samples with the holistic recorder (option 2) so the export has signer IDs. Old word data has none.
- Re-extract alphabet landmarks from the cleaned images on a machine with MediaPipe 0.10.21 (landmark_extraction.py,
  option 1), then export. This is needed to retrain the letter model on the new images.
- Test live_predictor.py with a real trained model (needs TensorFlow and a camera).
- Run verify_schema_compat.py on your computer after recording (the ML side was checked here: layout matches, values are raw, not double-normalised).
- Merge new samples with the team's word data by class NAME (not class index).
- Confirm with Member 2 which left/right convention MP_Data used (recorder is unmirrored; not verifiable from the files).
- Confirm the Training split. It covers letters A to R only; S to Z exist only in Validation.
- Confirm the origin of the 1662-value word data (the layout matches Member 2's training data, but the original recorder is not in any upload).
- Check the left and right hand behaviour for words (see docs/LIVE_TRANSLATOR.md).

## Member 2: Sign recognition (models)

Implemented
- Letter model (scaler + MLP, 26 letters). Measured 90.46% on 4,633 held-out images (before the dataset cleaning).
- The same letter model runs in the browser from alphabet.json. Browser output matches scikit-learn on the test samples checked.
- Word model (new package, this round): three trained networks (GRU, LSTM, Transformer) for 40 words.
  The app uses the GRU: 89.2% test accuracy, macro F1 0.890 (240 test recordings). Details: docs/WORD_MODEL.md.
- The backend applies the same steps as Member 2's code: normalise with the saved average and spread, then classify.
- Labels are checked against the sign vocabulary by tests (they match exactly).
- The old word models (trained without normalisation, on an older export) were removed from the project.

Tested here: backend tests check the labels, the normalisation shape and the file paths. The model itself was NOT run here
(TensorFlow is not installed in the build sandbox).

Review (done this round): see docs/MEMBER2_REVIEW.md. Fixed: TensorFlow requirement (2.20+, Keras 3), broken live script (rewritten), artifact checker added (python check_word_artifacts.py, passes).

Still to do
- Install TensorFlow on the backend and test word recognition with a real signed word (not tested here).
- Re-split the data by person (needs person IDs from Member 1), re-train, and re-measure. The 89.2% may be optimistic (random split).
- Mask missing frames before normalising, then retrain (design flaw, see review).
- Record more test samples per word (240 is a small test; one word changes the score by about 17 points).
- Test the live script on a laptop with a camera (not tested here).
- Retrain the letter model on the cleaned images after Member 1 re-extracts the landmarks.
- Optional: fix live_word_recognition.py (it cannot run as shipped; the web app replaces it).
- Optional: the training data (word_dataset.npz, 209 MB) is kept by Member 2, not in the project.

## Member 3: NLP and translation (sentences, text to sign, speech)

Update: Whisper and ffmpeg were installed on the team computer by the team (not verified by the build team). Check with `python -m scripts.check_setup`.

Implemented
- Gemini-based sentence formation and gloss extraction (nlp_translation/gemini_nlp.py), with the rule-based fallback
  when no key is set.
- Text to sign mapping with the sign vocabulary (sign_vocab.json), with fingerspelling for unknown words.
- Speech to text (Whisper) and text to speech (gTTS) adapter functions (member3_adapter.py).
- Whisper recording in Text & Voice: record, transcribe, then turn into signs (uses /translate/speech-to-sign). Whisper and ffmpeg are installed on the team's machine; the recording path has not been run in a browser here.
- Backend routes: /translate/signs-to-sentence, /translate/text-to-sign, /translate/speech-to-sign, /translate/text-to-speech.
- Browser speech: "Speak" reads the sentence aloud with the browser's voice (no installation). Voice typing works in Chrome and Edge.
- Live Translator "Form sentence" uses Member 3's sentence engine.
- Text & Voice page: type or speak a sentence and see each sign as a picture.

Tested here: the backend integration tests pass with the rule-based fallback. The Gemini path and Whisper/gTTS were not run.

Still to do
- Gemini: set GEMINI_API_KEY in backend/.env and install google-genai (docs/SETUP_SEMANTIC_GEMINI_WHISPER.md).
- Speech to text: Whisper and ffmpeg installed on your machine. Test the Record (Whisper) button (see docs/SETUP_SEMANTIC_GEMINI_WHISPER.md).
- Text to speech on the server: install gTTS (needs internet).
- Sign video clips: the vocabulary refers to clips (signs/*.mp4). No clip file exists in any upload, so the app shows
  picture cards. Someone must record or source the clips, or the team keeps the picture cards.
- Sentence quality without Gemini is weak ("Water want.").
- Member 2 to Member 3 hook (transformer_prediction_to_sentence) is written but not connected to a live route.

## Member 4: AI tutor (RAG and LLM)

Update (22 November): Gemini answers now display as cited passages with source cards and a single status banner (no duplicate notes, no raw [S1] markers).

Update (third round, done in code and tested with stand-ins): the semantic search path (sentence-transformers and FAISS) builds, saves, reloads and answers; the Gemini path answers from the passages and cites them. Tests: 67 run, 65 pass, 2 skipped. The real packages and the real Gemini call still need your computer.

Update: the tutor now uses Gemini automatically when GEMINI_API_KEY is set (RAG_LLM_PROVIDER=auto). Semantic search is used when sentence-transformers is installed and the index is rebuilt. Both were written and tested in code paths that run without those packages; the semantic path is not tested here.

Implemented
- Knowledge base, ingestion, embeddings, vector search, retrieval with a relevance threshold, grounded answers, sources.
- Refusal when the knowledge base has no answer (no LLM call).
- General ISL reference added (unverified): how signs are formed, fingerspelling, learning advice, the app vocabulary list.
- Member 3 bridge: question and emergency intents go to the tutor.

Tested here: 62 tutor tests (60 pass, 2 skipped). The tutor answers the ISL basics questions and refuses off-topic ones.

Still to do
- Verified ISL material from a qualified teacher. The current ISL content is general and unverified, and answers are
  extractive (they quote the document) unless Gemini is set up.
- Semantic search: install sentence-transformers (docs/SETUP_SEMANTIC_GEMINI_WHISPER.md), rebuild the index, tune RAG_MIN_SCORE, run the evaluation.
- The tutor shows how each answer was produced: written by Gemini, taken from the knowledge base, or not found.
- Gemini answers: set the key and RAG_LLM_PROVIDER=gemini (not tested here).
- Re-run the evaluation with verified material (rag/evaluation/questions.json).

## Member 5: Backend (API, auth, progress, database)

Implemented
- Login and sign-up with a profile (new user_profiles table, created automatically).
- Hand-shape feedback for letters (cv/feedback.py): compares the learner's hand shape with 30 real examples of the letter. On 650 test images the correct letter was closer than the closest wrong letter 82% of the time; a correct letter scores a median of 94. Words get a fixed score of 70 and a note (word correctness comes from the word model).
- Prediction routes for letters and words, the AI adapter with fallback to the mock on failure.
- Translate, learn, progress and conversation routes (from the upload).

Tested here: the integration tests pass; the Member 5 API test suite was not run here (needs FastAPI and pytest).

Still to do
- Run the full API test suite on your machine: python -m pytest tests -q (in backend/).
- Shape feedback for words (currently a fixed score of 70 with a note). A real word-level score needs a design decision.
- Shape feedback is a heuristic: it is not a teacher. Check it with real learners before relying on it.
- Change SECRET_KEY and the admin password before any shared deployment.
- Decide on PostgreSQL (docker-compose) or SQLite for the demo.

## Member 6: Frontend (web app)

Implemented
- Sign-in and create-account (two steps, with profile questions and a privacy agreement). Session is kept across reloads.
- Live Translator: letters read in the browser with the trained model, hand or body skeleton overlay, letter smoothing,
  word recording (words go to the server), sentence builder, sentence forming (Member 3), speech.
- Text & Voice: type or speak, see signs, read aloud.
- Learn ISL: sign gallery with the 26 letter photos and 40 word cards (click a card to hear the word).
- AI Tutor connected to the backend with sources.
- My Progress: real XP, level, streaks, badges, categories and lessons from the backend (empty state for new learners).
- Styling for all pages (gradients, cards, focus states, responsive layout).

Tested here: JavaScript tests for the browser model (400/400 labels match Python), letter smoothing, and the holistic
layout; TypeScript syntax check passes for all files.

Not tested here (needs a browser)
- The Live Translator, the camera, and the MediaPipe models (load from the internet).
- The production build (npm run build). The lucide-react version in package.json is 1.x; if a build error names an icon, tell me.

Still to do
- Run the app in a browser and check the Live Translator with real signs (see docs/LIVE_TRANSLATOR.md).
- Word photos: the word tiles are picture cards until photos are added to frontend/public/signs/words/.
- Check My Progress with a learner account that has attempts (not tested here).
- Mobile layout check.

## Cross-cutting: what must happen before a final demo

1. Run ./scripts/run_tests.sh and python -m pytest tests -q (backend).
2. Start the backend and frontend, sign in, and try the Live Translator and Text & Voice.
3. Decide the LLM provider (Gemini key, or the built-in rules).
4. Decide who records the sign video clips, or keep the picture cards.
5. Add verified ISL material to backend/rag/knowledge_base/verified/.
