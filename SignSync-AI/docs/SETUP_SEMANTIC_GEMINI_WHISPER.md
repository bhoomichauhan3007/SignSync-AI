# Setup: Whisper speech, Gemini answers and semantic search (run on your computer)

Run every command in VS Code's terminal with the `.venv` active. Start in the project folder unless a step says otherwise.
Check progress at any time with:  python scripts/check_features.py   (add --live for real calls)

## 1. Whisper speech to text (Member 3)
Whisper and ffmpeg are installed. Check with the script above; the Whisper line should show OK.
- The first time you record, Whisper downloads its "base" model (about 140 MB). Let it finish.
- After installing ffmpeg, close and reopen VS Code so the terminal can find it.
- Test: open Text & Voice, press **Record (Whisper)**, say "I need water, please", press again to stop.
  The words should appear in the box, then the signs.

## 2. Gemini answers for the tutor and sentences (Members 3 and 4)
1. Get a free key from Google AI Studio.
2. Open backend\.env and set these lines (paste your own key; keep the file private):
   GEMINI_API_KEY=your-key-here
   RAG_LLM_PROVIDER=gemini
   GEMINI_MODEL=gemini-2.5-flash
3. Install the Gemini package used by Member 3:
   pip install google-genai python-dotenv
4. Restart the backend (stop it with Ctrl+C, then start it again).
5. Test in the terminal (from backend\):
   python -m rag.scripts.query_cli "How can I learn Indian Sign Language?"
   The result should show "mode: llm".
   If it shows an error about quota or the key, check the key and try again later (free limits apply).

Free-tier limits are set by Google and change over time.

## 3. Semantic search for the tutor (Member 4)
This lets the tutor find answers by meaning, not only by shared words. It downloads one model (about 90 MB) the first time.
1. From backend\:
   pip install -r rag\requirements-rag.txt
   (This installs sentence-transformers, which pulls in PyTorch, so it is large. Let it finish.)
2. Rebuild the tutor index so it uses the new model:
   python -m rag.scripts.build_index
   The output should say "Embedder: sentence-transformers".
3. Check which answers are found and their scores:
   python -m rag.scripts.query_cli --retrieve-only "How do I start learning ISL?"
   Relevant passages should have higher scores. If good answers are refused, lower RAG_MIN_SCORE in backend\.env
   (try 0.25). If wrong passages appear, raise it (try 0.45). Then rebuild is NOT needed; only restart the backend.
4. Measure it (uses rag\evaluation\questions.json):
   python -m rag.scripts.evaluate --output rag\evaluation\results_semantic.json
   The printed source_hit_rate and refusal_accuracy are the real numbers for this computer.

## 4. Final check
From the project folder:
   python scripts/check_features.py --live
Every line should show OK. Send me the output if anything shows "--".

## Notes
- The knowledge base still holds only the sample document and the general ISL reference (unverified).
  Add verified ISL material to backend\rag\knowledge_base\verified\ and rebuild the index (step 3.2).
- Gemini answers are only as good as the documents they are given. The tutor still refuses when nothing relevant is found.
- Never share backend\.env or your key.
