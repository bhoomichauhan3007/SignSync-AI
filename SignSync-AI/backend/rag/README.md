# SignSync AI - RAG + LLM Tutor (Member 4)

The tutor answers questions about Indian Sign Language from a knowledge base of documents.
It only answers from retrieved passages, cites them, and refuses when the knowledge base has no answer.

Entry point used by the backend: `rag.assistant.ask(question, history) -> {"answer", "sources", "mode"}`.
Backend route: `POST /assistant/ask` (Member 5), which calls this through `app/services/ai_adapter.py`.

## 1. Install (run in the backend folder)

    cd backend
    python -m venv .venv
    # Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt            # Member 5's backend packages
    pip install -r rag/requirements-rag.txt    # numpy + pypdf (required); sentence-transformers + faiss-cpu (optional)

Minimum for the RAG module: numpy and pypdf. Without sentence-transformers the module uses the
lexical hashing embedder; without faiss-cpu it uses exact numpy search. Both still work.

## 2. Configure (no key needed to start)

    cp rag/.env.example rag/.env

Leave `RAG_LLM_PROVIDER=extractive` to run with no LLM and no API key. To use Gemini later, see section 7.
Never put a real key in code or commit `rag/.env`.

## 3. Add knowledge-base documents

Put files in `rag/knowledge_base/`:
- `verified/` checked ISL reference material
- `sample/` test material (already contains `sample_project_info.md`)
- `unverified/` anything not yet checked

Supported: `.pdf`, `.txt`, `.md`. Record every file in `rag/knowledge_base/SOURCES.md` with origin and licence.

## 4. Build the index

    python -m rag.scripts.build_index                                    # auto: semantic if installed
    python -m rag.scripts.build_index --embedder hashing --vector-store numpy   # offline, no downloads

Rebuild after any document change. Output goes to `rag/index/`.

## 5. Test retrieval and ask in the terminal

    python -m rag.scripts.query_cli --retrieve-only "What is the emergency number in India?"
    python -m rag.scripts.query_cli "How can I learn Indian Sign Language?"
    python -m rag.scripts.query_cli                  # interactive, type quit to exit

`--retrieve-only` prints each passage with its score. Use it to pick `RAG_MIN_SCORE`.

## 6. Run the backend and test `/assistant/ask`

Set `AI_MODE=real` in `backend/.env` (copy from `.env.example`), then:

    python -m uvicorn app.main:app --reload --port 8000

Open http://localhost:8000/docs. Check `GET /health/ai`: `ask_assistant` should show `real`.
Log in with `POST /auth/login` (admin login is in `.env`), copy `access_token`, then call
`POST /assistant/ask` with body `{"question": "How can I learn Indian Sign Language?"}` and header
`Authorization: Bearer <token>`. The response contains `answer` and `sources`.

## 7. Enable an LLM (optional)

Gemini (free tier, quotas set by Google and subject to change; check Google AI Studio):
1. Create a key in Google AI Studio.
2. In `rag/.env` set `GEMINI_API_KEY=<your key>` and `RAG_LLM_PROVIDER=gemini`.
3. If the model name is retired, change `GEMINI_MODEL`.

Ollama (local, no key): install Ollama, run `ollama pull llama3.2` and `ollama serve`, then set
`RAG_LLM_PROVIDER=ollama`.

If the LLM fails (quota, network, bad key), the tutor shows the retrieved passages and says why.

## 8. Frontend (Member 6)

See `../frontend_contract/AssistantPanel.jsx` for a complete React component that calls `/assistant/ask`
and shows the answer and sources. Member 6 copies it into `src/components/`.

## 9. Tests

    python -m unittest discover -s rag/tests -t . -v

or with pytest if installed: `python -m pytest rag/tests -v`

### Test status (what was actually executed)

Executed in the build environment (Python 3.12, numpy and pypdf available, no network):
- 57 unit and integration tests PASS, 2 SKIPPED.
- Covered: cleaning, chunking (sizes, overlap, stable IDs), PDF loading with page numbers (generated in-memory PDF),
  status-by-folder, hashing embeddings, numpy vector store save/load, de-duplication, retrieval threshold and top-k,
  refusal without LLM call, LLM prompt contents, LLM-failure fallback, Gemini and Ollama provider logic
  (HTTP mocked, no network), missing-index message, the `ask(question, history)` signature, and an end-to-end run
  over the real `knowledge_base/` folder.
- Skipped here: sentence-transformers test and FAISS test (packages not installed in the build environment).

NOT executed here (run locally):
- FastAPI route `/assistant/ask` through uvicorn (FastAPI is not installed in the build environment).
- sentence-transformers / all-MiniLM-L6-v2 embeddings and FAISS index (packages not installed; model download needs internet).
- Live Gemini or Ollama calls (no API key and no network in the build environment).
- Frontend component in a running React app.

## 10. Troubleshooting

| Message | Fix |
|---|---|
| "knowledge base has not been indexed" | run `python -m rag.scripts.build_index` |
| "No supported documents" | add .pdf/.txt/.md files to `rag/knowledge_base/` |
| "scanned PDFs need OCR" | the PDF has only images; use OCR first or supply a text version |
| Every question refused | lower `RAG_MIN_SCORE` after checking scores with `--retrieve-only` |
| Gemini quota message | wait, or set `RAG_LLM_PROVIDER=extractive` |
| index built with FAISS, faiss not installed | install faiss-cpu or rebuild with `--vector-store numpy` |
