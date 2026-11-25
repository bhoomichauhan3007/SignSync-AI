# Member 4 (RAG + LLM) - roadmap, viva points and resume notes

## 1. What is done
- Ingestion of PDF, TXT and Markdown with page numbers and status (verified, sample, unverified)
- Cleaning, chunking with overlap and stable IDs
- Embeddings: sentence-transformers (all-MiniLM-L6-v2) and a dependency-free hashing fallback
- Vector store: FAISS (IndexFlatIP) or exact numpy search, with metadata saved as JSON
- Retrieval with top-k, relevance threshold, de-duplication and source information
- Generation: grounded prompt with citations; Gemini and Ollama providers; extractive fallback
- Safety behaviour: refusal before LLM when nothing relevant, emergency number, sign-video limitation, provisional label
- Backend integration through the existing `rag.assistant.ask` adapter (no changes to Member 5's code)
- Tests (57 run, see TESTING_REPORT.md), API documentation, React component for Member 6

## 2. Remaining work for Member 4 (in priority order)
1. **Real ISL material.** Collect verified ISL learning material with written permission or an open licence,
   record it in `SOURCES.md`, and put it in `knowledge_base/verified/`. This matters more than any code change.
2. **Install and run the semantic path locally.** `pip install -r rag/requirements-rag.txt`, build the index with
   sentence-transformers and FAISS, then tune `RAG_MIN_SCORE` using `query_cli --retrieve-only`. Record the scores you see.
3. **Evaluation set.** A starter set exists (`rag/evaluation/questions.json`, 8 questions, sample document only). It is a smoke test. Write 20 to 30 questions with the expected source for each (for example, greetings, emergency phrases,
   learning tips, "what is the difference between two signs"). Measure whether the expected source is in the top 4 (recall@4)
   and how often the tutor refuses when it should answer. Save the results in the testing report. Do not report numbers you have not measured.
4. **Member 3 link.** Done in code: `rag/integration/nlp_bridge.py` routes `question` and `emergency` intents from
   `signs_to_speech_result` to the tutor. Still to decide with Member 3: whether the tutor reply should also go to their
   text-to-sign path and audio output.
5. **Conversation history.** Confirm that `conversation_id` history reaches the prompt in a two-turn test.
6. **Deployment.** Put the index build in the Dockerfile or a setup script, and document the model download.

## 3. Viva talking points
- Explain the whole flow in one sentence: question → embedding → nearest chunks → threshold → prompt with citations → answer with sources.
- Explain why the tutor refuses: a fluent wrong answer about sign language would be harmful, so the system prefers "I don't know".
- Explain the two fallbacks and why they exist (offline testing, quota failure, provider outage).
- Explain why RAG does not do sign recognition: that is Member 2's job; the tutor only answers questions.
- Show the terminal: `build_index` output with counts per status, then `query_cli --retrieve-only` with scores.
- Be honest about limits: small corpus, lexical fallback scores, prompt rules are not a guarantee.

## 4. Resume bullets (use only after you have run them locally)
- Built a retrieval-augmented tutor for Indian Sign Language: document ingestion (PDF/text), chunking with metadata, sentence-transformer embeddings, FAISS vector search, and grounded LLM answers with source citations.
- Designed the tutor to refuse when retrieval finds no relevant passage, and to label unverified material, to reduce unsupported answers.
- Integrated the RAG service into a FastAPI backend through a defined contract with mock fallback, with 57 unit and integration tests.
- Added a provider abstraction for Gemini and Ollama, with graceful fallback to retrieved passages when the LLM is unavailable.

## 5. Suggested improvements for a stronger project
- Hybrid search (keyword BM25 plus embeddings) and a cross-encoder reranker, measured against your evaluation set
- Multilingual embeddings if Hindi material is added
- Feedback button on answers, stored in the database, to collect real failure cases
- Caching of frequent questions and embeddings
- Automated evaluation in CI, so retrieval changes cannot silently break answers
