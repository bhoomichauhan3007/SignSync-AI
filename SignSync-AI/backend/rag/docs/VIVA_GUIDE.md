# RAG Tutor - Viva Guide (Member 4)

Simple explanations for presenting and defending the module.

## Concepts in plain English

**RAG (Retrieval-Augmented Generation).** The tutor first searches our own documents for the
passages that match the question, then gives those passages to a language model and asks it to
answer only from them. "Retrieval" finds the facts, "generation" writes the reply.

**LLM (Large Language Model).** A model trained on large amounts of text that writes fluent answers.
On its own it can make things up. In our system it is only a writer: it is given the facts and told
to cite them.

**Embedding.** A list of numbers (a vector) that represents the meaning of a piece of text. Texts with
similar meaning get vectors that point in similar directions. We use all-MiniLM-L6-v2, which produces
384 numbers per text. We normalise each vector to length 1, so a dot product equals cosine similarity.

**Semantic search.** Finding text by meaning rather than exact words. The question and every document
chunk are embedded, and we return the chunks whose vectors are closest to the question's vector.

**FAISS (Facebook AI Similarity Search).** A library that stores many vectors and finds the nearest
ones quickly. We use IndexFlatIP: an exact search that ranks by inner product (cosine, because the
vectors are normalised). For our small knowledge base exact search is fast enough, and approximate
indexes are unnecessary.

**Why FAISS?** It is the standard open-source vector index, it runs locally with no server, and it
stores the index as a single file. A hosted database would add a server and cost for no benefit at
this scale. When FAISS is not installed we use numpy, which gives identical rankings.

**How retrieval works in our code.** Question -> embed -> take the top 16 candidates (4 x top_k) ->
drop those below the relevance threshold -> remove duplicates -> keep the top 4 -> return them with
title, page and score.

**How the LLM gives grounded answers.** The prompt has three parts: a system prompt with rules
("answer only from the passages, cite [S1], say when the answer is missing, never invent sign
meanings"), the numbered passages with their document names and pages, and the question. We also
refuse before calling the LLM when no passage passes the threshold.

**Normal chatbot vs our RAG tutor.** A normal chatbot answers from what the model learned during
training, which may be out of date or wrong, and it cannot show where an answer came from. Our tutor
answers from the documents we chose, shows its sources, labels material that is not verified, and says
"I don't know" instead of guessing.

## File-by-file

**rag/__init__.py** - Package marker and the one-line description of the public function.
Viva: "This package exposes ask(question, history) to the backend."

**rag/config.py** - Reads every setting from environment variables (RAG_*, GEMINI_*, OLLAMA_*) and
optional .env files. Secrets are never in code. Viva: "Changing the model or threshold needs no code change."

**rag/ingestion/loaders.py** - Reads PDF (pypdf, one document per page so page numbers are kept), TXT
and Markdown. Works out each file's status from its folder: verified, sample or unverified.
Why: students must see where information came from and whether it is checked.

**rag/ingestion/cleaning.py** - Normalises Unicode, joins words broken across lines, removes page
markers like "Page 3 of 10". Why: PDF text is messy, and clean text embeds better.

**rag/ingestion/chunking.py** - Splits text into chunks of about 900 characters at paragraph and
sentence boundaries, with 150 characters of overlap. Each chunk gets a stable ID (SHA-1 of its
content). Why: one small chunk = one topic, so retrieval is precise, and overlap prevents cut answers.

**rag/embeddings/embedder.py** - Two embedders with the same interface. SentenceTransformerEmbedder
(all-MiniLM-L6-v2, semantic) and HashingEmbedder (word-based fallback that runs offline). Both
normalise vectors. Viva: "The hashing one matches words, not meaning, so I use it only when the
semantic model is not installed."

**rag/vector_store/store.py** - Builds the index (FAISS or numpy), saves it with chunk metadata in
separate JSON, and reloads it. The manifest stores which embedder built the index, so queries are
embedded the same way. Viva: "If I change the embedder I must rebuild, and the manifest catches the mismatch."

**rag/retrieval/retriever.py** - Embeds the question, searches, applies the threshold, removes
duplicate or overlapping chunks, returns the top k with sources. Viva: "The threshold is what makes the
tutor say 'I don't know' instead of forcing a weak answer."

**rag/generation/prompts.py** - The system prompt with the rules that prevent unsupported claims,
the passage formatter that labels [S1], [S2] and keeps the source list in sync with the citations,
the refusal and safety texts, and the extractive (no-LLM) answer formatter.

**rag/generation/llm.py** - Provider abstraction. GeminiProvider and OllamaProvider use only the
standard library (urllib), so no extra package is needed. Errors are turned into readable messages,
and the API key is sent in a header and never printed. Viva: "I can switch provider with one setting."

**rag/assistant.py** - The public ask() function. Loads the index once, runs answer_question(), which
is a pure function that tests can drive with fake components. Handles emergencies (adds the 112
number for India), sign-video questions (adds a limitation note), refusals, LLM failure (falls back to
passages with the reason), and unverified sources (adds a provisional note).

**rag/scripts/build_index.py** - Command that reads the knowledge base, chunks it, embeds it and saves
the index. Prints counts per status so you can show what was indexed.

**rag/scripts/query_cli.py** - Terminal client for asking questions or viewing raw retrieval scores.

**rag/knowledge_base/** - The documents. SOURCES.md records origin and licence of each file. The
sample file is labelled as test material and contains no sign meanings.

**rag/tests/** - Unit and integration tests, runnable offline.

**backend/app/services/ai_adapter.py (Member 5, not changed)** - Imports rag.assistant.ask when
AI_MODE=real and calls it with (question, history). Our module needed no change to their code.

## Likely viva questions

- Why not just use ChatGPT? Answers must come from our documents, with sources, and the tutor must refuse when they do not contain the answer.
- What stops the model inventing sign meanings? Three layers: the prompt forbids it, the refusal runs before the LLM when retrieval is empty, and the knowledge base contains no invented sign descriptions.
- What if the LLM is down or over quota? The student still gets the matched passages, with the reason.
- How do you know retrieval works? The retrieve-only command and the unit tests check ranking, thresholds and de-duplication.
- Why is the hashing embedder in the code? To run offline and in tests. It is not the production model.
- What would you improve? Add a cross-encoder reranker, evaluate with a labelled question set, and add more verified ISL material.

## Limitations to state honestly

- The sample knowledge base is tiny, so retrieval is coarse.
- Hashing scores are low and only match shared words. Thresholds must be tuned after installing the semantic model.
- Prompt rules reduce unsupported claims but cannot guarantee them. The code checks citations and logs a warning when they are missing.
- The tutor cannot see video, so it cannot judge a student's signing.
