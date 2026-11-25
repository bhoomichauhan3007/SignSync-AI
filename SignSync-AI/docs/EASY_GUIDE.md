# SignSync AI: the whole project in simple words

## 1. What is SignSync AI?

SignSync AI is an app that helps people talk with Indian Sign Language (ISL). It can:

- read letters and words that you sign to a camera,
- make a sentence from the words,
- turn typed or spoken sentences into signs,
- read sentences aloud,
- answer questions about ISL with an AI tutor.

The project has six parts, one for each team member. Each part is a folder in the project.

## 2. The six parts, in simple words

| Member | What it does | Folder |
|---|---|---|
| 1. Camera and data | Finds the hand and body points in camera images, and prepares the data for training | computer_vision, data |
| 2. Sign recognition | Trained models that guess the letter or word from the hand points | sign_recognition |
| 3. Language and speech | Makes sentences, turns text into signs, listens to speech, speaks text | nlp_translation |
| 4. AI tutor | Answers questions about ISL from a set of documents | backend/rag |
| 5. Backend | The server that stores users and connects all parts | backend |
| 6. Frontend | The web pages you see and click | frontend |

## 3. How a sign becomes a sentence (the main flow)

1. The camera sees your hand.
2. The browser finds 21 points on your hand (thumb, fingers, wrist).
3. Those points go into the letter model. The model says "this looks like A" and how sure it is.
4. The app waits until most of the last frames agree, so one wrong frame does not add a wrong letter.
5. Letters join into a word. Words join into a sentence.
6. Member 3's sentence engine makes the sentence sound natural.
7. You can read it aloud.

For words, the app records about two seconds of your body and face, and the word model guesses the whole word.

## 4. The other direction: text to signs

You type or speak a sentence. The app shows each sign as a picture, one after another. Words it does not know are
spelled letter by letter.

## 5. The AI tutor, explained for a beginner

The tutor is a question-answering assistant. It does not know everything. It only answers from documents we give it.
This is important: a tutor that makes up sign meanings would teach people wrong signs.

### The words you need to know

**Document.** A text file or PDF with information, for example a page about ISL basics.

**Chunk.** A document is cut into small pieces, each about a paragraph long. Small pieces are easier to search.

**Embedding.** A list of numbers that describes what a piece of text means. Texts with similar meaning get similar
numbers. Example: "start learning" and "beginner advice" end up with similar numbers.

**Vector database (index).** A place where all the numbers for all the chunks are stored, so we can search them fast.
FAISS is a library that does this search. Our fallback is a simple numpy search that gives the same results for this size.

**Semantic search.** Searching by meaning, not by exact words. You ask "How do I start?", and the tutor finds the
paragraph about beginners even though the words are different. This needs the sentence-transformers model
(all-MiniLM-L6-v2). Without it, the tutor only matches shared words.

**Retrieval.** Finding the most useful chunks for the question.

**Threshold.** A minimum score. If no chunk is similar enough, the tutor says it does not know. It is better to say
"I don't know" than to guess.

**LLM (large language model).** A program that writes fluent text. Gemini is the LLM we use. Google runs it. We send
it the question and the chunks we found, and it writes an answer.

**RAG (retrieval-augmented generation).** The whole method: first retrieve the right chunks, then let the LLM write an
answer using only those chunks. "Retrieval" finds the facts. "Generation" writes the answer.

### The difference between a normal chatbot and our tutor

| Normal chatbot | Our tutor (RAG) |
|---|---|
| Answers from what it learned long ago | Answers from documents we chose |
| Cannot show where an answer came from | Shows the document and page it used |
| May invent facts when unsure | Says "I don't know" when the documents have no answer |
| Hard to check | Easy to check: open the source |

### How a question is answered, step by step

1. You ask: "How do I start learning ISL?"
2. The question is turned into numbers (embedding).
3. The tutor finds the chunks whose numbers are closest (search).
4. If the closest chunk is not close enough, the tutor says "I don't know" and stops.
5. Otherwise, the chunks go to Gemini with the rule: answer only from these passages, and say which passage you used.
6. The answer is shown with its sources.

If there is no Gemini key, the tutor shows the matching passages directly. This still works, just without writing.

### The three ways the tutor can answer

- **Gemini (when GEMINI_API_KEY is set):** Gemini writes a short answer from the passages.
- **Passages only (no key):** the tutor shows the most relevant passages, marked as passages.
- **Refusal:** no passage is relevant, so the tutor says it does not have the answer.

## 6. Member 4 setup in four steps

1. **Semantic search.** Install the packages: `pip install sentence-transformers faiss-cpu`.
   Then rebuild the index: `python -m rag.scripts.build_index`. The first run downloads the model (internet needed).
2. **Gemini answers.** Put your key in `backend/.env` as `GEMINI_API_KEY=your-key`. Keep `RAG_LLM_PROVIDER=auto`.
   The tutor uses Gemini automatically when the key is there.
3. **Check it.** Run `python -m scripts.check_setup`. It shows READY or NOT READY for each part, and what to do.
4. **Test a question.** `python -m rag.scripts.query_cli "How do I start learning ISL?"`.

If semantic search refuses questions it should answer, lower `RAG_MIN_SCORE` a little, after looking at the scores with
`python -m rag.scripts.query_cli --retrieve-only "your question"`.

## 7. Member 3 setup in four steps (speech, sentences, text to sign)

1. **Whisper (speech to text):** `pip install openai-whisper`. You also need ffmpeg installed, which you have done.
2. **Gemini sentences:** the same key as in step 6.2. Install `pip install google-genai python-dotenv`.
3. **Server voice (optional):** `pip install gTTS`. The app uses the browser voice when this is missing.
4. **Check it.** `python -m scripts.check_setup`.

## 8. Where things stand (in short)

- Working on the computer where we built it: the whole app, the letter model, the word-sign path (needs TensorFlow),
  the tutor with passages, the Text and Voice page, sign-in, and the sign gallery.
- Needs your computer to confirm: the browser camera and MediaPipe, Whisper speech, Gemini answers, semantic search,
  word recognition with TensorFlow.
- Still missing in the project: verified ISL content from a teacher, sign video clips, and retraining the letter model
  on the cleaned images.

Full details are in docs/WORK_STATUS.md.

## 9. Simple answers for a viva

- **What is the project?** An app to help people learn and use Indian Sign Language, with an AI that reads signs, forms sentences and answers questions.
- **What is RAG?** A method where the AI first searches our own documents, then writes the answer from what it found.
- **What is an embedding?** A list of numbers that stands for the meaning of a text.
- **Why FAISS?** It searches many embeddings quickly.
- **Why does the tutor sometimes say "I don't know"?** On purpose. It answers only from documents, and it refuses when they do not cover the question.
- **Why use a model in the browser?** So letters appear instantly, without waiting for the server.
- **What is the biggest limit?** The data: we need verified ISL material and sign videos from teachers.

## 10. Common problems

| What you see | What to do |
|---|---|
| "Gemini is not configured" | Add GEMINI_API_KEY to backend/.env and restart the backend |
| The tutor says "I could not find this" for a fair question | Rebuild the index after installing sentence-transformers; check scores with --retrieve-only |
| Speech to text does not work | Check `python -m scripts.check_setup` for Whisper and ffmpeg; restart the terminal after installing ffmpeg |
| Word signs say "Demo result" | Install TensorFlow on the backend |
| The camera does not start | Allow the camera in the browser, close other apps using it, then try again |
