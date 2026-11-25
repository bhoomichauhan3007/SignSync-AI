# SignSync AI: frontend (React + Vite)

Pages: Overview, Live Translator, Text & Voice, Learn ISL (sign gallery), AI Tutor, My Progress.
Sign in or create an account first.

## Run

    cd frontend
    copy .env.example .env        (Windows)   or   cp .env.example .env   (Mac/Linux)
    npm install
    npm run dev

Open http://localhost:5173. The backend must be running on http://localhost:8000.

## Tests (no browser needed)

    node tests/alphabetModel.test.cjs
    node tests/liveLogic.test.cjs

## Files you may change

- src/App.jsx: pages and navigation
- src/LiveTranslator.jsx: live letter and word recognition
- src/TextVoice.jsx: text and voice to signs
- src/alphabetModel.js: the letter model running in the browser (weights in public/models/alphabet.json)
- src/speech.js: reading aloud and voice typing
- src/api.js: calls to the backend
- src/App.css: styling
- public/signs/: sign pictures (see public/signs/README.txt)
