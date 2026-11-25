# Live Sign Translator: how it works and how to test it

Page: frontend/src/LiveTranslator.jsx. Text and voice page: frontend/src/TextVoice.jsx.

## How recognition works

Letters (A to Z), in the browser:
1. The camera frame goes to MediaPipe Hands (lighter model, the same hand model used for Member 1's alphabet images).
2. The 21 hand points (x, y, z) are passed to Member 2's alphabet model, which runs in the browser from
   frontend/public/models/alphabet.json. No server call is made, so letters appear without a network wait.
3. Letters are smoothed: a letter is added only when at least 5 of the last 7 frames agree. After a letter is added,
   the same letter is not added again until the hand has been lowered.

Words (40), on the server:
1. MediaPipe Holistic records 30 frames (about two seconds) of pose, face and both hands: 1662 values per frame.
   The layout is Member 1's (computer_vision/holistic_features.py).
2. The frames go to POST /predict/sign (mode "dynamic"). The server runs Member 2's word_gru.keras (89.2% on their test set), after the same normalisation used in training. This needs TensorFlow.

Sentences:
- Finished words and letters are sent to POST /translate/signs-to-sentence. Member 3's engine forms the sentence
  (Gemini when a key is set, the built-in rules otherwise). The page shows which one was used.

Speech:
- "Speak" first tries the server voice (Member 3's gTTS, when installed), then the browser's built-in voice.
- Voice typing (Text & Voice page) uses the browser's speech recognition. It works in Chrome and Edge.

Only hand and body points are sent to the server. The video never leaves the computer.

## Tests done here

- Browser letter model: 400 of 400 labels match the Python model, confidence difference about 1e-15 (frontend/tests).
- Letter smoothing, hand-gone reset and repeat rules (frontend/tests/liveLogic.test.cjs).
- Holistic layout: 1662 values, correct offsets (same test file).
- Backend word input checks (30 frames of 1662 values, rejects wrong sizes and NaN).
- TypeScript syntax check for every frontend file.

## NOT tested here (do these on your computer)

1. Start the backend and frontend, sign in, open Live Translator, click Start camera.
2. Letters: sign A with your right hand. It should be added after about a second of steady signing.
   Note which letters are wrong and send the list. Live accuracy may be lower than the still-photo test score.
3. Words: install TensorFlow on the backend, then press Record a word and sign HELLO. Without TensorFlow the result
   is marked "Demo result".
4. Left and right hands: Member 2's word model was trained on MediaPipe's left and right hand outputs. If words only
   work with one hand, tell me which one.
5. Form sentence and Speak.
6. Text & Voice: type "I need water, please", then Translate to signs. Try Speak (voice typing needs Chrome or Edge).

## Known limits

- Letter accuracy in live use is not measured yet. The still-photo test score is 90.46%.
- Word recognition needs TensorFlow. Word models are not evaluated on held-out data yet.
- Sentences without a Gemini key use simple rules and can be weak.
- Sign video clips are not in the project. The app shows picture cards.
