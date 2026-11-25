# Word recognition (Member 2): what was built, in simple words

## What the word model does

You sign a whole word for about two seconds. The camera records 30 small snapshots of your body, face and hands.
Each snapshot has 1662 numbers (the positions of your body points, Member 1's layout). The word model looks at all
30 snapshots together and guesses which of the 40 words you signed.

## The three models

Member 2 trained three kinds of neural network on the same data:

| Model | What it is | Test accuracy | Macro F1 (balanced score) |
|---|---|---|---|
| **GRU** (used by the app) | a network that reads the snapshots in order | **89.2%** | **0.890** |
| LSTM | another order-reading network | 82.9% | 0.828 |
| Transformer | a network that compares all snapshots with each other | 83.3% | 0.825 |

The GRU is the default because it scored best on the test set.

**Honest note:** the test set has 240 recordings, which is 6 per word. That is small. One or two mistakes change a
word's score a lot. Words such as "good", "night" and "please" were weaker for some models. Treat 89% as a
measurement on a small test, not a guarantee for every person and every camera.

## How the app uses it

1. The Live Translator page records 30 frames.
2. Each number is adjusted with the same average and spread used in training (normalisation).
3. The GRU model gives a score for each of the 40 words.
4. The word with the highest score is shown, with its confidence. Low confidence asks you to sign again.
5. The word is shown with its friendly text, for example "I love you".

## Files (sign_recognition/word/)

- artifacts/word_gru.keras, word_lstm.keras, word_transformer.keras: the trained models
- artifacts/normalization.npz: the average and spread used for adjusting numbers
- artifacts/label_encoder.json: the 40 word names, in order
- artifacts/*_metrics.json: the full test results (per word, confusion matrix)
- train_word_models.py, models.py, dataset_utils.py: training code (needs TensorFlow)
- predict_word.py: predict from a saved sequence file (command line)
- word_model_api.py: a standalone web API from Member 2 (optional, not needed by the app)
- member2_handoff.md: Member 2's notes

## What is needed to run it

- TensorFlow installed on the backend: `pip install tensorflow` (see requirements-optional.txt).
- The three model files and the two artifact files (already in the project).

Without TensorFlow, the page still runs and shows "Demo result" for words.

## What is NOT included

- The training data (word_dataset.npz, about 209 MB). It is only needed to retrain. Member 2 keeps it.
- The 41st class "i" has no recordings in the data, so the model does not know it. The app uses only the 40 words.

## Known issues found in Member 2's package

- live_word_recognition.py imports a file that does not exist, so it cannot run. Live use goes through the web page.
  The file is kept, with a note at the top.
- The training data lists 41 classes but only 40 have recordings. The label file has the 40 that were trained.
