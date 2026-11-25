# Member 2 word package: review (what was wrong, what was fixed, what is still open)

## Summary in plain words

Member 2's word models work as a package: the files load in code, the labels match the vocabulary, and the test numbers
were produced by their script. But there were four real problems. Two are fixed in the project. Two need Member 2 and
Member 1 to act, because they depend on data we do not have.

## Problems found

| # | Problem | Why it matters | Status |
|---|---|---|---|
| 1 | **Requirement too old.** The package asked for TensorFlow 2.15 or newer. The models were saved with Keras 3.15.1, which TensorFlow 2.15 cannot load. | Anyone installing as the file says would get an error loading the models. | **Fixed**: requirement is now TensorFlow 2.20 or newer, with a check command. |
| 2 | **Broken live script.** live_word_recognition.py imports a file called updated_files_placeholder that does not exist. | The script cannot run at all. | **Fixed**: rewritten as a working webcam script (untested here: needs a camera, MediaPipe and TensorFlow). The original is kept for reference. |
| 3 | **Random split, not split by person.** train_word_models.py splits recordings at random (stratified by word). If the same person or session is in both training and test, the test score is too high. | The 89.2% may be optimistic. The package has no person or session IDs, so this cannot be checked here. | **Open** (Member 1 and Member 2). See the plan below. |
| 4 | **Small test set.** 240 recordings, 6 per word. One word wrong changes that word's score by about 17 points. | Scores have a wide margin of error. "89%" should be read as roughly 80 to 95%. | **Open**: more recordings per word, ideally from several people. |
| 5 | **Masking after normalisation.** The models use a Masking layer that skips all-zero frames. But frames with missing hands are filled with zeros and then normalised, so they are no longer zero. The mask therefore does not hide them. | Missing hands are treated as real data. Training and use are consistent, so it is not a crash, but it is a design flaw. | **Open** (Member 2). A fix is to mask missing frames before normalising, then retrain. |
| 6 | **Very small spread in some features.** The smallest spread is 0.00098. Dividing by a tiny spread makes small noise look large. | Live camera noise on those features could push predictions off. | **Open**: check after retraining; consider a larger minimum spread. |
| 7 | **Empty class.** The export has 41 classes; the class "i" has no recordings. | Harmless for the 40 trained words, but the export is confusing. | **Handled**: the app uses the 40 trained labels only. |

## Fix plan (for Member 2 and Member 1)

1. **Member 1:** add a person or session ID to each recording in the export (word_dataset.npz), so the split can be done by person.
2. **Member 2:** re-split by person (for example, hold out 1 or 2 people for testing), re-train, and report the new numbers.
3. **Member 2:** mask missing frames before normalising, retrain, and compare.
4. **Both:** record more test samples per word, from more people, and report accuracy with a margin of error.
5. **Member 2:** run `python check_word_artifacts.py` after each retrain (no TensorFlow needed).

## What the app does with the models

- Uses word_gru.keras (the best score on their test set), with the same normalisation as their training code.
- Shows "Demo result" until TensorFlow is installed on the backend.
- The result is only as good as the test above: treat word recognition as a demonstration, not a tested product, until
  steps 1 to 4 are done.
