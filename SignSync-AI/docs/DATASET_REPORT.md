# Dataset report (Member 1 integration)

## 1. Alphabet images (data/raw/alphabets/)

Three image sets were received and checked:

| Source folder | Used as | Images | Letters |
|---|---|---|---|
| Dataset/Validation | Validation | 2,579 | A to Z |
| datasett/Testing | Testing | 5,200 | A to Z |
| datasett/Training | Training | 12,566 | **A to R only** |
| Dataset/Letters | Gallery photos (one per letter) | 26 | A to Z |

Problems found and what was done:

1. **Duplicates across splits.** 487 groups of identical images existed, and 395 of them appeared in more than one split
   (for example Training and Testing). A model tested on images it was trained on gives inflated scores.
   Fix: duplicates were removed with a priority order (Training, then Validation, then Testing), so no identical image is in two splits.
2. **Training has no S to Z.** The Training folder covers only 18 letters. The Validation folder covers all 26 letters
   and can be used to train S to Z. Please confirm before training.

Result after cleaning:

| Split | Images after cleaning | Letters | Min to max per letter |
|---|---|---|---|
| Training | 12,495 | 18 (A to R) | 659 to 700 |
| Validation | 2,519 | 26 | 74 to 101 |
| Testing | 4,770 | 26 | 100 to 200 |

Total kept: 19,784 of 20,345 original images. Removed: 561 duplicates (399 Testing copies of Training, 71 repeats inside Training, 50 Validation copies of Training, 26 repeats inside Testing, 10 repeats inside Validation, 5 Testing copies of Validation).

Limit: this check removes **exact** duplicates only (identical bytes). Re-saved or resized copies of the same photo are not detected.

## 2. Word data (Member 2's MP_Data, 1662 values per frame)

The updated `fix_word_export.py` was run against Member 2's raw word folders:

- 41 class folders, 30 sequences each, 1,230 sequences in total
- output shape X = (1230, 30, 1662), y = (1230,)
- the folder `i` has 30 sequences. The earlier export that Member 2's 40-class model was trained on did not include `i`.

Consequence: Member 2's word models expect 40 classes. Their label indices must be mapped by **class name**
(saved in `classes`), not by position, because the position of letters after `i` moved by one.

The original Holistic `.npy` files (582 MB) and the exported `word_dataset.npz` (214 MB) are not included in the project ZIP because of their size.

## 3. Run order (from the project root, in a terminal with the .venv active)

    python computer_vision/landmark_extraction.py      (choose 1: extract alphabet landmarks)
    python computer_vision/export_features.py           (writes data/exports/alphabet_dataset.npz)
    python computer_vision/fix_word_export.py           (needs data/raw/words/MP_Data)
    python computer_vision/dataset_audit.py             (checks counts)

Landmark extraction needs MediaPipe 0.10.21 (see computer_vision/requirements.txt).

## 4. Video word dataset (Member 1, integrated on 22 November)

**NOT SHIPPED IN THE PROJECT ZIP.** The upload's own note says the source datasets (ISL500, INCLUDE, CISLR) allow only
research and education use and forbid redistribution. Features derived from them are therefore not included in the project.
The code is included, and the data is regenerated on the team's own computer from a copy the team is allowed to use.
Steps to regenerate: see computer_vision/video_dataset/README.md, starting at step 3 (the feature files come from the
upload new__1_.zip, data/video_features_raw; copy them into the project's data folder yourself).

Source: a folder of word videos from two named sources, ISL500 and CISLR. The videos themselves are NOT in the project.
The feature files (614, one per video) are in data/video_features_raw. The upload contained no licence or permission notes.

What was built (computer_vision/video_dataset):
- Extracted features: 614 videos, 30 frames each, 1662 values per frame. The audit found no NaN, no Inf, and the correct schema in all 614.
- 30 words, 407 recording groups (97 ISL500 sessions and 63 CISLR clips counted one by one, plus other clips grouped as their own sessions).
- Split by recording session: training 424, validation 93, test 97 samples. All 30 words are in every split. No session appears in more than one split (checked by the validation script).

Fixes made to Member 1's code during integration:
1. A hard-coded path on Member 1's computer was removed. The video folder is set with SIGNSYNC_VIDEO_DATASET.
2. The code now imports the feature functions from computer_vision/ (one schema, no duplicate copy).
3. Missing frames are no longer filled by interpolation, which invented half-visible hands. The shared mask code is used.
4. The split is by session, not by video, so the same person and room do not appear in both training and test.
5. A coverage step puts every word into the test split, without splitting any session.

Limits:
- Only about 14 clips per word. This is far too little for a reliable model; it is a starting point.
- The 30 words share only 9 words with the app's 40-word vocabulary (food, hello, no, please, sorry, teacher, water, yes, you). The current word model (Member 2) is unchanged. A new model would need the team to decide the vocabulary.
- Licence: ISL500 and CISLR terms are not in the upload. Many video datasets are for research only, and clips taken from YouTube often have restrictions. Check the terms before any use outside the team. Do not publish the videos or the features.

### 4a. Quality check of the video word data (run with check_video_dataset.py)

- Pose points with visibility below 0.5: 44.3%. Pose points outside the picture: 39.9%.
- Clips still with points outside the picture after ignoring low-confidence points: 73.3%.
  The people in these videos are often partly out of frame, so the body points are guesses. A live webcam shows the whole upper body; this is a domain difference.
- Vocabulary: 9 of the app's 40 words are covered.
- Signer overlap: 0 of 97 test clips share an ISL500 session with training.

Conclusion: these videos are weak as training data for the live camera. Use them only after a teacher checks the labels, and consider recording the team's own videos framed like the webcam.

## 5. Sign videos from ISL_40_Words (received 22 November)

The archive has 642 clips for 40 words, from four sources. Each source has its own licence:

| Source | Clips | Licence | May we share the clips? |
|---|---|---|---|
| INCLUDE | 143 | CC-BY-4.0 | Yes, with credit to the authors and a link to the licence |
| ISL500 | 405 | see the source card | No, not without checking the terms |
| CISLR | 81 | research (AFL-3.0) | No, not without checking the terms |
| ISLRTC dictionary | 13 | MIT, plus data.gov.in terms | Check the data.gov.in terms first |

What is in the project:
- Only the INCLUDE clips for words in the app: hello and thanks (2 clips), with CREDITS.txt listing each clip's licence and links.
- The other clips are NOT in the ZIP. scripts/install_sign_videos.py copies them into frontend/public/signs/words on your own computer, for research and education use. It writes CREDITS.txt for them too.

Mismatch to decide: the archive's 40 words share only 9 words with the app's 40-word vocabulary (food, hello, no, please, sorry, teacher, water, yes, you). Matching the app to this list would also need Member 2's word model to be retrained on it.

The archive's JSON files contain a personal Windows path on another person's computer. They are not used or shipped.
