# External video word dataset (Member 1): what it is, what is wrong with it, what to do

## What was received

A folder called new (Member 1, "words VLA dataset"). It holds:
- Scripts that take video clips, run MediaPipe Holistic on 30 evenly spaced frames, and save 1662 values per frame
  (the same layout as the rest of the project).
- A ready split of that data: train 418 clips, validation 98, test 98, all 30 words classes listed in label_mapping.
- 615 raw feature files from the videos (not included in the project, see below).

Important: the zip contains FEATURES, not the videos. The sign video clips themselves are not in it.

## Where the clips come from

| Source | Clips in training | Terms (as far as we checked) |
|---|---|---|
| ISL500 | 273 | Not checked. Read its terms before use. |
| INCLUDE (AI4Bharat) | 103 | Released for research and education. The published card says it is for isolated sign recognition and education, and asks users to respect the privacy of the people shown. |
| CISLR (IIT Kanpur) | 39 | Access is gated: you must agree to terms before downloading. Terms not read here. |

Conclusion: research and education use only. Not for commercial use. Do not redistribute the data.
The feature files are derived from these datasets, so they are treated the same way: they are NOT part of the shared project.

## What the checker found (docs and checker: computer_vision/video_pipeline/check_video_dataset.py)

1. **Only 9 of SignSync's 40 words are covered.** Shared: food, hello, no, please, sorry, teacher, water, yes, you.
   21 words are only in this data (for example drink, eat, father). 31 SignSync words have no data at all.
   So this data cannot train the current 40-word model without a new data collection.

2. **Pose points are often guesses.** About 44% of pose points have visibility below 0.5. Of the points outside the picture,
   96% are low-visibility guesses. Ignoring the low-visibility points still leaves 73% of clips with a point outside the picture.
   The team's rule (Member 1's feature_schema.py) is to mark visibility below 0.5 as missing. The new data does not do this.
   Member 2's word recordings may have the same effect, so both should be treated the same way.

3. **Same signers may be in train and test.** 38 of the 98 test clips come from ISL500 sessions that also appear in training.
   A session is usually one person recorded in one sitting, so the test score would be too high.
   The split in this data is by video and by class, not by person.

4. **The final validation script says "ready for handover".** It only checks shapes and overlaps. It does not check
   the problems above, so do not rely on its message.

## What was done in the project

- Scripts copied to computer_vision/video_pipeline/ (unchanged, except the checker is new).
- The checker is included. Run it from computer_vision: python video_pipeline/check_video_dataset.py
- The data is NOT in the shared ZIP. It is in a separate ZIP marked RESEARCH ONLY (see the file name).
- The data is NOT merged into Member 2's word model. Merging needs the decisions below.

## What is needed before this can be used

1. Decide with the team: is this data for research only? Then read and record the terms of ISL500, INCLUDE and CISLR.
2. Mask pose points with visibility below 0.5 (set to zero) in training AND in live use, then re-check.
3. Split by person (ISL500 session ID is the best available proxy), not by video.
4. Decide whether to collect new recordings for the 31 missing SignSync words, using the team's own recorder.
5. Map class names to SignSync's labels by NAME (not by position), as Member 1's CHANGES.md already says.
