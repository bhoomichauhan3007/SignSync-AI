# Member 2 — ML / Sign Recognition

A practical starter package for the ML member of the sign-language recognition project.
It supports:
- Dataset inspection for `.npz`, `.npy`, and CSV files
- Alphabet classification using Random Forest and MLP
- Optional 1D-CNN for landmark vectors
- Sequence classification using LSTM, GRU, and Transformer (TensorFlow/Keras)
- Evaluation reports, confusion matrix, saved models, and single-sample prediction

## 1. Put your data here

Copy the dataset files supplied by Member 1 into `data/`, for example:
- `alphabet_dataset.npz`
- `word_dataset.npz`

The package does not contain your dataset.

## 2. Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

TensorFlow is optional. Install it separately if you want CNN/LSTM/GRU/Transformer:
```bash
pip install tensorflow
```

## 3. Inspect the dataset first

```bash
python src/inspect_dataset.py --data_dir data
```

Read the printed NPZ keys and shapes. The loader tries common names (`X`, `x`, `features`, `X_train`, etc.),
but if your archive uses unusual keys, pass explicit keys to the training scripts.

## 4. Train alphabet models

```bash
python src/train_alphabet.py --file data/alphabet_dataset.npz --x_key X --y_key y --model all
```

If the keys differ, replace `X` and `y` with the exact keys printed by the inspector.
For a split archive, use `--x_key X_train --y_key y_train --x_test_key X_test --y_test_key y_test`.

Models: `rf`, `mlp`, `cnn`, or `all`. CNN requires TensorFlow.

## 5. Train sequence models

Expected input shape: `(samples, frames, features)`, e.g. `(1200, 30, 1662)`.
Expected labels: one class ID per sequence. If the word archive has a different schema,
use the keys shown by the inspector.

```bash
python src/train_sequence.py --file data/word_dataset.npz --x_key X --y_key y --model lstm
python src/train_sequence.py --file data/word_dataset.npz --x_key X --y_key y --model gru
python src/train_sequence.py --file data/word_dataset.npz --x_key X --y_key y --model transformer
```

## Important notes
- Do not assume the sequence feature dimension is correct until it matches the feature-extraction schema.
- Keep train/test splits grouped by signer if signer IDs are available; this avoids signer leakage.
- Never report accuracy until the script has actually run on your dataset.
- This is a baseline pipeline. Tune architecture, augmentation, class balancing, and validation only after confirming the data schema.
