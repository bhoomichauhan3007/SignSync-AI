# SignSync AI — Member 2: Word Sign Recognition

This package is the ML hand-off for the word-recognition part. It consumes Member 1's exported `word_dataset.npz` (`X`, `y`, `classes`) with shape `(samples, 30, 1662)`, trains LSTM, GRU and Transformer classifiers, evaluates them, and exposes a reusable predictor.

## 1. Put the dataset in place

Copy Member 1's `word_dataset.npz` into this folder (same level as `train_word_models.py`). The expected arrays are:
- `X`: float array `(N, 30, 1662)`
- `y`: one label per sample (integer or text)
- `classes`: optional class-name array; used when `y` contains integer IDs.

## 2. Windows PowerShell setup

```powershell
cd path\to\SignSync_Member2_Word_Recognition
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python inspect_dataset.py --data word_dataset.npz
python train_word_models.py --data word_dataset.npz --epochs 60
```

TensorFlow GPU training is optional. On a low-memory laptop, train in Google Colab and download the generated `artifacts` folder.

## 3. Outputs

`artifacts/word_lstm.keras`, `word_gru.keras`, `word_transformer.keras`, `label_encoder.json`, `normalization.npz`, and per-model metrics JSON. The script prints test accuracy and macro-F1. These are measured results, not guaranteed scores.

## 4. Predict a sequence

```powershell
python predict_word.py --model artifacts/word_lstm.keras --sequence sample_sequence.npy
```
`sample_sequence.npy` must contain `(30, 1662)` float features in the same feature order used by Member 1.

## Important

- A class with zero examples (for example, `i` if absent from the export) cannot be learned. Add real samples and re-export before expecting that class to be recognized.
- Do not mix the 63-feature alphabet data with this 1662-feature word data.
- Keep the exact feature order from Member 1's `holistic_features.py`.
- This is an ML module; it does not replace the team's frontend/backend integration.
