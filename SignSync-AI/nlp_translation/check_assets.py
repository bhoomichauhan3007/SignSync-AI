
import json
from pathlib import Path

# Load vocabulary
with open("sign_vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Project Integrated folder
project_root = Path(__file__).resolve().parent.parent

available = []
missing = []

for word, relative_path in vocab.items():
    full_path = project_root / relative_path

    if full_path.exists():
        available.append(word)
    else:
        missing.append((word, relative_path))

print("TOTAL ASSETS:", len(vocab))
print("AVAILABLE:", len(available))
print("MISSING:", len(missing))

print("\nMISSING ASSETS:")
for word, path in missing:
    print(f"{word} -> {path}")
