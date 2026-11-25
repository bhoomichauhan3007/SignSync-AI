
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "raw" / "alphabets"

for split in ["Training", "Testing", "Validation"]:
    print(f"\nChecking {split}...", flush=True)

    checked = 0
    corrupt = 0

    for path in DATASET.joinpath(split).rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        try:
            with Image.open(path) as img:
                img.verify()
            checked += 1
        except Exception:
            corrupt += 1

        if checked % 500 == 0 and checked:
            print(f"Checked: {checked}", flush=True)

    print(f"Valid images: {checked}", flush=True)
    print(f"Corrupt images: {corrupt}", flush=True)

print("\nQUALITY CHECK COMPLETE", flush=True)
