"""Audit the extracted word features (Member 1, integrated). Writes data/video_features_raw/audit_report.csv."""
from pathlib import Path

import numpy as np
import csv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA_DIR = ROOT / "data" / "video_features_raw"
FEATURE_COUNT = 1662

rows = []
for path in sorted(DATA_DIR.glob("*.npz")):
    d = np.load(path, allow_pickle=True)
    f = d["features"]
    rows.append({
        "video": str(d["video_name"]),
        "label": str(d["label"]),
        "frames": f.shape[0],
        "feature_count": f.shape[1],
        "correct_schema": f.shape[1] == FEATURE_COUNT,
        "nan_values": int(np.isnan(f).sum()),
        "inf_values": int(np.isinf(f).sum()),
        "zero_frames": int(np.sum(np.all(f == 0, axis=1))),
        "status": "OK" if (f.shape[1] == FEATURE_COUNT and np.isfinite(f).all()) else "CHECK",
    })
out = DATA_DIR / "audit_report.csv"
with open(out, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["video"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Audited {len(rows)} files -> {out}")
