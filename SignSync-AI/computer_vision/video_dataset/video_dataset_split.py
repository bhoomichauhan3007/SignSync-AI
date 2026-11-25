"""Split the word dataset by RECORDING SESSION (Member 1, integrated and fixed).

Fix in this integration: the upload split by video, so clips from the same session could be in both training
and testing (the same person and room). Here every ISL500 session is kept in one split only. CISLR clips have
no session number in their names, so each clip counts as its own group (flagged in the report).
Split sizes: about 70% training, 15% validation, 15% testing, chosen by group.
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INPUT_FILE = ROOT / "data" / "video_sequences" / "isl_word_sequences.npz"
OUTPUT_DIR = ROOT / "data" / "final_dataset"
SEED = 42


def group_of(video_name):
    match = re.search(r"session(\d+)", video_name)
    if match:
        return f"ISL500-session{match.group(1)}"
    return f"clip-{video_name}"            # CISLR: no session in the name


def main():
    data = np.load(INPUT_FILE, allow_pickle=True)
    X, M, labels, names = data["X"], data["M"], data["y"], data["video_names"]
    classes = sorted(set(labels.tolist()))
    class_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_id[c] for c in labels], dtype=np.int64)
    groups = np.array([group_of(n) for n in names])

    rng = np.random.default_rng(SEED)
    unique = sorted(set(groups.tolist()))
    rng.shuffle(unique)
    target = {"test": 0.15 * len(y), "validation": 0.15 * len(y)}
    assigned = {"test": [], "validation": [], "train": []}
    counts = Counter()
    for g in unique:
        members = np.where(groups == g)[0]
        if counts["test"] < target["test"]:
            split = "test"
        elif counts["validation"] < target["validation"]:
            split = "validation"
        else:
            split = "train"
        assigned[split].extend(members.tolist())
        counts[split] += len(members)

    # Coverage repair: every word should appear in the test split. A whole group (session or clip) is moved
    # from training to testing, never a single clip, so the no-shared-group rule still holds. A move is refused
    # if it would leave any word with no training examples.
    def class_set(indices):
        return set(labels[indices].tolist())

    for word in classes:
        test_words = class_set(np.array(assigned["test"], dtype=np.int64)) if assigned["test"] else set()
        if word in test_words:
            continue
        candidates = sorted(
            {groups[i] for i in assigned["train"] if labels[i] == word},
            key=lambda g: int(np.sum(groups == g)),
        )
        for g in candidates:
            members = [i for i in assigned["train"] if groups[i] == g]
            train_left = [i for i in assigned["train"] if groups[i] != g]
            if set(labels[train_left].tolist()) < set(classes):
                continue                          # moving this group would empty some word in training
            assigned["train"] = train_left
            assigned["test"].extend(members)
            break

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, idx in assigned.items():
        idx = np.array(sorted(idx), dtype=np.int64)
        filename = {"train": "train", "validation": "validation", "test": "test"}[split]
        np.savez_compressed(
            OUTPUT_DIR / f"{filename}.npz",
            X=X[idx], M=M[idx], y=y[idx], labels=labels[idx], video_names=names[idx],
            groups=groups[idx], class_names=np.array(classes),
        )
        print(f"{split:<11} {len(idx):>5} samples, {len(set(labels[idx].tolist())):>2} words")
    np.savez_compressed(OUTPUT_DIR / "label_mapping.npz", class_names=np.array(classes),
                        label_ids=np.arange(len(classes)))
    print("Sessions or clips:", len(unique), "| words:", len(classes))
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
