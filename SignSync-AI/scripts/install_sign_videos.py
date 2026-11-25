"""Install sign videos from the ISL_40_Words archive into the app (LOCAL USE, not shipped in the ZIP).

Usage (from the project root):
    python scripts\\install_sign_videos.py --zip "C:\\Users\\dell\\Downloads\\ISL_40_Words.zip"
    python scripts\\install_sign_videos.py --folder "C:\\Users\\dell\\Downloads\\ISL_40_Words"   (extracted folder)
    python scripts\\install_sign_videos.py --zip ... --sources INCLUDE     (CC-BY clips only; safe to share with credit)
    python scripts\\install_sign_videos.py --zip ... --dest "C:\\temp\\test"  (try it somewhere else first)

What it writes:
  <dest>/videos/<word>.mp4      one video for each of the 40 archive words (the Video dictionary tab)
  <dest>/words/<gloss>.mp4      the same video for the app's own words that match (hello, water, ...)
  <dest>/videos/CREDITS.txt     source, licence and link of every video written

Licences (from the archive README):
  INCLUDE            CC-BY-4.0      share with credit
  ISL500             see source card  research/education; do NOT redistribute without checking
  CISLR              AFL-3.0 (research) research use; do NOT redistribute without checking
  ISLRTC_dictionary  MIT, plus data.gov.in terms
Do not put the ISL500 or CISLR videos in a project ZIP that you share.
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {"THANKS": "thank you", "BYEBYE": "goodbye", "HOUSE": "home"}
SOURCE_LINKS = {
    "INCLUDE": "https://creativecommons.org/licenses/by/4.0/ (INCLUDE dataset, CC-BY-4.0: credit the authors)",
    "ISL500": "https://huggingface.co/datasets/ISL500/ISL-DATA (terms on the source card)",
    "CISLR": "https://huggingface.co/datasets/Exploration-Lab/CISLR (research use, AFL-3.0)",
    "ISLRTC_dictionary": "https://huggingface.co/datasets/Vignesh3816/Indian_Sign_Language_Data.gov_Rencoded (MIT; check data.gov.in terms)",
}


class ZipSource:
    """Reads files from ISL_40_Words.zip (files live under the ISL_40_Words/ folder inside the zip)."""
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
    def names(self):
        return {n[len("ISL_40_Words/"):] for n in self.zip.namelist() if n.startswith("ISL_40_Words/")}
    def read_text(self, rel):
        return self.zip.read("ISL_40_Words/" + rel).decode("utf-8")
    def read_bytes(self, rel):
        return self.zip.read("ISL_40_Words/" + rel)


class FolderSource:
    """Reads files from an extracted ISL_40_Words folder (the folder that contains metadata.csv)."""
    def __init__(self, path):
        self.root = Path(path)
    def names(self):
        return {str(p.relative_to(self.root)).replace("\\", "/") for p in self.root.rglob("*.mp4")}
    def read_text(self, rel):
        return (self.root / rel).read_text(encoding="utf-8")
    def read_bytes(self, rel):
        return (self.root / rel).read_bytes()


def app_vocabulary():
    return [w.lower() for w in json.loads((ROOT / "nlp_translation" / "sign_vocab.json").read_text(encoding="utf-8")) if len(w) > 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, help="the ISL_40_Words.zip file")
    parser.add_argument("--folder", type=Path, help="or the extracted ISL_40_Words folder (the one with metadata.csv)")
    parser.add_argument("--sources", nargs="*", default=None, help="limit to these sources, e.g. INCLUDE")
    parser.add_argument("--dest", type=Path, default=ROOT / "frontend" / "public" / "signs")
    args = parser.parse_args()

    if bool(args.zip) == bool(args.folder):
        parser.error("give exactly one of --zip or --folder")
    if args.folder:
        source = FolderSource(args.folder)
    else:
        source = ZipSource(args.zip)
    rows = list(csv.DictReader(io.StringIO(source.read_text("metadata.csv"))))
    present = source.names()
    chosen = {}                       # archive word -> first accepted row whose video is in the archive
    missing = 0
    for row in rows:
        if row["review_status"] != "accepted":
            continue
        if args.sources and row["dataset"] not in args.sources:
            continue
        if row["video_path"].replace("\\", "/") not in present:
            missing += 1
            continue
        chosen.setdefault(row["normalized_word"].lower(), row)
    if missing:
        print(f"Note: {missing} metadata row(s) point to a video that is not in the archive; they were skipped.")

    videos_dir = args.dest / "videos"
    words_dir = args.dest / "words"
    videos_dir.mkdir(parents=True, exist_ok=True)
    words_dir.mkdir(parents=True, exist_ok=True)
    credits = []
    for word, row in sorted(chosen.items()):
        data = source.read_bytes(row["video_path"].replace("\\", "/"))
        slug = word.replace(" ", "_")
        (videos_dir / f"{slug}.mp4").write_bytes(data)
        for gloss in app_vocabulary():
            if ALIASES.get(gloss.upper(), gloss).lower() == word:
                (words_dir / f"{gloss}.mp4").write_bytes(data)
        credits.append(f"{slug}.mp4 | {row['dataset']} | {row['license']} | {SOURCE_LINKS.get(row['dataset'], '')} | source {row['original_filename']}")
    (videos_dir / "CREDITS.txt").write_text("Sign video credits (check each licence before sharing).\n" + "\n".join(credits) + "\n", encoding="utf-8")
    print(f"{len(chosen)} video(s) written to {videos_dir}")
    print(f"Credits: {videos_dir / 'CREDITS.txt'}")


if __name__ == "__main__":
    main()
