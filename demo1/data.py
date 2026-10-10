"""Reproducible ESC-50 subset downloader. Audio is cached, never synthesized silently."""
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ESC_REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
BASE = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{ESC_REVISION}"
CLASSES = ("engine", "siren", "keyboard_typing", "rain", "dog")


def fetch(url, destination):
    destination = Path(destination)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=90) as response:
            payload = response.read()
        temp = destination.with_suffix(destination.suffix + ".tmp")
        temp.write_bytes(payload)
        temp.replace(destination)
    return destination


def metadata():
    path = fetch(BASE + "/meta/esc50.csv", DATA / "esc50.csv")
    fetch(BASE + "/LICENSE", DATA / "ESC50_LICENSE.txt")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare_data(classifier=False):
    rows = [r for r in metadata() if r["category"] in CLASSES]
    if not classifier:
        rows = [next(r for r in rows if r["category"] == c) for c in CLASSES[:3]]
    def download(row):
        path = fetch(BASE + "/audio/" + row["filename"], DATA / "audio" / row["filename"])
        return {**row, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "url": BASE + "/audio/" + row["filename"]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(download, rows))
    (DATA / ("classifier_manifest.json" if classifier else "demo_a_manifest.json")).write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return rows


def demo_clips():
    import librosa
    import shutil
    import contextlib
    import io
    rows = prepare_data()
    clips = [(r["category"], DATA / "audio" / r["filename"]) for r in rows]
    speech = DATA / "speech" / "libri1.ogg"
    if not speech.exists():
        speech.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(librosa.ex("libri1"), speech)
    license_path = DATA / "SPEECH_LICENSE.txt"
    if not license_path.exists():
        info = io.StringIO()
        with contextlib.redirect_stdout(info):
            librosa.util.example_info("libri1")
        license_path.write_text(info.getvalue(), encoding="utf-8")
    (DATA / "speech_manifest.json").write_text(json.dumps({
        "example":"libri1", "file":speech.name,
        "sha256":hashlib.sha256(speech.read_bytes()).hexdigest(),
        "source":"https://librosa.org/data/audio/5703-47212-0000.ogg",
        "analysis_duration_seconds":5}, indent=2)+"\n", encoding="utf-8")
    clips.append(("speech (LibriSpeech, separate from ESC-50)", speech))
    return clips
