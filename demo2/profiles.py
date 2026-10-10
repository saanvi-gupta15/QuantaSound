"""Local, explicit preference storage. This is not automatic preference learning."""
import json
import os
from pathlib import Path
import re
import tempfile

PROFILE_DIR = Path(__file__).resolve().parent / "local" / "profiles"
CHOICES = ("engine", "keyboard")


def validate_selection(reduce):
    if not isinstance(reduce, (list, tuple)) or any(x not in CHOICES for x in reduce):
        raise ValueError("Choose only engine and/or keyboard; speech is preserved")
    if len(set(reduce)) != len(reduce):
        raise ValueError("Duplicate sound selection")
    return [sound for sound in CHOICES if sound in reduce]


def profile_path(name, directory=PROFILE_DIR):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", name):
        raise ValueError("Profile name: 1–32 letters, numbers, hyphens or underscores")
    return Path(directory) / (name + ".json")


def save_profile(name, reduce, directory=PROFILE_DIR):
    destination = profile_path(name, directory)
    profile = {"version":1, "name":name, "reduce":validate_selection(reduce), "preserve_speech":True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                     suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(profile, handle, indent=2)
        handle.write("\n")
    try:
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return profile


def load_profile(name, directory=PROFILE_DIR):
    path = profile_path(name, directory)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"No saved profile named {name}") from None
    except json.JSONDecodeError:
        raise ValueError(f"Profile {name} is not valid JSON") from None
    if (not isinstance(profile, dict) or profile.get("version") != 1
            or profile.get("name") != name or profile.get("preserve_speech") is not True):
        raise ValueError("Unsupported or invalid profile format")
    profile["reduce"] = validate_selection(profile.get("reduce"))
    return profile


def list_profiles(directory=PROFILE_DIR):
    return sorted(p.stem for p in Path(directory).glob("*.json"))
