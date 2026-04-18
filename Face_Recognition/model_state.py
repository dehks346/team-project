"""Helpers for tracking face dataset changes.

This module keeps the logic small and testable so the app can detect when
new face images were added and retrain the recognizer automatically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_dataset_images(dataset_dir: Path) -> Iterable[Path]:
    """Yield image files in a deterministic order."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        return []

    for person_dir in sorted(
        (path for path in dataset_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.lower(),
    ):
        for image_path in sorted(
            (
                path
                for path in person_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        ):
            yield image_path


def build_dataset_fingerprint(dataset_dir: Path) -> Dict[str, object]:
    """Return a stable fingerprint for the current dataset contents."""
    dataset_dir = Path(dataset_dir)
    digest = hashlib.sha256()
    image_count = 0
    people = set()

    if not dataset_dir.exists():
        return {
            "dataset_signature": digest.hexdigest(),
            "image_count": 0,
            "person_count": 0,
        }

    for image_path in iter_dataset_images(dataset_dir):
        people.add(image_path.parent.name)
        digest.update(image_path.parent.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(image_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(image_path.read_bytes())
        digest.update(b"\0")
        image_count += 1

    digest.update(str(image_count).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(len(people)).encode("utf-8"))

    return {
        "dataset_signature": digest.hexdigest(),
        "image_count": image_count,
        "person_count": len(people),
    }


def load_training_state(state_path: Path) -> Dict[str, object]:
    """Load the last saved dataset fingerprint, if present."""
    state_path = Path(state_path)
    if not state_path.exists():
        return {}

    try:
        with open(state_path, "r", encoding="utf-8") as file:
            state = json.load(file)
            return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def save_training_state(state_path: Path, state: Dict[str, object]) -> None:
    """Persist the last known dataset fingerprint to disk."""
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=4, sort_keys=True)


def dataset_requires_retraining(dataset_dir: Path, state_path: Path) -> bool:
    """Return True when the dataset fingerprint changed since the last train."""
    current = build_dataset_fingerprint(dataset_dir)
    saved = load_training_state(state_path)
    return current.get("dataset_signature") != saved.get("dataset_signature")
