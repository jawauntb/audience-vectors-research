"""Memento10k adapter (Newman et al., ECCV 2020).

Memento10k is a dynamic video memorability dataset: ~10,000 in-the-wild
clips, each ~3 seconds, with mean memorability + decay (alpha), action
labels, and 5 captions per video. Source: http://memento.csail.mit.edu/

Expected on-disk layout under `MEMENTO10K_ROOT`:

    memento10k/
    ├── videos/
    │   ├── video_00001.webm
    │   └── ...
    └── annotations/
        ├── memento_train_data.json
        ├── memento_val_data.json
        └── memento_test_data.json

Each JSON file is a list of entries with at least:

    {
        "filename": "video_00001.webm",
        "mem_score": 0.87,            # average memorability across viewers
        "alpha": -0.012,              # memorability decay slope
        "captions": ["a dog runs ...", ...],
        "actions": ["running", "dog", ...]
    }

If your downloaded copy has different field names or file layout, adjust
the `FIELD_*` constants below — that's the only place field assumptions
live. The rest of the adapter is schema-agnostic.

Memorability and decay are video-level, so labels are emitted with
`granularity="video"`. The segmentation step downstream broadcasts
them to per-segment LabelValues (Memento clips are already ~3s, so
broadcast is faithful).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from audience_vectors.datasets.base import DatasetAdapter
from audience_vectors.schemas import CanonicalVideo

# ---------------------------------------------------------------------------
# Schema knobs — adjust here if your local copy uses different field names.
# ---------------------------------------------------------------------------

FIELD_FILENAME = "filename"
FIELD_MEM_SCORE = "mem_score"
FIELD_ALPHA = "alpha"
FIELD_CAPTIONS = "captions"
FIELD_ACTIONS = "actions"

ANNOTATION_FILES: dict[str, str] = {
    "train": "memento_train_data.json",
    "val": "memento_val_data.json",
    "test": "memento_test_data.json",
}

VIDEOS_SUBDIR = "videos"
ANNOTATIONS_SUBDIR = "annotations"


class Memento10kAdapter(DatasetAdapter):
    """Yield one CanonicalVideo per Memento10k clip across all splits."""

    name = "Memento10k"
    default_domain = "everyday"

    def __init__(
        self,
        root: Path,
        *,
        splits: tuple[str, ...] = ("train", "val", "test"),
    ) -> None:
        super().__init__(root)
        unknown = [s for s in splits if s not in ANNOTATION_FILES]
        if unknown:
            raise ValueError(
                f"unknown Memento10k splits: {unknown}; "
                f"valid: {tuple(ANNOTATION_FILES)}"
            )
        self.splits = splits
        self.videos_dir = self.root / VIDEOS_SUBDIR
        self.annotations_dir = self.root / ANNOTATIONS_SUBDIR

    # -- iteration ---------------------------------------------------------

    def iter_videos(self) -> Iterator[CanonicalVideo]:
        for split in self.splits:
            ann_path = self.annotations_dir / ANNOTATION_FILES[split]
            if not ann_path.exists():
                # Skip missing splits silently — some downloads only ship
                # train+val, and test labels are sometimes held out.
                continue
            with ann_path.open("r", encoding="utf-8") as fh:
                entries = json.load(fh)
            if not isinstance(entries, list):
                raise ValueError(
                    f"Memento10k annotation file is not a JSON list: {ann_path}"
                )
            for entry in entries:
                yield self._entry_to_canonical(entry, split=split)

    # -- per-entry mapping -------------------------------------------------

    def _entry_to_canonical(
        self,
        entry: dict[str, Any],
        *,
        split: str,
    ) -> CanonicalVideo:
        filename = entry.get(FIELD_FILENAME)
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"Memento10k entry missing {FIELD_FILENAME!r}: {entry!r}"
            )

        video_id = Path(filename).stem
        media_uri = str(self.videos_dir / filename)

        raw_labels: dict[str, float | str | None] = {}
        if (score := entry.get(FIELD_MEM_SCORE)) is not None:
            raw_labels["mem_score"] = float(score)
        if (alpha := entry.get(FIELD_ALPHA)) is not None:
            raw_labels["alpha"] = float(alpha)

        captions = entry.get(FIELD_CAPTIONS) or []
        actions = entry.get(FIELD_ACTIONS) or []

        metadata: dict[str, str | int | float | bool | None] = {
            "split": split,
            "filename": filename,
            "n_captions": len(captions) if isinstance(captions, list) else 0,
            "n_actions": len(actions) if isinstance(actions, list) else 0,
        }
        # Keep the joined caption + first action in metadata so downstream
        # synthetic labelers have text without needing to re-open the JSON.
        if isinstance(captions, list) and captions:
            metadata["caption_joined"] = " | ".join(str(c) for c in captions[:5])
        if isinstance(actions, list) and actions:
            metadata["first_action"] = str(actions[0])

        return CanonicalVideo(
            video_id=f"memento_{video_id}",
            source_dataset=self.name,
            media_uri=media_uri,
            duration_s=None,  # Memento clips are ~3s but we don't probe here
            domain=self.default_domain,
            raw_labels=raw_labels,
            metadata=metadata,
        )
