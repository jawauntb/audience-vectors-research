"""BOLD Moments Dataset adapter (Lahner et al., Nature Communications 2024).

1,102 × 3-second video clips with rich human annotations + fMRI responses
from 10 subjects. The fMRI data + stimulus annotations are publicly
downloadable from OpenNeuro `ds005165` (no auth, no form):

    aws s3 cp --no-sign-request \\
        s3://openneuro.org/ds005165/derivatives/stimuli_metadata/annotations.json \\
        $BOLD_MOMENTS_ROOT/annotations.json

The stimulus videos themselves are NOT included in OpenNeuro (licensing),
but each annotation entry carries `MiT_url` — a direct link to the
underlying Moments in Time clip on MIT's csail bucket. The optional
`scripts/download_bold_moments_videos.py` helper walks those URLs.

Expected on-disk layout under `BOLD_MOMENTS_ROOT`:

    bold_moments/
    ├── annotations.json         (REQUIRED — from OpenNeuro S3)
    └── videos/                  (OPTIONAL — populated by download script)
        ├── vid_idx0001.mp4
        └── ...

Real annotation schema (per entry, 1102 entries keyed "0001".."1102"):

    {
        "bmd_matrixfilename": "vid_idx0001",
        "MiT_url": "https://data.csail.mit.edu/soundnet/.../clip.mp4",
        "MiT_filename": "wetting/0-0-1-6-7-2-8-0-17500167280.mp4",
        "set": "train" | "test",
        "objects":             list[list[str]]  # per-annotator triples
        "scenes":              list[str]         # flat across annotators
        "actions":             list[str]         # flat across annotators
        "text_descriptions":   list[str]         # 5 free-form captions
        "spoken_transcription": str
        "memorability_score":  float            # [0, 1]
        "memorability_decay":  float            # decay slope
        ...
    }

Memorability and decay are per-video — emitted with `granularity="video"`.
Segmentation downstream broadcasts; clips are ~3s so broadcast is faithful.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from audience_vectors.datasets.base import DatasetAdapter
from audience_vectors.schemas import CanonicalVideo

# ---------------------------------------------------------------------------
# Schema knobs — one-line patches if the upstream format ever shifts.
# ---------------------------------------------------------------------------

ANNOTATIONS_FILE = "annotations.json"
VIDEOS_SUBDIR = "videos"
VIDEO_FILENAME_TEMPLATE = "{matrix_name}.mp4"

FIELD_MATRIX_NAME = "bmd_matrixfilename"
FIELD_MIT_URL = "MiT_url"
FIELD_MIT_FILENAME = "MiT_filename"
FIELD_SET = "set"
FIELD_OBJECTS = "objects"
FIELD_SCENES = "scenes"
FIELD_ACTIONS = "actions"
FIELD_TEXT = "text_descriptions"
FIELD_TRANSCRIPTION = "spoken_transcription"
FIELD_MEM_SCORE = "memorability_score"
FIELD_MEM_DECAY = "memorability_decay"

# BMD clips reuse Moments in Time stimuli — each clip is documented as 3s.
NOMINAL_DURATION_S = 3.0


def _flatten_objects(objects: Any) -> list[str]:
    """`objects` is list[list[str]]; flatten + drop the '--' sentinel."""
    if not isinstance(objects, list):
        return []
    flat: list[str] = []
    for group in objects:
        if isinstance(group, list):
            flat.extend(o for o in group if isinstance(o, str) and o != "--")
        elif isinstance(group, str) and group != "--":
            flat.append(group)
    return flat


def _filter_strs(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [v for v in values if isinstance(v, str) and v]


class BoldMomentsAdapter(DatasetAdapter):
    """Yield one CanonicalVideo per BMD clip across train + test sets."""

    name = "BOLDMoments"
    default_domain = "everyday"

    def __init__(
        self,
        root: Path,
        *,
        sets: tuple[str, ...] = ("train", "test"),
        prefer_local_videos: bool = True,
    ) -> None:
        """Args:
            root: directory containing `annotations.json` (and optionally `videos/`).
            sets: which subsets to emit. ('train',) ~1000 clips, ('test',) ~102.
            prefer_local_videos: if True and `videos/{matrix_name}.mp4` exists,
                use that local path as `media_uri`; otherwise fall back to MiT_url.
        """
        super().__init__(root)
        valid_sets = {"train", "test"}
        unknown = [s for s in sets if s not in valid_sets]
        if unknown:
            raise ValueError(
                f"unknown BOLD Moments sets: {unknown}; valid: {sorted(valid_sets)}"
            )
        self.sets = sets
        self.prefer_local_videos = prefer_local_videos
        self.annotations_path = self.root / ANNOTATIONS_FILE
        self.videos_dir = self.root / VIDEOS_SUBDIR

        if not self.annotations_path.exists():
            raise FileNotFoundError(
                f"BOLD Moments annotations.json not found at {self.annotations_path}. "
                "Download with:\n"
                f"  aws s3 cp --no-sign-request "
                f"s3://openneuro.org/ds005165/derivatives/stimuli_metadata/annotations.json "
                f"{self.annotations_path}"
            )

    # -- iteration ---------------------------------------------------------

    def iter_videos(self) -> Iterator[CanonicalVideo]:
        with self.annotations_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(
                f"BOLD Moments annotations.json is not a JSON object: {self.annotations_path}"
            )
        for entry_id, entry in data.items():
            entry_set = entry.get(FIELD_SET)
            if entry_set not in self.sets:
                continue
            yield self._entry_to_canonical(entry_id, entry)

    # -- per-entry mapping -------------------------------------------------

    def _entry_to_canonical(
        self,
        entry_id: str,
        entry: dict[str, Any],
    ) -> CanonicalVideo:
        matrix_name = entry.get(FIELD_MATRIX_NAME) or f"vid_idx{entry_id}"
        mit_url = entry.get(FIELD_MIT_URL) or ""

        local_video = self.videos_dir / VIDEO_FILENAME_TEMPLATE.format(
            matrix_name=matrix_name,
        )
        if self.prefer_local_videos and local_video.exists():
            media_uri = str(local_video)
        else:
            media_uri = mit_url

        raw_labels: dict[str, float | str | None] = {}
        if (score := entry.get(FIELD_MEM_SCORE)) is not None:
            raw_labels["memorability_score"] = float(score)
        if (decay := entry.get(FIELD_MEM_DECAY)) is not None:
            raw_labels["memorability_decay"] = float(decay)

        objects = _flatten_objects(entry.get(FIELD_OBJECTS))
        scenes = _filter_strs(entry.get(FIELD_SCENES))
        actions = _filter_strs(entry.get(FIELD_ACTIONS))
        captions = _filter_strs(entry.get(FIELD_TEXT))
        transcription = entry.get(FIELD_TRANSCRIPTION)

        metadata: dict[str, str | int | float | bool | None] = {
            "split": entry.get(FIELD_SET, "unknown"),
            "matrix_name": matrix_name,
            "entry_id": entry_id,
            "mit_url": mit_url,
            "mit_filename": entry.get(FIELD_MIT_FILENAME) or "",
            "n_objects": len(objects),
            "n_scenes": len(scenes),
            "n_actions": len(actions),
            "n_captions": len(captions),
            "local_video_present": local_video.exists() if self.prefer_local_videos else False,
        }
        if captions:
            metadata["caption_joined"] = " | ".join(captions[:5])
        if scenes:
            metadata["top_scene"] = scenes[0]
        if actions:
            metadata["top_action"] = actions[0]
        if isinstance(transcription, str) and transcription:
            metadata["spoken_transcription"] = transcription

        return CanonicalVideo(
            video_id=f"bmd_{matrix_name}",
            source_dataset=self.name,
            media_uri=media_uri,
            duration_s=NOMINAL_DURATION_S,
            domain=self.default_domain,
            raw_labels=raw_labels,
            metadata=metadata,
        )
