from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "extract_attention_capture_tribe_features.py"
    )
    spec = importlib.util.spec_from_file_location(
        "extract_attention_capture_tribe_features",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_jobs_from_csv_builds_expected_output_paths(tmp_path: Path) -> None:
    module = load_module()
    source_csv = tmp_path / "labels.csv"
    output_dir = tmp_path / "features"
    source_csv.write_text(
        "sample_id,video_path\ns1,/tmp/s1.mp4\ns2,/tmp/s2.mp4\n",
        encoding="utf-8",
    )

    jobs = module.load_jobs_from_csv(source_csv=source_csv, output_dir=output_dir)

    assert [job.sample_id for job in jobs] == ["s1", "s2"]
    assert jobs[0].media_path == "/tmp/s1.mp4"
    assert jobs[0].output_path == output_dir / "s1.npz"


def test_result_to_arrays_accepts_object_and_mapping_results() -> None:
    module = load_module()
    obj = SimpleNamespace(frames=[1.0, 2.0, 3.0], duration_seconds=2.5)
    mapping = {
        "frames": [[1.0, 2.0], [3.0, 4.0]],
        "duration_seconds": 1.25,
    }

    obj_frames, obj_duration = module.result_to_arrays(obj)
    map_frames, map_duration = module.result_to_arrays(mapping)

    assert obj_frames.dtype == np.float32
    assert obj_frames.shape == (1, 3)
    assert obj_duration == 2.5
    assert map_frames.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert map_duration == 1.25


def test_extract_one_passes_audio_only_mode_and_records_metadata(tmp_path: Path) -> None:
    module = load_module()
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake-video")
    job = module.VideoFeatureJob(
        sample_id="s1",
        media_path=str(media),
        output_path=tmp_path / "features" / "s1.npz",
    )

    class FakeService:
        calls: list[dict[str, object]]

        def __init__(self) -> None:
            self.calls = []

        async def predict_video_bytes(
            self,
            video_bytes: bytes,
            suffix: str = ".mp4",
            *,
            audio_only: bool = False,
        ):
            self.calls.append(
                {
                    "video_bytes": video_bytes,
                    "suffix": suffix,
                    "audio_only": audio_only,
                }
            )
            return SimpleNamespace(frames=[[1.0, 2.0]], duration_seconds=1.5)

    service = FakeService()

    output = asyncio.run(
        module.extract_one(
            service=service,
            sem=asyncio.Semaphore(1),
            job=job,
            transport="bytes",
            event_mode="audio-only",
        )
    )

    assert output == job.output_path
    assert service.calls == [
        {"video_bytes": b"fake-video", "suffix": ".mp4", "audio_only": True}
    ]
    payload = np.load(job.output_path)
    assert payload["event_mode"].item() == "audio-only"
    assert payload["transport"].item() == "bytes"
