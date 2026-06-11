from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_snapugc_public_format_smoke_cli_with_local_csvs(tmp_path: Path) -> None:
    metadata = tmp_path / "val_data_sample.csv"
    scores = tmp_path / "submission_baseline.csv"
    output_json = tmp_path / "smoke.json"
    output_md = tmp_path / "smoke.md"
    metadata.write_text(
        (
            "Id,Title,Description,Download_link\n"
            "snap_a,A title,A description,https://example.test/a.mp4\n"
            "snap_b,B title,B description,https://example.test/b.mp4\n"
            "snap_c,C title,C description,https://example.test/c.mp4\n"
            "snap_d,D title,D description,https://example.test/d.mp4\n"
        ),
        encoding="utf-8",
    )
    scores.write_text(
        "Id,ECR\nsnap_a,0.1\nsnap_b,0.3\nsnap_c,0.6\nsnap_d,0.9\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_snapugc_public_format_smoke.py",
            "--metadata-csv-or-url",
            str(metadata),
            "--scores-csv-or-url",
            str(scores),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--min-samples",
            "4",
            "--min-distinct-ecr",
            "4",
        ],
        check=True,
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["experiment"] == "snapugc_public_format_smoke"
    assert report["smoke_passed"] is True
    assert report["claim_blocked"] is True
    assert report["counts"]["canonical_rows"] == 4
    assert "Do not use this smoke report for claims" in output_md.read_text(
        encoding="utf-8"
    )
