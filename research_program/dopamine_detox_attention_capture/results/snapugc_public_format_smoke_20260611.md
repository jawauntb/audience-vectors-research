# SnapUGC Public-Format Smoke

## Verdict

- Smoke passed: True
- Claim blocked: True
- Claim blocking reason: The default scores source is a public baseline/prediction file, not granted behavioral ECR labels.
- Builder ready: True
- Label audit ready: True
- Baseline ready: True
- Canonical rows: 10
- Best baseline feature: `download_link_char_count`
- Best baseline rho: 0.5338

## Sources

- Metadata: https://raw.githubusercontent.com/dasongli1/SnapUGC_Engagement/main/ECR_inference/dataset/val_data_sample.csv
- Scores: https://raw.githubusercontent.com/dasongli1/SnapUGC_Engagement/main/ECR_inference/submission_baseline.csv

## Blocking Reasons

- none

## Warnings

- none

## Next Actions

- Do not use this smoke report for claims.
- Replace the scores source with granted behavioral ECR labels, then rerun build_snapugc_retention_labels.py without --allow-prediction-score-file.
- If the real-label audits pass, run a small full-mode Modal TRIBE slice from the canonical CSV.
