# Attention-Capture Modal Asset Audit

## Verdict

- Retention labels maybe available: False
- External dataset dirs maybe available: False
- Feature caches maybe available: True
- Full multimodal token env present: False
- Claim boundary: This Modal CPU audit checks remote asset availability only. It does not score TRIBE features or validate attentional capture.

## Blocking Reasons

- no Modal-hosted SnapUGC/VQualA retention label candidate found
- no Modal secret exposes a HuggingFace token env name

## Secret Presence

- Secrets checked: underlying-analyzer-env, llm-api-keys
- Token envs checked: HF_TOKEN, HUGGINGFACE_TOKEN, HUGGINGFACE_HUB_TOKEN
- Matching env names: none

## Volume Summary

| volume | entries | files | dirs | truncated | labels | datasets | features |
|---|---:|---:|---:|---|---:|---:|---:|
| rde-activation-results | 3 | 2 | 1 | False | 0 | 0 | 3 |
| audience-analyzer-runs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| wan22-lora-data-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| wan22-lora-outputs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| svd-outputs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| wan22-outputs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| cogvideox-outputs-v1 | 0 | 0 | 0 | False | 0 | 0 | 0 |
| bmd-videos-v1 | 314 | 312 | 2 | False | 0 | 0 | 0 |
| fr-dev-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| fr-prd-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| fr-stg-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| flytrap-review-prod-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| flytrap-review-data | 0 | 0 | 0 | False | 0 | 0 | 0 |
| tac-docker-data | 0 | 0 | 0 | False | 0 | 0 | 0 |

## Label Candidates

| volume | path | kind | claim blocked |
|---|---|---|---|
| none | n/a | n/a | False |

## Dataset Candidates

| volume | path | kind | claim blocked |
|---|---|---|---|
| none | n/a | n/a | False |

## Feature Candidates

| volume | path | kind | claim blocked |
|---|---|---|---|
| rde-activation-results | activation_geometry | dir | False |
| rde-activation-results | activation_geometry/pythia160_layer3_pocket_scale_sweep_seed20260610_raw.json | file | False |
| rde-activation-results | activation_geometry/pythia160_layer3_pocket_seed20260610_raw.json | file | False |
