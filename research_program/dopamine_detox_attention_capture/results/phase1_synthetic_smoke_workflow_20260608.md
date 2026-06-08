# Phase 1 Capture-Score Workflow

## Verdict

- Manifest: research_program/dopamine_detox_attention_capture/phase1_synthetic_smoke_manifest_20260608.json
- Mechanical ready: True
- Claim update allowed: False
- Claim ready: False
- Scoring executed: True
- Decision reason: claim_blocked_scored_for_diagnostic_only
- Claim boundary: This workflow only allows claim-relevant scoring after preflight passes. Smoke or control manifests may be scored for diagnostics only when explicitly requested, and remain claim-blocked.

## Preflight Blocks

- none

## Preflight Warnings

- manifest status or dataset names block claim updates

## Primary Score

- Claim validated: False
- Gate passed groups: DHF1K_fixture, SnapUGC_fixture
- Invalid capture denominators: 0

| group | n valid | capture rho | permutation p | gate |
|---|---:|---:|---:|---|
| DHF1K_fixture | 8 | 1.0000 | 0.0010 | True |
| SnapUGC_fixture | 8 | 1.0000 | 0.0010 | True |
| pooled | 16 | 0.9882 | 0.0010 | True |

## Sensitivity Delta

| group | sensitivity | primary rho | sensitivity rho | delta |
|---|---|---:|---:|---:|
| DHF1K_fixture | overlapping | 1.0000 | 1.0000 | 0.0000 |
| SnapUGC_fixture | overlapping | 1.0000 | 1.0000 | 0.0000 |
| pooled | overlapping | 0.9882 | 0.9882 | 0.0000 |
