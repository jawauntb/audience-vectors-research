# Attention-Capture TRIBE Full-Preflight Audit

## Verdict

- OK: True
- App: audience-vectors-dev
- Media path: /bmd-videos/generated/bo_memorability_replay/bo_replay_00_sobol_prompt_search_518_slot18_rep00.mp4
- Event mode: full
- Claim boundary: This verifies that the deployed Modal TRIBE path can construct events for one video. It does not validate attentional capture, does not score Phase 1, and does not replace external labels.

## Preflight

- Resolved path exists: True
- Duration seconds: 3.5720
- Event rows: 1
- Event columns: type, start, duration, timeline, subject, session, task, run, filepath, frequency, offset, stop, context

## Step Seconds

| step | seconds |
|---|---:|
| volume_reload | 0.2542 |
| resolve_local_path | 0.0000 |
| probe_duration | 2.0001 |
| ensure_suffix | 0.0000 |
| get_events_dataframe | 0.7629 |
