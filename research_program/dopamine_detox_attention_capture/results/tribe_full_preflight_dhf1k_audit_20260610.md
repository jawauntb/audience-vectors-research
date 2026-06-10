# Attention-Capture TRIBE Full-Preflight Audit

## Verdict

- OK: True
- App: audience-vectors-dev
- Media path: /bmd-videos/attention_capture/DHF1K/video/003.AVI
- Event mode: full
- Claim boundary: This verifies that the deployed Modal TRIBE path can construct events for one video. It does not validate attentional capture, does not score Phase 1, and does not replace external labels.

## Preflight

- Resolved path exists: True
- Duration seconds: 14.9333
- Event rows: 50
- Event columns: type, start, duration, timeline, subject, session, task, run, filepath, frequency, offset, stop, text, sequence_id, sentence, language, sentence_char, text_char, context, modality

## Step Seconds

| step | seconds |
|---|---:|
| volume_reload | 0.0524 |
| resolve_local_path | 0.0000 |
| probe_duration | 0.4654 |
| ensure_suffix | 0.0000 |
| get_events_dataframe | 118.3221 |
