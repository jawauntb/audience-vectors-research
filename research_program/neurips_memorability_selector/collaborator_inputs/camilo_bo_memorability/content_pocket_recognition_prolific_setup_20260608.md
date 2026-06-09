# Content-Pocket Recognition Prolific Setup

Date: 2026-06-09T00:05:55Z

## Status

- Status: `ready_for_final_prolific_configuration`
- Session pages are public GitHub Pages URLs.
- A sampled MP4 URL was verified as public `video/mp4`.
- This artifact is launch setup only, not human-memory evidence.

## Study URLs

Session 1 base URL:

`https://jawauntb.github.io/audience-vectors-research/session1.html`

Session 2 base URL:

`https://jawauntb.github.io/audience-vectors-research/session2.html`

Use these Prolific URL templates, replacing
`<YOUR_HTTPS_RESPONSE_ENDPOINT>` with the response collection endpoint:

```text
https://jawauntb.github.io/audience-vectors-research/session1.html?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}&submit_url=<YOUR_HTTPS_RESPONSE_ENDPOINT>
```

```text
https://jawauntb.github.io/audience-vectors-research/session2.html?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}&submit_url=<YOUR_HTTPS_RESPONSE_ENDPOINT>
```

If the endpoint itself has query parameters, URL-encode the endpoint value
before placing it after `submit_url=`.

## Completion Codes

- Session 1: `CPR_SESSION1_DONE`
- Session 2: `CPR_SESSION2_DONE`

## Participant Plan

- Recommended Session 1 slots: 350.
- Target Session 2 usable participants: 300.
- Minimum Session 2 usable participants before interpreting the memory gate:
  200.
- Session 2 should invite only Session 1 completers using the same
  `PROLIFIC_PID`.
- The delay between sessions should be 24-48 hours.
- Any optional link/video/endpoint shakedown rows must be excluded from the
  evidence gate.

## Response Capture

The HTML records Prolific query parameters, hashes `PROLIFIC_PID` into one of
six sparse forms, and supports automatic JSON POST collection through
`submit_url` or `endpoint`.

Session 1 rows must retain participant ID, form ID, target ID, arm, analysis
group, video URL, cover-task rating, exposure-completion status, and
timestamps. Session 1 also records `media_error` for failed video loads.

Session 2 rows must retain participant ID, form ID, target ID, arm, analysis
group, old/lure URLs, old side, choice side, correctness, response time, and
timestamps. Session 2 also records `left_media_error`, `right_media_error`, and
`any_media_error` for failed video loads.

Excluded participants and failed-load trials must remain in the exported
dataset with explicit exclusion reasons.

## Claim Boundary

- Do not call this setup a human memorability result.
- Do not interpret the gate before delayed Session 2 reaches at least 200
  usable participants.
- The pass/fail claim depends on primary-positive old-vs-lure recognition
  exceeding hard-negative-control recognition, with positive directions for
  both orange flowers and hanging clothes.
