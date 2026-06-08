# Content-Pocket Validation Stimulus Freeze

Date: 2026-06-08T21:40:57+00:00

## Status

Prelaunch stimulus freeze only. These MP4s are proxy-selected and have
not yet cleared a human memorability or measured-BMD gate.

Primary analysis remains V-JEPA-caveated: orange flowers and hanging
clothes are TRIBE/V-JEPA compute-proxy candidates, while generated-video
CLIP did not pass prospectively. Blue jellyfish and old car are
exploratory boundary arms because exact V-JEPA did not pass their
boundary audit.

## Frozen Task Pool

| comparison | tasks |
|---|---:|
| `exploratory_boundary_content_pocket_vs_hard_negative` | 12 |
| `primary_content_pocket_vs_hard_negative` | 12 |

Unique MP4 paths: 45
Task payload SHA-256: `b151326f1e120d7d6c6440c97e9341784bb3b25a1393b8ca7a8481fbcb3cef6c`
Video path set SHA-256: `8e18aa36d23cf65142c08cdf61748c1b9403298ce55692fc6a169c0ea7804072`

## Selected Candidates

| tier | pocket | task | recipe | reps | mean TRIBE | min | max |
|---|---|---|---:|---:|---:|---:|---:|
| primary | `fresh24_orange_flowers` | `sobol_prompt_search_519_slot10` | 519 | 3 | 4.1035 | 3.7013 | 4.4512 |
| primary | `fresh24_orange_flowers` | `sobol_prompt_search_520_slot10` | 520 | 3 | 4.0783 | 3.8722 | 4.3062 |
| primary | `fresh24_hanging_clothes` | `sobol_prompt_search_521_slot12` | 521 | 3 | 3.6030 | 3.1999 | 3.9995 |
| primary | `fresh24_hanging_clothes` | `sobol_prompt_search_520_slot12` | 520 | 3 | 3.3852 | 2.4989 | 4.1540 |
| exploratory_boundary | `fresh24_blue_jellyfish` | `sobol_prompt_search_519_slot18` | 519 | 3 | 2.6652 | 2.2628 | 3.0441 |
| exploratory_boundary | `fresh24_blue_jellyfish` | `sobol_prompt_search_521_slot18` | 521 | 3 | 2.3349 | 2.1152 | 2.5229 |
| exploratory_boundary | `fresh24_old_car` | `sobol_prompt_search_522_slot00` | 522 | 3 | 1.5023 | 1.4442 | 1.5976 |
| exploratory_boundary | `fresh24_old_car` | `sobol_prompt_search_518_slot00` | 518 | 3 | 1.4943 | 1.3995 | 1.5894 |

## Control Matching

Each positive replicate is paired with the hard-negative control from the
same Sobol recipe index and stochastic replicate index:

- `rep00` -> `fresh24_aerial_beach`
- `rep01` -> `fresh24_city_street`
- `rep02` -> `fresh24_storm_beach`

## Output Artifacts

- `content_pocket_validation_stimuli_manifest_20260608.json`
- `content_pocket_validation_pairwise_tasks_20260608.json`
- `content_pocket_validation_prolific_survey_20260608.html`

## Launch Blockers

- Human/IRB approval and participant compensation are not recorded in this artifact.
- Every selected MP4 must be manually screened before launch.
- Hosted HTTPS URLs must replace local data-lake paths before Prolific launch.
- The packet remains proxy-selected until human or measured-BMD results clear.

## Next Action

Manually screen the frozen MP4s, host the screened videos at stable HTTPS
URLs, then build the blinded forced-choice survey from the frozen task
JSON. Keep any BMD/measured-brain transfer report on the same frozen
stimulus set so the human and BMD gates adjudicate identical clips.
