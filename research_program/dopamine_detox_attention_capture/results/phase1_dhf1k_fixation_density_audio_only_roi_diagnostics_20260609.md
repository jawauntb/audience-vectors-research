# DHF1K Fixation-Density Audio-Only ROI Diagnostics

Exploratory after primary gate failure; not claim-updating metric selection.

## disjoint masks

| metric | n | rho | p(two-sided) |
|---|---:|---:|---:|
| V1 | 350 | -0.0929 | 0.08278 |
| language | 350 | -0.0882 | 0.09954 |
| sensory_mean | 350 | -0.0849 | 0.1127 |
| frontoparietal | 350 | -0.0848 | 0.1131 |
| neg_frontoparietal | 350 | 0.0848 | 0.1131 |
| PPA | 350 | 0.0631 | 0.239 |
| absden_capture_score | 350 | -0.0608 | 0.2569 |
| v1_ppa_mean | 350 | -0.0586 | 0.2739 |
| capture_score | 302 | -0.0348 | 0.5473 |
| v1_ppa_delta | 350 | 0.0215 | 0.6879 |
| v1_ppa_absden_score | 350 | -0.0189 | 0.7245 |
| capture_delta | 350 | 0.0164 | 0.7592 |

### Tail means

| tail | n | ground_truth | frontoparietal | -frontoparietal | sensory_mean | capture_delta | absden_capture_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 175 | 0.000655 | 0.0549 | -0.0549 | 0.0421 | -0.0129 | 0.7348 |
| high | 175 | 0.000833 | 0.0470 | -0.0470 | 0.0353 | -0.0117 | -0.0124 |

## overlap masks

| metric | n | rho | p(two-sided) |
|---|---:|---:|---:|
| language | 350 | -0.0912 | 0.08844 |
| frontoparietal | 350 | -0.0862 | 0.1074 |
| neg_frontoparietal | 350 | 0.0862 | 0.1074 |
| V1 | 350 | -0.0661 | 0.2176 |
| sensory_mean | 350 | -0.0487 | 0.364 |
| v1_ppa_delta | 350 | 0.0308 | 0.5658 |
| capture_delta | 350 | 0.0299 | 0.5773 |
| v1_ppa_mean | 350 | -0.0264 | 0.6228 |
| absden_capture_score | 350 | -0.0261 | 0.6266 |
| capture_score | 301 | 0.0245 | 0.6717 |
| PPA | 350 | 0.0225 | 0.6745 |
| v1_ppa_absden_score | 350 | -0.0158 | 0.7677 |

### Tail means

| tail | n | ground_truth | frontoparietal | -frontoparietal | sensory_mean | capture_delta | absden_capture_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 175 | 0.000655 | 0.0471 | -0.0471 | 0.0614 | 0.0143 | 4.4078 |
| high | 175 | 0.000833 | 0.0402 | -0.0402 | 0.0546 | 0.0144 | 4.2748 |
