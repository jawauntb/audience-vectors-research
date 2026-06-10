# DHF1K Audio-Only ROI Diagnostics

Exploratory after primary gate failure; not claim-updating metric selection.

## disjoint masks

| metric | n | rho | p(two-sided) |
|---|---:|---:|---:|
| language | 350 | -0.3902 | 3.544e-14 |
| v1_ppa_delta | 350 | 0.2829 | 7.248e-08 |
| frontoparietal | 350 | -0.2308 | 1.288e-05 |
| neg_frontoparietal | 350 | 0.2308 | 1.288e-05 |
| v1_ppa_absden_score | 350 | 0.2275 | 1.732e-05 |
| capture_delta | 350 | 0.2150 | 5.019e-05 |
| PPA | 350 | 0.1894 | 0.0003669 |
| capture_score | 301 | 0.1256 | 0.02933 |
| v1_ppa_mean | 350 | 0.0822 | 0.1249 |
| sensory_mean | 350 | -0.0520 | 0.3317 |
| absden_capture_score | 350 | 0.0338 | 0.5288 |
| V1 | 350 | 0.0311 | 0.562 |

### Tail means

| tail | n | ground_truth | frontoparietal | -frontoparietal | sensory_mean | capture_delta | absden_capture_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 175 | 0.0138 | 0.0577 | -0.0577 | 0.0361 | -0.0215 | 0.7273 |
| high | 175 | 0.0191 | 0.0336 | -0.0336 | 0.0313 | -0.0023 | 0.4755 |

## overlap masks

| metric | n | rho | p(two-sided) |
|---|---:|---:|---:|
| language | 350 | -0.3830 | 1.123e-13 |
| v1_ppa_delta | 350 | 0.3143 | 1.838e-09 |
| capture_delta | 350 | 0.2942 | 2.027e-08 |
| v1_ppa_absden_score | 350 | 0.2753 | 1.666e-07 |
| capture_score | 296 | 0.2590 | 6.369e-06 |
| PPA | 350 | 0.2554 | 1.288e-06 |
| frontoparietal | 350 | -0.2209 | 3.047e-05 |
| neg_frontoparietal | 350 | 0.2209 | 3.047e-05 |
| absden_capture_score | 350 | 0.2006 | 0.0001587 |
| v1_ppa_mean | 350 | 0.1819 | 0.0006256 |
| V1 | 350 | 0.1005 | 0.0603 |
| sensory_mean | 350 | 0.0869 | 0.1045 |

### Tail means

| tail | n | ground_truth | frontoparietal | -frontoparietal | sensory_mean | capture_delta | absden_capture_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 175 | 0.0138 | 0.0493 | -0.0493 | 0.0479 | -0.0015 | 1.9624 |
| high | 175 | 0.0191 | 0.0281 | -0.0281 | 0.0580 | 0.0299 | 5.4557 |
