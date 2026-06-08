# Destrieux ROI Mask Audit

- Atlas: nilearn.fetch_atlas_surf_destrieux fsaverage5
- Overlap policy: allow
- Vertices: 20484
- Labels: 152
- Claim boundary: Exploratory ROI masks for proxy dry runs. These masks are not validated attention, dopamine, or executive-control measurements.

## ROI Coverage

| ROI | vertices | fraction | selected labels | mask sha256 |
|---|---:|---:|---:|---|
| V1 | 2217 | 0.1082 | 18 | `1f9c0dd2653c` |
| PPA | 866 | 0.0423 | 6 | `9840dc3ddaea` |
| language | 4142 | 0.2022 | 30 | `5688d4493017` |
| frontoparietal | 4104 | 0.2004 | 18 | `106fa9e4360b` |

## Selected Labels

### V1

- `L_G_and_S_occipital_inf`
- `L_G_occipital_middle`
- `L_G_occipital_sup`
- `L_G_oc-temp_med-Lingual`
- `L_Pole_occipital`
- `L_S_calcarine`
- `L_S_occipital_ant`
- `L_S_oc-temp_med_and_Lingual`
- `L_S_parieto_occipital`
- `R_G_and_S_occipital_inf`
- `R_G_occipital_middle`
- `R_G_occipital_sup`
- `R_G_oc-temp_med-Lingual`
- `R_Pole_occipital`
- `R_S_calcarine`
- `R_S_occipital_ant`
- `R_S_oc-temp_med_and_Lingual`
- `R_S_parieto_occipital`

### PPA

- `L_G_oc-temp_med-Lingual`
- `L_G_oc-temp_med-Parahip`
- `L_S_oc-temp_med_and_Lingual`
- `R_G_oc-temp_med-Lingual`
- `R_G_oc-temp_med-Parahip`
- `R_S_oc-temp_med_and_Lingual`

### language

- `L_G_front_inf-Opercular`
- `L_G_front_inf-Orbital`
- `L_G_front_inf-Triangul`
- `L_G_front_middle`
- `L_G_temporal_inf`
- `L_G_temporal_middle`
- `L_Pole_temporal`
- `L_S_circular_insula_ant`
- `L_S_circular_insula_inf`
- `L_S_circular_insula_sup`
- `L_S_front_inf`
- `L_S_front_middle`
- `L_S_temporal_inf`
- `L_S_temporal_sup`
- `L_S_temporal_transverse`
- `R_G_front_inf-Opercular`
- `R_G_front_inf-Orbital`
- `R_G_front_inf-Triangul`
- `R_G_front_middle`
- `R_G_temporal_inf`
- `R_G_temporal_middle`
- `R_Pole_temporal`
- `R_S_circular_insula_ant`
- `R_S_circular_insula_inf`
- `R_S_circular_insula_sup`
- `R_S_front_inf`
- `R_S_front_middle`
- `R_S_temporal_inf`
- `R_S_temporal_sup`
- `R_S_temporal_transverse`

### frontoparietal

- `L_G_front_middle`
- `L_G_front_sup`
- `L_G_parietal_sup`
- `L_G_precentral`
- `L_S_front_middle`
- `L_S_front_sup`
- `L_S_intrapariet_and_P_trans`
- `L_S_precentral-inf-part`
- `L_S_precentral-sup-part`
- `R_G_front_middle`
- `R_G_front_sup`
- `R_G_parietal_sup`
- `R_G_precentral`
- `R_S_front_middle`
- `R_S_front_sup`
- `R_S_intrapariet_and_P_trans`
- `R_S_precentral-inf-part`
- `R_S_precentral-sup-part`

## Vertex Overlaps

| ROI | V1 | PPA | language | frontoparietal |
|---|---:|---:|---:|---:|
| V1 | 2217 | 598 | 0 | 0 |
| PPA | 598 | 866 | 0 | 0 |
| language | 0 | 0 | 4142 | 742 |
| frontoparietal | 0 | 0 | 742 | 4104 |
