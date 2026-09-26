# A0 utility cross-seed stability

E3 node tables were reused directly. Only rows marked train or val were included; no E3 retraining and no test labels or metrics were used. Same-split summaries require at least 100 matched nodes.

## Results

- Movies / G_T_given_VU: common Train+Validation n=3604; mean pairwise Spearman=0.677, sign agreement=0.771, unanimous sign=0.656, top-10% Jaccard=0.416, bottom-10% Jaccard=0.400, between/within variance=5.240; same-split n=3604.
- Movies / G_V_given_TU: common Train+Validation n=3604; mean pairwise Spearman=0.628, sign agreement=0.760, unanimous sign=0.640, top-10% Jaccard=0.365, bottom-10% Jaccard=0.365, between/within variance=3.689; same-split n=3604.
- Toys / G_T_given_VU: common Train+Validation n=4494; mean pairwise Spearman=0.660, sign agreement=0.798, unanimous sign=0.697, top-10% Jaccard=0.394, bottom-10% Jaccard=0.470, between/within variance=3.966; same-split n=4494.
- Toys / G_V_given_TU: common Train+Validation n=4494; mean pairwise Spearman=0.677, sign agreement=0.807, unanimous sign=0.711, top-10% Jaccard=0.425, bottom-10% Jaccard=0.470, between/within variance=4.286; same-split n=4494.
- Grocery / G_T_given_VU: common Train+Validation n=3606; mean pairwise Spearman=0.679, sign agreement=0.803, unanimous sign=0.704, top-10% Jaccard=0.420, bottom-10% Jaccard=0.454, between/within variance=4.036; same-split n=3606.
- Grocery / G_V_given_TU: common Train+Validation n=3606; mean pairwise Spearman=0.666, sign agreement=0.814, unanimous sign=0.720, top-10% Jaccard=0.359, bottom-10% Jaccard=0.439, between/within variance=2.842; same-split n=3606.
- ele-fashion / G_T_given_VU: common Train+Validation n=58659; mean pairwise Spearman=0.780, sign agreement=0.872, unanimous sign=0.808, top-10% Jaccard=0.546, bottom-10% Jaccard=0.595, between/within variance=10.918; same-split n=58659.
- ele-fashion / G_V_given_TU: common Train+Validation n=58659; mean pairwise Spearman=0.742, sign agreement=0.841, unanimous sign=0.762, top-10% Jaccard=0.494, bottom-10% Jaccard=0.607, between/within variance=6.590; same-split n=58659.
- Reddit-S / G_T_given_VU: common Train+Validation n=3438; mean pairwise Spearman=0.610, sign agreement=0.761, unanimous sign=0.646, top-10% Jaccard=0.431, bottom-10% Jaccard=0.528, between/within variance=2.770; same-split n=3438.
- Reddit-S / G_V_given_TU: common Train+Validation n=3438; mean pairwise Spearman=0.743, sign agreement=0.797, unanimous sign=0.703, top-10% Jaccard=0.605, bottom-10% Jaccard=0.605, between/within variance=6.901; same-split n=3438.
