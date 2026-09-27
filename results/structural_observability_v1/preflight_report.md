# E7-CSAO preflight

C0 was reloaded from the frozen E6 reference checkpoints. No C0 parameters were trained; all Train/Validation action labels came from the verified E6 losses. Test indices and labels were not read.

Screening contexts checked: 9/9. E6 node ids, split order, raw action landscape, and D0 C0 checkpoint hashes matched. Maximum 5x5 action CE recomputation error: 2.28882e-05.
The recomputation uses the exact E6 action definitions and 32,768-node chunks; small remaining float32 differences arise from repeated graph aggregation (largest per-run P99: 7.15256e-07). E6 NPZ/raw values were cross-checked within serialization precision.
E6 corrected restart audit was recomputed from within-chunk, three-restart groups; E6 source outputs were left unchanged.
