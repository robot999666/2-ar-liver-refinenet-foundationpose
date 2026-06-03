# Current Best Inference Profile

Case tested: `Patient1/02`

Current infer profile: `coarse0p30_refine0p10_rawstop`

Parameters baked into `scripts/07_infer_export_stl.py`:

```text
AR_MAX_ITER=10
AR_TRANS_DAMPING=0.4
AR_ROT_DAMPING=0.4
AR_FIRST_STEP_DAMPING=0.75
AR_REFINE_TRANS_DAMPING=0.1
AR_REFINE_ROT_DAMPING=0.1
AR_RAW_WORSEN_PATIENCE=1
AR_RAW_WORSEN_MIN_DELTA=0.5
AR_RAW_SCORE_ROT_WEIGHT=5.0
AR_SELECT_BEST_BY=last
```

Inference logic:

1. Apply a larger coarse first step: `0.4 * 0.75 = 0.30` of the network-predicted offset.
2. Continue with small refinement steps: `0.10` of each later predicted offset.
3. Stop when the model residual proxy stops improving:

```text
raw_score = raw_trans_mm + 5.0 * raw_rot_deg
```

This stopping rule does not use MATLAB, TRE, IC, or any ground-truth evaluation result during inference. Python/MATLAB evaluation is only used afterward to verify the trajectory.

Latest verification on `Patient1/02`:

```text
iter00 TRE 25.7714, IC 0
iter01 TRE 12.6766, IC 1
iter04 TRE 11.3079, IC 1
iter05 TRE 10.0375, IC 1
iter06 TRE  8.7966, IC 1
```

Final selected output is `iter06`, stopped by `raw_score_no_improve`.
