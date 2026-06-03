# Patient1 02-10 Batch Result

This folder stores the batch preprocessing, inference, and post-hoc evaluation results for Patient1 frames 02-10.

- `summary.csv`: one row per requested frame. The average uses `selected_TRE_mm` only.
- `debug_all_frames.csv`: post-hoc TRE/IC for every debug iteration STL.
- `average.json`: mean of script-selected final outputs only.
- `debug_eval/`: per-frame debug iteration TRE/IC tables and inference histories.
- `final_eval/`: per-frame final text reports for script-selected outputs.
- `logs/`: preprocessing, depth, and inference logs.
- `skipped_frames.csv`: frames skipped because required source inputs were missing.

Inference selection is done by the inference script's raw update score and stop rule. TRE/IC is computed after inference only, so `posthoc_best_*` is reported for diagnosis but is not used to select the final output.
