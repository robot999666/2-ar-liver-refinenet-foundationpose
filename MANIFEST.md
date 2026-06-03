# Manifest

Minimal reproducible package for the Patient1 liver AR refinement experiment.

## Kept

- `scripts/01_prepare_case.py` to `scripts/07_infer_export_stl.py`
- `scripts/11_batch_patient_frames.py`
- `scripts/_eval_debug_tre_python.py` as an internal dependency for post-hoc TRE/IC tables
- `src/` runtime libraries: renderer, shared config, Depth Anything V2 source, RefineNet
- `src_data/Dataset/Patient1/` original Patient1 inputs
- `src_data/contours_and_models/Patient1/` meshes and contour annotations
- `src_data/*.m` and `src_data/stlTools/` MATLAB evaluation dependencies
- `result/Patient1/02/best.pth` best model checkpoint
- `result/Patient1/batch_02_10/` final result tables and small logs
- `configs/repro.env` saved seed and inference/training parameters

## Removed

- `data_case/` generated case workspace, rendered samples, DA2 depth caches, predictions, and debug STLs
- `result/Patient1/02/last.pth`, `best_full.pth`, plots, and old training logs not needed for reproduction
- old sweeps, archives, history snapshots, pycache files
- Depth Anything V2 `weights/depth_anything_v2_vitl.pth` because it is 1.34 GB

## Key Hashes

- `result/Patient1/02/best.pth`
  - SHA256: `3B84F1E992ED836915018857337C5F180A11371B55C45F02BDEF4302E0B5C307`
- `result/Patient1/batch_02_10/summary.csv`
  - SHA256: `957FCC89353D80FA4CD78355CB800B1CB40388506678DC7350A88DA89A08F0F2`
- `result/Patient1/batch_02_10/average.json`
  - SHA256: `C7194446B6A2AE3DBE5F9A4BA188563C34625476E49071F5109F2EE3046116CF`
