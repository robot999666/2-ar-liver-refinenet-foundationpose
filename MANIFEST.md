# Reproducibility Manifest

This file distinguishes version-controlled project files from large runtime assets.

## Version-Controlled Model

| Path | Purpose | SHA256 |
| --- | --- | --- |
| `weights/refinenet_patient1_02_best.pth` | Curated rigid RefineNet checkpoint | `3B84F1E992ED836915018857337C5F180A11371B55C45F02BDEF4302E0B5C307` |

## Required Runtime Assets

The following large files must exist locally before running all pipeline stages:

| Path | Purpose | Restore method |
| --- | --- | --- |
| `weights/depth_anything_v2_vitl.pth` | Depth Anything V2 Large checkpoint used by scripts 04, 05, and 07 | Download from the official Depth Anything V2 Large release |
| `archives/wheelhouse.zip` | Offline Linux/Python 3.12 training dependencies | Regenerate from `requirements_06_train_refine_net.txt` |

After either file is restored, record its byte size and SHA256 in this manifest before use.

## Source Data Scope

The current repository contains Patient1 source files required by scripts 01 and 08:

- laparoscopic frames and camera parameters;
- LUS calibration and pose `.mat` files;
- LUS segmentation JSON files used by Python TRE/IC evaluation;
- contour XML annotations;
- liver and tumour OBJ models.

Generated `data_case/` content and generated `result/` content are intentionally excluded from Git.

