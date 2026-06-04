# Reproducibility Manifest

This file distinguishes version-controlled project files from large runtime assets.

## Version-Controlled Model

| Path | Purpose | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `weights/refinenet_patient1_02_best.pth` | Curated rigid RefineNet checkpoint | `87866632` | `3B84F1E992ED836915018857337C5F180A11371B55C45F02BDEF4302E0B5C307` |

## Required Runtime Assets

The following large files are intentionally ignored by Git. They must be
backed up separately and restored before the relevant stages are run.

| Path | Purpose | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `weights/depth_anything_v2_vitl.pth` | Official Depth Anything V2 Large checkpoint used by scripts 04, 05, and 07 | `1341395338` | `A7EA19FA0ED99244E67B624C72B8580B7E9553043245905BE58796A608EB9345` |
| `archives/wheelhouse.zip` | Offline Linux x86_64 / Python 3.12 training dependencies | `632832579` | `02AC2F551C9AD55034ACE3C740E2110F55CE0B2B37F03EA0765F09DD1D98C255` |

`archives/wheelhouse.zip` contains 78 wheels under the top-level
`wheelhouse/` directory. It is for the documented cloud Python 3.12 workflow,
not for the Python 3.10 environment built by `Dockerfile`.

Verify restored files before use:

```powershell
Get-FileHash weights\depth_anything_v2_vitl.pth -Algorithm SHA256
Get-FileHash archives\wheelhouse.zip -Algorithm SHA256
python -m zipfile -t archives\wheelhouse.zip
```

## Source Data Scope

The current repository contains the Patient1 source files required by scripts
01 and 08:

- laparoscopic frames and camera parameters;
- LUS calibration and pose `.mat` files;
- LUS segmentation JSON files used by Python TRE/IC evaluation;
- contour XML annotations;
- liver and tumour OBJ models.

Additional published Patient1 source files that are not currently read by the
rigid pipeline are retained for audit and future validation. They must not be
confused with generated `data_case/` artifacts.

Complete source-input sets are present for frames `02`, `03`, `04`, `06`,
`07`, `08`, `09`, and `10`. Frame `05` is absent from the provided source
subset and is reported as incomplete by script 08.

Generated `data_case/` content and generated `result/` content are intentionally excluded from Git.
This includes manual mask bundles produced by script 01b. The clean repository
does not ship those annotations; create and externally back them up before
running scripts 05, 07, or 08.
