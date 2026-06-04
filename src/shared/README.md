# Shared Modules

| Module | Purpose |
| --- | --- |
| `case_config.py` | Central paths, patient/frame defaults, seeds, weights, and result layout. |
| `da2_engine.py` | Depth Anything V2 loading and inference. |
| `depth_augment.py` | Shared depth normalization and training augmentation. |
| `evaluation.py` | Python TRE/IC evaluation used by script 08. |
| `intraop_masks.py` | Real-frame mask semantics, validation, provenance, and QA overlays. |
| `pose_render.py` | Open3D raycast rendering for contours, masks, and DA2 RGB inputs. |

Common overrides include `PATIENT_ID`, `FRAME_ID`, `AR_SEED`,
`AR_WEIGHT_PATH`, `CASE_ROOT`, `DATA_ROOT`, `RESULT_ROOT`, and `WEIGHTS_ROOT`.
