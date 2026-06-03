# Shared Modules

These modules support the scripts in `scripts/`.

| Module | Purpose |
| --- | --- |
| `case_config.py` | Central project paths and patient/frame defaults. |
| `da2_engine.py` | Depth Anything V2 loading and inference wrapper. |
| `depth_augment.py` | Depth preprocessing and training augmentation. |
| `pose_render.py` | Open3D raycast rendering for contours, masks, and DA2 RGB inputs. |

Edit `case_config.py` or environment variables such as `PATIENT_ID`,
`FRAME_ID`, `CASE_ROOT`, `DATA_ROOT`, `RESULT_ROOT`, `DA2_PROJECT_DIR`, and
`DA2_CHECKPOINT_PATH` to switch cases or paths.
