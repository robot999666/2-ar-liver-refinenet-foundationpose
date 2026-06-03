# External Weights

This repository stores `result/Patient1/02/best.pth` directly. It is about 87.9 MB and stays below GitHub's 100 MB per-file hard limit.

Depth Anything V2 is not committed because `depth_anything_v2_vitl.pth` is about 1.34 GB. Put it here before running scripts 04, 05, 07, or 11:

```bash
weights/depth_anything_v2_vitl.pth
```

The default path is configured in `src/shared/case_config.py` through `DA2_CHECKPOINT_PATH`.
