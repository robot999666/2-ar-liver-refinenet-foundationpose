# 共享模块

| Module | 用途 |
| --- | --- |
| `case_config.py` | 集中管理路径、病例/帧默认值、随机种子、权重和结果目录结构。 |
| `da2_engine.py` | Depth Anything V2 加载和推理。 |
| `depth_augment.py` | 共享深度归一化和训练增强。 |
| `evaluation.py` | Script 08 使用的 Python TRE/IC 评估。 |
| `intraop_masks.py` | 真实帧掩码语义、校验、来源追踪和 QA overlay。 |
| `pose_render.py` | 使用 Open3D raycast 渲染轮廓、mask 和 DA2 RGB 输入。 |

常用覆盖参数包括 `PATIENT_ID`、`FRAME_ID`、`AR_SEED`、`AR_WEIGHT_PATH`、
`CASE_ROOT`、`DATA_ROOT`、`RESULT_ROOT` 和 `WEIGHTS_ROOT`。
