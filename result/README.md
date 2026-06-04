# result

训练、推理和 Python 评估的统一输出根目录。

每个病例帧使用 `result/<Patient>/<Frame>/`，其中包含 `logs/`、`checkpoints/`、
`training/`、`inference/`、`evaluation/` 和 `mask_qa/`。批处理汇总保存在
`result/<Patient>/batch_<start>_<end>/`。

