# result

训练、推理和 Python 评估的统一输出根目录。生成内容不纳入 Git。

每帧使用 `result/<patient_id>/<frame_id>/`：

- `logs/`：`train.log`、`infer_preflight.log`、`infer.log`、`evaluation.log`
- `checkpoints/`：训练产生的权重
- `training/`：训练曲线
- `preprocess/`：初始位姿调试图
- `inference/`：最终位姿、STL/PLY 和逐轮 debug
- `evaluation/`：Python TRE/IC 结果
- `mask_qa/`：掩码质量快照

批处理汇总保存在 `result/<patient_id>/batch_<start_frame>_<end_frame>/`。
