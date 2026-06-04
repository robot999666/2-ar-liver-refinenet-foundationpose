# 配置参数说明

公共配置优先级为：显式环境变量 > 代码默认值。脚本 02 的初始位姿另外支持
`initial_pose.json`，但显式 `AR_TVIEW_*` 环境变量仍具有最高优先级。

公共配置由 `src/shared/case_config.py` 解析。建议通过 `configs/repro.env`
管理实验，不要直接修改多份脚本。

## 病例、路径和输入契约

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PATIENT_ID` | `Patient1` | 病例英文标识，对应 `src_data`、`data_case` 和 `result` 子目录 |
| `FRAME_ID` | `02` | 帧英文/数字标识，建议保留两位数字 |
| `AR_PROJECT_ROOT` | 项目根目录 | 项目绝对路径 |
| `DATA_ROOT` | `<root>/src_data` | 只读源数据根目录 |
| `CASE_ROOT` | `<root>/data_case` | 可重新生成的病例数据根目录 |
| `RESULT_ROOT` | `<root>/result` | 训练、推理、评估结果根目录 |
| `WEIGHTS_ROOT` | `<root>/weights` | 模型权重目录 |
| `AR_ANNOTATOR` | `koo` | 源轮廓标注者子目录 |
| `AR_WORK_WIDTH` | `480` | 全流程工作图像宽度 |
| `AR_WORK_HEIGHT` | `270` | 全流程工作图像高度 |
| `AR_CONTOUR_THICKNESS` | `2` | GT 与渲染轮廓共同线宽 |
| `AR_SEED` | `42` | 采样、划分、训练和 worker 随机种子 |

修改工作分辨率或轮廓线宽后，必须重新运行 `01` 至 `06`；旧样本和旧权重不再兼容。

## 人工掩码质量门槛

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `AR_FULL_MASK_MIN_RATIO` | `0.05` | 完整肝脏掩码占整幅图像的最小比例 |
| `AR_FULL_MASK_MAX_RATIO` | `0.95` | 完整肝脏掩码占整幅图像的最大比例 |
| `AR_FULL_MASK_LARGEST_COMPONENT_MIN` | `0.95` | 最大连通域占完整肝脏掩码的最小比例 |
| `AR_DEPTH_VALID_TO_FULL_MIN` | `0.10` | 深度有效区占完整肝脏掩码的最小比例 |
| `AR_DEPTH_COVERAGE_MIN` | `0.95` | 深度有效区内非零深度的最小覆盖率 |

这些阈值用于拒绝空掩码、碎裂掩码、错误填充、过度遮挡和过期深度。修改阈值后必须重新验证掩码与深度。

## 初始位姿

脚本 `02_setup_initial_pose.py` 优先读取：

1. `data_case/<patient_id>/<frame_id>/initial_pose.json`
2. `AR_TVIEW_TX/TY/TZ/RX/RY/RZ`
3. 脚本中的 fallback 默认值

| 参数 | 单位 | 说明 |
| --- | --- | --- |
| `AR_TVIEW_TX/TY/TZ` | mm | object-to-camera 初始平移 |
| `AR_TVIEW_RX/RY/RZ` | degree | Euler 旋转，矩阵顺序为 `Rz @ Ry @ Rx` |
| `AR_TVIEW_CONFIG` | path | 自定义初始位姿 JSON 路径 |

最终参数和来源写入 `T_view_meta.json`。

## Script 08 批处理

| CLI 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--patient-id` | `Patient1` | 要处理的患者 |
| `--frames` | `all` | 自动发现全部 Lap PNG；也可传 `02-10` 或 `02,03,06` |
| `--stage` | `run` | `prepare`、`annotate`、`run` 或 `all` |
| `--initial-pose-frame` | 所选第一帧 | 只在该参考帧运行 script 02，并将 `T_view` 共享给其余帧 |
| `--initial-pose-config` | 未设置 | 参考帧初始位姿 JSON；通过 `AR_TVIEW_CONFIG` 传给 script 02 |
| `--weight` | 标准权重解析结果 | 全部帧共用的 RefineNet checkpoint |
| `--force-mask` | 关闭 | 重新标注并覆盖已有人工 mask |

`--stage all` 会依次执行批量预处理、逐帧交互式 mask 标注、深度、推理和评估。
它仍要求人工完成 mask 标注，不会用轮廓自动填充代替人工掩码。

## 训练配对生成

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `AR_NUM_SAMPLES` | `5000` | 显式 A/B 配对总数，必须至少为 2 |
| `AR_TRAIN_RATIO` | `0.9` | 确定性 train/val 划分比例 |
| `AR_TRANS_RANGE_MM` | `50` | broadly sampled target 的平移范围 |
| `AR_ROT_RANGE_DEG` | `20` | broadly sampled target 的旋转范围 |
| `AR_LOCAL_TRANS_RANGE_MM` | `5` | local_refine A 相对 B 的平移范围 |
| `AR_LOCAL_ROT_RANGE_DEG` | `3` | local_refine A 相对 B 的旋转范围 |
| `AR_LOCAL_REFINE_RATIO` | `0.7` | local_refine 配对占比 |
| `AR_MAX_ATTEMPTS` | `0` | 最大采样尝试次数；`0` 表示 `NUM_SAMPLES * 50` |

`base_to_target` 用于学习大范围初始修正；`local_refine` 用于学习目标附近的小修正。

## 训练

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `AR_BATCH_SIZE` | `32` | batch size |
| `AR_NUM_EPOCHS` | `50` | 训练 epoch 数 |
| `AR_NUM_WORKERS` | `4` | DataLoader worker 数 |
| `AR_RESUME` | `0` | `1` 时从 `checkpoints/last.pth` 恢复 |

除非数据和关键参数完全一致，否则不要启用 `AR_RESUME=1`。

## 推理

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `AR_WEIGHT_PATH` | 当前帧 `best.pth` 或固定权重 | 显式 RefineNet 权重 |
| `AR_PRETRAINED_WEIGHT_PATH` | `weights/refinenet_patient1_02_best.pth` | 未显式指定且当前帧没有 `best.pth` 时的回退权重 |
| `AR_INFER_PROFILE` | `paper_translation_stop1mm_last` | 写入日志的推理策略审计标签 |
| `AR_UPDATE_MODE` | `right` | 当前位姿与偏移的组合方式 |
| `AR_MAX_ITER` | `10` | 最大迭代次数 |
| `AR_STOP_TRANS_MM` | `1.0` | 接受更新后，预测平移残差低于该值时停止 |
| `AR_TRANS_DAMPING` | `0.4` | 第一次基础平移阻尼 |
| `AR_ROT_DAMPING` | `0.4` | 第一次基础旋转阻尼 |
| `AR_FIRST_STEP_DAMPING` | `0.75` | 第一次额外阻尼 |
| `AR_REFINE_TRANS_DAMPING` | `0.1` | 后续平移阻尼 |
| `AR_REFINE_ROT_DAMPING` | `0.1` | 后续旋转阻尼 |
| `AR_MAX_STEP_TRANS_MM` | `0` | 可选单步平移上限；`0` 表示关闭 |
| `AR_MAX_STEP_ROT_DEG` | `0` | 可选单步旋转上限；`0` 表示关闭 |
| `AR_DIVERGE_RAW_TRANS_MM` | `120` | 原始预测平移发散阈值 |
| `AR_DIVERGE_RAW_ROT_DEG` | `45` | 原始预测旋转发散阈值 |
| `AR_DIVERGE_POSE_TRANS_MM` | `200` | 相对初始位姿总漂移阈值 |
| `AR_STOP_DRIFT_MM` | `0` | 可选总漂移停止阈值；`0` 表示关闭 |
| `AR_ALLOW_ZERO_DEPTH` | `0` | 仅缺失模态消融时设为 `1` |

当前没有证据证明轮廓/mask proxy 与 TRE 单调一致。下列参数默认不允许控制接受、停止或最终选择；
只有明确做消融实验时，才允许设置 `AR_ALLOW_UNVALIDATED_SCORE_CONTROL=1`：

| 环境变量 | 默认值 | 说明 |
| --- | ---: | --- |
| `AR_SELECT_BEST_BY` | `last` | 默认必须选择最后安全接受位姿 |
| `AR_ACCEPT_BY_SCORE` | `0` | 是否用图像 proxy 决定接受更新 |
| `AR_SCORE_PATIENCE` | `0` | 图像 proxy 恶化停止耐心轮数 |
| `AR_RAW_WORSEN_PATIENCE` | `0` | 原始预测残差恶化停止耐心轮数 |
| `AR_RAW_SCORE_ROT_WEIGHT` | `5.0` | 原始平移/旋转残差诊断分数中的旋转权重 |
| `AR_RAW_WORSEN_MIN_DELTA` | `0.5` | 未验证残差恶化控制的最小变化 |
| `AR_SCORE_ACCEPT_TOL` | `0.0` | 未验证 proxy 接受容差 |
| `AR_CONTOUR_SCORE_MAX_DIST_PX` | `50.0` | 轮廓 Chamfer 诊断距离截断，单位 pixel |
| `AR_CONTOUR_SCORE_MASK_WEIGHT` | `20.0` | mask IoU 诊断项权重 |
| `AR_ALLOW_UNVALIDATED_SCORE_CONTROL` | `0` | 允许上述未验证控制项的显式消融开关 |

## Depth Anything V2

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `DA2_PROJECT_DIR` | `<root>/src` | 包含 `depth_anything_v2` 的目录 |
| `DA2_ENCODER` | `vitl` | DA2 encoder |
| `DA2_INPUT_SIZE` | `518` | DA2 推理输入尺寸 |
| `DA2_CHECKPOINT_PATH` | `weights/depth_anything_v2_vitl.pth` | DA2 权重路径 |
