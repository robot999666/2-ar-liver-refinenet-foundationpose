# AR Liver RefineNet Reproducible Package

这个仓库是 Patient1 肝脏 AR 配准实验的干净可复现副本。它保留核心脚本、源数据、最佳模型权重、固定 seed 和最终结果表，删除了训练样本、depth cache、预测中间文件、旧 sweep 和历史归档。

## 数据
https://encov.ip.uca.fr/ab/code_and_datasets/datasets/llr_reg_evaluation_by_lus/index.php

## 内容

- `scripts/01_prepare_case.py` 到 `scripts/07_infer_export_stl.py`: 主流程脚本
- `scripts/11_batch_patient_frames.py`: Patient1 多帧批量预处理、推理和事后评估汇总
- `scripts/_eval_debug_tre_python.py`: `11` 用到的 TRE/IC debug 评估依赖
- `src/`: 渲染、网络、Depth Anything V2 源码和共享配置
- `src_data/`: Patient1 原始数据、标注、模型和 MATLAB 评估依赖
- `result/Patient1/02/best.pth`: 当前最佳 RefineNet 权重
- `result/Patient1/batch_02_10/`: 02-10 批量推理的最终表格
- `configs/repro.env`: seed 和当前最佳推理参数

没有保存 `data_case/`，因为那里是可重新生成的中间数据。

## 外部权重

Depth Anything V2 的大权重

https://github.com/DepthAnything/Depth-Anything-V2

没有提交到 GitHub，因为文件约 1.34GB。运行 `04`、`05`、`07` 或 `11` 前，请把它放到：

```bash
weights/depth_anything_v2_vitl.pth
```

`result/Patient1/02/best.pth` 约 87.9MB，低于 GitHub 单文件 100MB 硬限制，直接保存在私有仓库中。

## 环境

Docker 方式：

```bash
docker build -t ar-liver-repro .
docker run --gpus all -it --name ar-liver-repro -v ${PWD}:/workspace/AR ar-liver-repro
```

云端已有 PyTorch/CUDA 环境时，可直接安装训练/推理依赖：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_06_train_refine_net.txt
```

## 复现当前批量推理结果

进入容器或云端项目目录：

```bash
cd /workspace/AR
export AR_SEED=42
python scripts/11_batch_patient_frames.py \
  --patient-id Patient1 \
  --frames 02-10 \
  --weight /workspace/AR/result/Patient1/02/best.pth
```

说明：

- 05 帧源数据缺失，会被脚本跳过。
- 推理选择不使用 TRE/IC 真值；TRE/IC 只在推理后用于表格统计。
- 输出会写入 `result/Patient1/batch_02_10/`。

当前已保存结果：

| Frame | Selected Iter | TRE mm | IC |
|---|---:|---:|---|
| 02 | 6 | 8.7662 | Pass |
| 03 | 6 | 9.8264 | Pass |
| 04 | 10 | 15.5117 | Failed |
| 06 | 6 | 14.1527 | Failed |
| 07 | 6 | 13.5747 | Failed |
| 08 | 7 | 19.4491 | Failed |
| 09 | 5 | 27.2291 | Failed |
| 10 | 5 | 29.1876 | Failed |

平均值：8 个有效帧，mean TRE = `17.2122 mm`，IC pass = `2/8`。

## 从源数据重新生成训练数据

中间样本没有保存，可以用固定 seed 重新生成：

```bash
cd /workspace/AR
export PATIENT_ID=Patient1
export FRAME_ID=02
export AR_SEED=42
export AR_NUM_SAMPLES=5000
export AR_LOCAL_REFINE_RATIO=0.7

python scripts/01_prepare_case.py
python scripts/02_setup_initial_pose.py
python scripts/03_render_pose_samples.py
python scripts/04_depth_anything_samples.py
python scripts/05_depth_anything_intraop.py
python scripts/06_train_refine_net.py
```

训练会重新生成 `data_case/` 和 `result/` 下的运行产物。由于 GPU 算子和库版本差异，重新训练不保证 bit-for-bit 得到完全相同权重；要复现当前表格，请使用仓库中保存的 `best.pth`。

## 关键参数

当前最佳推理配置已经写入 `configs/repro.env`，核心为：

```bash
AR_MAX_ITER=10
AR_TRANS_DAMPING=0.4
AR_ROT_DAMPING=0.4
AR_FIRST_STEP_DAMPING=0.75
AR_REFINE_TRANS_DAMPING=0.1
AR_REFINE_ROT_DAMPING=0.1
AR_RAW_WORSEN_PATIENCE=1
AR_RAW_WORSEN_MIN_DELTA=0.5
AR_RAW_SCORE_ROT_WEIGHT=5.0
AR_SELECT_BEST_BY=last
```

更多文件取舍和 hash 见 `MANIFEST.md`。
