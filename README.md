# AR Liver RefineNet Rigid Reproduction

本项目只复现论文刚性部分：位姿偏移预测、迭代刚性推理，以及 Python TRE/IC 评估。
当前目录已去除 MATLAB 评估、旧实验脚本和历史生成数据。所有可重复生成的病例中间数据放在
`data_case/`，训练、推理、评估日志和结果统一放在 `result/<Patient>/<Frame>/`。

## 推荐流程

```bash
python scripts/01_prepare_case.py
python scripts/01b_annotate_intraop_masks.py
python scripts/02_setup_initial_pose.py
python scripts/03_render_pose_samples.py
python scripts/04_depth_anything_samples.py
python scripts/05_depth_anything_intraop.py
python scripts/06_train_refine_net.py
python scripts/07_infer_export_stl.py
python scripts/08_batch_patient_frames.py --patient-id Patient1 --frames 02-10
```

病例与帧默认由 `PATIENT_ID`、`FRAME_ID` 环境变量控制。固定实验参数见
`configs/repro.env`。训练默认从头开始；只有显式设置 `AR_RESUME=1` 才会读取
`checkpoints/last.pth`。`07` 默认使用当前经过验证的推理策略：

- 第一次更新应用网络偏移的 `0.30`，后续更新应用 `0.10`。
- 最多迭代 `10` 次；接受更新后预测平移残差小于 `1 mm` 时停止。
- 最终选择最后一个安全接受的位姿。
- 轮廓距离与掩码 IoU 只用于诊断，默认不参与接受、停止或选优。
- TRE/IC 只在推理完成后由 `08` 评估，绝不参与推理选择。

## 掩码要求

真实帧不能再从轮廓线自动填充掩码。每帧必须先运行 `01b`，保存：

- `full_liver_mask.png`：完整可推断肝脏区域，包含被器械遮挡的区域。
- `depth_valid_mask.png`：可用于 DA2 深度的区域，不包含器械遮挡。
- 掩码标注、质量报告和可视化叠加图。

`05` 与 `07` 会校验面积、连通性、文件指纹、深度覆盖率及深度来源；文件不一致会直接失败。

## Docker 与云上训练

```bash
docker build -t ar-liver-refinenet .
docker run --gpus all --rm -it -v "$PWD:/workspace/AR" ar-liver-refinenet
```

云上已有 CUDA/PyTorch 环境时，可离线安装训练依赖：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-index --find-links=wheelhouse -r requirements_06_train_refine_net.txt
```

`archives/wheelhouse.zip` 解压后目录名应为 `wheelhouse/`。DA2 权重放在
`weights/depth_anything_v2_vitl.pth`，固定 RefineNet 权重放在
`weights/refinenet_patient1_02_best.pth`。

## 结果布局

```text
result/<Patient>/<Frame>/
  checkpoints/       训练权重
  training/          损失曲线等训练产物
  inference/         最终位姿、STL、每轮调试结果
  evaluation/        Python TRE/IC 结果
  mask_qa/           批处理保存的掩码质量快照
  logs/              train.log、infer.log、evaluation.log 等日志
```

本项目用于研究复现，不可直接用于临床决策。
