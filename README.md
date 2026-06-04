# AR Liver RefineNet: Rigid Pose Reproduction

本项目用于复现论文刚性部分：

- `III. METHOD A. Pose Offset Prediction`
- `IV. EXPERIMENTS A. Rigid Offset Prediction`

项目实现从源数据预处理、显式 A/B 位姿配对、RefineNet 训练、迭代刚性推理，到 Python
TRE/IC 事后评估的完整流程。项目不包含非刚性配准，也不允许使用 TRE/IC 真值参与推理选优。

> **用途声明**：这是医疗影像研究复现项目，不是经过临床验证的医疗器械，不可直接用于临床决策。

## 当前状态

- 核心脚本保留为 `01`、`01b`、`02` 至 `08`。
- 原脚本 `11_batch_patient_frames.py` 已重构为 `08_batch_patient_frames.py`。
- MATLAB 评估脚本已删除，TRE/IC 全部由 Python 完成。
- 训练、推理、评估日志统一写入 `result/<patient_id>/<frame_id>/logs/`。
- 真实帧掩码必须人工标注并通过质量校验，不再从可能未闭合的轮廓线自动填充。
- 当前代码由 Git 管理。整理前版本与旧结果仍可从 Git 历史恢复。
- 项目中的文件夹名均为英文；`<patient_id>`、`<frame_id>` 是路径占位符，不是中文目录名。
- 当前 Patient1 完整源输入包含 `02`、`03`、`04`、`06` 至 `10`；源数据不含 `05`。
- 干净仓库不附带 `data_case/` 人工掩码；首次严格流程必须逐帧运行交互式 `01b`，否则 `05/07/08` 会安全失败。

## 项目目录

```text
AR/
├─ scripts/                         # 01-08 核心执行脚本
├─ src/
│  ├─ depth_anything_v2/            # Depth Anything V2 模型实现
│  ├─ learning/                     # RefineNet 网络实现
│  └─ shared/                       # 公共配置、渲染、掩码、深度与 Python 评估
├─ src_data/                        # 只读源数据；脚本不得向这里写结果
├─ data_case/                       # 可重新生成的病例中间数据
├─ result/                          # 训练、推理、评估结果与日志
├─ weights/                         # DA2 与 RefineNet 权重
├─ archives/                        # 云端离线依赖包 wheelhouse.zip
├─ configs/repro.env                # 可复现实验参数模板
├─ docs/                            # 配置、掩码、推理策略与 Git 工作流说明
├─ tests/                           # 掩码、深度来源、推理安全和批处理测试
├─ Dockerfile                       # CUDA 11.8 Docker 环境
├─ requirements.txt                 # Docker/完整流程 Python 依赖
├─ requirements_06_train_refine_net.txt
├─ MANIFEST.md                      # 权重和外部大文件校验信息
└─ README.md
```

### 核心脚本

| 脚本 | 作用 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `01_prepare_case.py` | 去畸变、轮廓栅格化、模型中心化 | `src_data/` | `data_case/<patient_id>/<frame_id>/` |
| `01b_annotate_intraop_masks.py` | 人工标注完整肝脏与器械遮挡 | `lap_undist.png` | `data_case/.../masks/` |
| `02_setup_initial_pose.py` | 建立并记录初始位姿 | prepared case、初始位姿参数 | `T_view.txt`、`result/.../preprocess/` |
| `03_render_pose_samples.py` | 生成显式 coarse/local A/B 配对 | prepared case、seed | `data_case/.../sample/` |
| `04_depth_anything_samples.py` | 为渲染训练样本生成 raw DA2 深度 | sample RGB、DA2 权重 | `*_depth.npy` |
| `05_depth_anything_intraop.py` | 为真实帧生成经掩码审计的 DA2 深度 | 人工掩码、真实图像 | `data_case/.../depth/` |
| `06_train_refine_net.py` | 训练刚性位姿偏移网络 | 03/04 样本 | `result/.../checkpoints/` |
| `07_infer_export_stl.py` | 多轮刚性推理并导出 STL | prepared case、05 深度、RefineNet 权重 | `result/.../inference/` |
| `08_batch_patient_frames.py` | 批处理 01/02/05/07 并做 Python TRE/IC 评估 | 多帧源数据、已标注掩码 | 每帧结果与 batch 汇总 |

`08` 不会自动执行交互式 `01b`，也不会自动训练模型。批处理前必须为每个目标帧准备并检查人工掩码。

## 路径与输出约定

单帧生成数据：

```text
data_case/<patient_id>/<frame_id>/
├─ image/                # 去畸变后的真实图像
├─ camera/               # 缩放后的相机内参与畸变参数
├─ contours/             # 三通道真实轮廓与模型轮廓定义
├─ masks/                # 人工完整肝脏、深度有效区、遮挡和 QA
├─ depth/                # raw/audited DA2 深度
├─ models/               # 中心化肝脏/肿瘤模型
├─ sample/               # 03/04 生成的训练/验证样本
├─ T_view.txt            # 初始 object-to-camera 位姿
└─ T_view_meta.json      # 初始位姿参数来源
```

单帧持久结果：

```text
result/<patient_id>/<frame_id>/
├─ logs/
│  ├─ train.log
│  ├─ infer_preflight.log
│  ├─ infer.log
│  └─ evaluation.log
├─ checkpoints/          # best.pth、best_full.pth、last.pth
├─ training/             # loss_curve.png 等
├─ preprocess/           # 初始位姿调试图
├─ inference/
│  ├─ final_pose.txt
│  ├─ tumour_pred_cam.stl
│  ├─ liver_pred_cam.ply
│  └─ debug_iterations/
├─ evaluation/           # Python TRE/IC 结果
└─ mask_qa/              # 批处理保存的掩码质量快照
```

批处理汇总位于：

```text
result/<patient_id>/batch_<start_frame>_<end_frame>/
```

## 必需文件

完整运行前检查：

```text
weights/depth_anything_v2_vitl.pth
weights/refinenet_patient1_02_best.pth
```

云端离线训练时还需要：

```text
archives/wheelhouse.zip
requirements_06_train_refine_net.txt
```

文件大小和 SHA256 记录在 [MANIFEST.md](MANIFEST.md)。大文件默认不提交到 Git，必须额外备份。

## 参数修改

所有脚本共同参数由 `src/shared/case_config.py` 解析。推荐复制或修改
`configs/repro.env`，不要在多个脚本中分别硬编码。

Linux / Docker：

```bash
set -a
source configs/repro.env
set +a
```

PowerShell 示例：

```powershell
$env:PATIENT_ID = "Patient1"
$env:FRAME_ID = "02"
$env:AR_SEED = "42"
$env:AR_WEIGHT_PATH = "D:\recent\AR\weights\refinenet_patient1_02_best.pth"
```

完整参数、单位、默认值和风险见 [docs/CONFIGURATION.md](docs/CONFIGURATION.md)。

## 单帧完整流程

### 1. 预处理并标注真实帧

```bash
python scripts/01_prepare_case.py
python scripts/01b_annotate_intraop_masks.py
python scripts/02_setup_initial_pose.py
python scripts/05_depth_anything_intraop.py
```

`01b` 必须标注：

- `full_liver_mask.png`：完整肝脏区域，包括被器械遮挡但应属于肝脏的区域。
- `depth_valid_mask.png`：允许使用 DA2 深度的区域，不包括器械遮挡。

`05` 和 `07` 会校验面积、连通域、文件指纹、深度覆盖率和深度来源；不一致时直接失败。
`01b` 需要可显示 OpenCV 窗口，建议在本机桌面环境完成；生成后应单独备份人工掩码。

### 2. 生成训练数据并训练

```bash
python scripts/03_render_pose_samples.py
python scripts/04_depth_anything_samples.py
python scripts/06_train_refine_net.py
```

训练默认从头开始。只有确认数据分布与关键参数未变化时，才设置：

```bash
export AR_RESUME=1
```

### 3. 推理

```bash
python scripts/07_infer_export_stl.py
```

默认推理策略：

- 第一次应用网络偏移的 `0.4 * 0.75 = 0.30`。
- 后续每轮应用网络偏移的 `0.10`。
- 最多迭代 `10` 次。
- 接受当前更新后，模型预测平移残差小于 `1 mm` 时停止。
- 默认选择最后一个安全接受的位姿。
- 轮廓 Chamfer 和 mask IoU 只记录为诊断值，不参与默认选优。
- TRE/IC 仅在推理结束后评估，绝不参与推理循环。

### 4. Python TRE/IC 评估

对单帧结果进行批处理式评估：

```bash
python scripts/08_batch_patient_frames.py \
  --patient-id Patient1 \
  --frames 02 \
  --skip-existing-case \
  --skip-existing-depth \
  --skip-existing-infer
```

## 多帧批处理

先逐帧运行 `01` 和 `01b` 完成人工掩码，再运行：

```bash
python scripts/08_batch_patient_frames.py \
  --patient-id Patient1 \
  --frames 02-10 \
  --weight weights/refinenet_patient1_02_best.pth
```

当源数据、掩码、深度或推理结果已存在时，可使用：

```bash
--skip-existing-case
--skip-existing-depth
--skip-existing-infer
```

## Docker

```bash
docker build -t ar-liver-refinenet .
docker run --gpus all --rm -it \
  -v "$PWD:/workspace/AR" \
  ar-liver-refinenet
```

进入容器后：

```bash
cd /workspace/AR
set -a
source configs/repro.env
set +a
python scripts/05_depth_anything_intraop.py
```

## 云端离线训练

该 wheelhouse 面向 **Linux x86_64 / Python 3.12**。`Dockerfile` 则使用 Ubuntu 22.04
系统 Python 3.10 和在线依赖安装；两套环境不要混用 wheel。

将 `archives/wheelhouse.zip` 解压到项目目录，使其形成 `wheelhouse/`：

```bash
python3 -m zipfile -e archives/wheelhouse.zip .
python3 -m pip install --no-index --find-links=wheelhouse pip setuptools wheel
python3 -m pip install --no-index --find-links=wheelhouse \
  -r requirements_06_train_refine_net.txt
python3 scripts/06_train_refine_net.py
```

`requirements_06_train_refine_net.txt` 故意不包含 PyTorch，避免覆盖云平台已安装的 CUDA PyTorch。

## Git 版本保护

当前整理工作位于分支 `cleanup-restructure-v2`。重要修改前必须先提交保护点：

```bash
git status
git add -A
git commit -m "checkpoint: describe current state"
git switch -c feature/<change-name>
```

恢复单个误删文件：

```bash
git restore --source=<commit> -- path/to/file
```

查看整理前版本：

```bash
git show c059ce8:path/to/file
```

禁止对本项目使用未经检查的 `git clean -fdx`、`git reset --hard` 或递归通配符删除。
完整规则见 [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)。
