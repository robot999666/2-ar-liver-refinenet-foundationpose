# 可复现性资产清单

本文件用于区分纳入版本控制的项目文件与外部大型运行资产。

## 纳入版本控制的模型

| Path | 用途 | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `weights/refinenet_patient1_02_best.pth` | 固定的刚性 RefineNet checkpoint | `87866632` | `3B84F1E992ED836915018857337C5F180A11371B55C45F02BDEF4302E0B5C307` |

## 必需的外部运行资产

以下大型文件有意被 Git 忽略，必须单独备份，并在运行相关阶段前恢复。

| Path | 用途 | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `weights/depth_anything_v2_vitl.pth` | Scripts 04、05 和 07 使用的官方 Depth Anything V2 Large checkpoint | `1341395338` | `A7EA19FA0ED99244E67B624C72B8580B7E9553043245905BE58796A608EB9345` |
| `archives/wheelhouse.zip` | Linux x86_64 / Python 3.12 云端离线训练依赖 | `632832579` | `02AC2F551C9AD55034ACE3C740E2110F55CE0B2B37F03EA0765F09DD1D98C255` |

`archives/wheelhouse.zip` 在顶层 `wheelhouse/` 目录中包含 78 个 wheel。
它用于文档说明的云端 Python 3.12 流程，不用于 `Dockerfile` 构建的
Python 3.10 环境。

使用前验证已恢复文件：

```powershell
Get-FileHash weights\depth_anything_v2_vitl.pth -Algorithm SHA256
Get-FileHash archives\wheelhouse.zip -Algorithm SHA256
python -m zipfile -t archives\wheelhouse.zip
```

## 源数据范围

当前仓库包含 scripts 01 和 08 所需的 Patient1-4 源文件：

- 腹腔镜帧和相机参数；
- LUS 标定和位姿 `.mat` 文件；
- Python TRE/IC 评估使用的 LUS segmentation JSON 文件；
- 轮廓 XML 标注；
- 肝脏和肿瘤 OBJ 模型。

当前刚性流程未直接读取的其他已发布源文件仍保留，用于审计和未来
验证。不得将这些文件与生成的 `data_case/` 数据混淆。

官方来源、官方 ZIP SHA256 和每位患者的完整帧记录在
`src_data/SOURCES.md`。Script 08 使用 `--frames all` 时会根据 Lap Images
自动发现患者全部帧，并在缺少其他必需输入时明确报告。

生成的 `data_case/` 和 `result/` 内容有意排除在 Git 之外，其中也包括
script 01b 生成的人工掩码包。干净仓库不附带这些标注；运行 scripts 05、
07 或 08 前必须创建它们并在外部备份。
