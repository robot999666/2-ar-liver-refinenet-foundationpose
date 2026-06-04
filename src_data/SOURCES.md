# 原始数据来源

`src_data` 来自 EnCoV 官方发布的 LUS 配准评估数据集和 Koo 等人的轮廓/简化模型数据。

## 官方下载

| 内容 | 官方地址 | SHA256 |
| --- | --- | --- |
| 四位患者的完整评估数据 | `https://encov.ip.uca.fr/ab/code_and_datasets/datasets/llr_reg_evaluation_by_lus/Dataset_v1p0.zip` | `EFE901F9DFAABF7AA9EF7F7A9EC25C88564C46934F9DB2C6ADC7DE7C550AC1E4` |
| Koo 轮廓与简化模型 | `https://encov.ip.uca.fr/ab/code_and_datasets/datasets/llr_reg_evaluation_by_lus/Data_2017_Koo_etal_v1p0.zip` | `113421C1D9C25911C98BABBD4FACF0C94D0F8E27563960B0058801C031D9679B` |

官方数据说明保存在 `src_data/Readme.pdf`。

Git 会按 `.gitattributes` 规范化部分文本文件的换行符，因此仓库内解压后的
文本文件不保证与 ZIP 内文件逐字节同哈希；官方 ZIP 的 SHA256 才是下载包校验依据。

## 当前完整帧

“完整帧”表示同时具有 Lap image、CameraParameters、LUS calibration/pose、
LUS segmentation JSON、Koo contours、Liver.obj 和 Tumour.obj，可由 script 08
完成预处理与 Python TRE/IC 评估。

| Patient | 完整帧 |
| --- | --- |
| `Patient1` | `02,03,04,06,07,08,09,10` |
| `Patient2` | `01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,19,20,21,22,23` |
| `Patient3` | `02,03,04,05,06,07,08,09,10` |
| `Patient4` | `02,03,04,05,06,07,08,09` |

## 恢复方法

将两个官方 ZIP 解压到同一临时目录，然后把其中的 `Dataset/` 和
`contours_and_models/` 合并到 `src_data/`。恢复前后必须核对 SHA256，
不得用生成的 `data_case/` 文件替代原始数据。
