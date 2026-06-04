# data_case

脚本生成的可重建工作目录，格式为
`data_case/<patient_id>/<frame_id>/`。除本说明外，内容不纳入 Git。

- `image/`、`camera/`、`contours/`、`models/`：脚本 01 输出
- `masks/`：脚本 01b 人工标注与 QA 元数据
- `depth/`：脚本 05 输出
- `sample/`：脚本 03/04 生成的训练样本与深度
- `T_view.txt`、`T_view_meta.json`：脚本 02 输出

删除某帧前，确认需要保留的人工掩码已经备份；其余内容可从
`src_data/`、配置和固定随机种子重新生成。
