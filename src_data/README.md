# src_data

论文复现所需的只读原始源数据。为保证医疗研究审计能力，原始数据即使暂未被
01-08 直接读取，也不应作为普通临时文件删除。

当前刚性流程直接读取以下内容：

```text
src_data/
├─ Dataset/<patient_id>/
│  ├─ Lap Images/                    # 腹腔镜帧与 CameraParameters
│  ├─ LUS Calibration and Pose/      # Python TRE/IC 评估输入
│  └─ LUS Segmentation/json/         # Python TRE/IC 评估输入
└─ contours_and_models/<patient_id>/
   ├─ Annotations/<frame_id>/<annotator>/contours.xml
   ├─ Liver.obj
   └─ Tumour.obj
```

脚本不得在此目录写入中间数据、推理结果或评估结果。

`Lap Augmented by LUS/`、`LUS Images/`、`LUS Segmentation/Human Mask/`、
`LUS Segmentation/Machine Mask/` 和 `Preoperative Model/` 当前不进入刚性
RefineNet 主路径，但保留用于源数据核查、可视化对照和后续评估验证。
