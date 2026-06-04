# 真实帧掩码规范

旧流程曾尝试从 `Silhouette` 轮廓线直接填充掩码。轮廓可能不闭合，且可见边界并不等于完整肝脏区域，
因此会得到空掩码、错误连通域或与渲染掩码语义不一致的输入。

当前流程要求每帧运行 `scripts/01b_annotate_intraop_masks.py`：

1. 标注完整肝脏区域，生成 `full_liver_mask.png`。
2. 标注器械遮挡区域，生成 `depth_valid_mask.png = full_liver_mask - occlusion`。
3. 同时保存标注元数据、掩码指纹、质量 JSON 和叠加预览。

`05` 只在 `depth_valid_mask` 内保存推理深度；`07` 将 `full_liver_mask` 作为网络的 mask 通道，
并在加载时验证深度确实由当前掩码生成。

