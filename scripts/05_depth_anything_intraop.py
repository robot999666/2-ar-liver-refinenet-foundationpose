import os
import sys

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config
from shared.da2_engine import get_estimator, save_depth_preview, save_raw_depth


# ============================================================
# 05_depth_anything_intraop.py
# ------------------------------------------------------------
# 对去畸变术中图 lap_undist.png 运行 Depth Anything V2，
# 保存原始深度 depth/da2_intraop.npy（无任何后处理）
#
# 直接运行：python 05_depth_anything_intraop.py
# ============================================================


def main():
    case_dir = case_config.case_dir()
    depth_dir = case_config.depth_dir()
    os.makedirs(depth_dir, exist_ok=True)

    lap_path = os.path.join(case_dir, "image", "lap_undist.png")
    out_npy = os.path.join(depth_dir, "da2_intraop.npy")
    out_vis = os.path.join(depth_dir, "da2_intraop_preview.png")

    print("=== 05. Depth Anything on intra-op lap_undist ===")
    print(f"  frame: {case_config.PATIENT_ID}/{case_config.FRAME_ID}")

    lap = cv2.imread(lap_path, cv2.IMREAD_COLOR)
    if lap is None:
        raise FileNotFoundError(f"Run 01_prepare_case.py first. Missing: {lap_path}")

    estimator = get_estimator()
    depth = estimator.infer(lap)
    save_raw_depth(depth, out_npy)
    save_depth_preview(depth, out_vis)

    print(f"[OK] raw depth: {out_npy}  shape={depth.shape}  range=[{depth.min():.4f}, {depth.max():.4f}]")
    print(f"[OK] preview only: {out_vis}")
    print("[Next] python 06_train_refine_net.py")


if __name__ == "__main__":
    main()
