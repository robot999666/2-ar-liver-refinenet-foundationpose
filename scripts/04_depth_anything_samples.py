"""为渲染训练样本生成原始 Depth Anything V2 深度数组。

保存的 *_depth.npy 是 DA2 原始输出。归一化统一延后到共享的
train/validation/inference 输入流程中执行，从而保证所有模式先使用相同的
确定性映射，再仅对训练数据进行增强。
"""

import argparse
import glob
import json
import os
import sys

import cv2
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config
from shared.da2_engine import get_estimator, save_raw_depth


def load_prefixes(split_dir):
    labels_path = os.path.join(split_dir, "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return sorted(labels.keys())

    images_dir = os.path.join(split_dir, "images")
    rgb_files = sorted(glob.glob(os.path.join(images_dir, "*_rgb.png")))
    return [os.path.basename(path).replace("_rgb.png", "") for path in rgb_files]


def depth_is_current(depth_path, rgb_path):
    if not os.path.exists(depth_path) or not os.path.exists(rgb_path):
        return False
    return os.path.getmtime(depth_path) >= os.path.getmtime(rgb_path)


def process_split(split_dir, estimator, skip_existing: bool):
    images_dir = os.path.join(split_dir, "images")
    prefixes = load_prefixes(split_dir)
    if not prefixes:
        print(f"[WARN] no labeled samples in {split_dir}")
        return 0

    count = 0
    for prefix in tqdm(prefixes, desc=os.path.basename(split_dir)):
        rgb_path = os.path.join(images_dir, f"{prefix}_rgb.png")
        depth_path = os.path.join(images_dir, f"{prefix}_depth.npy")
        if skip_existing and depth_is_current(depth_path, rgb_path):
            continue

        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[skip] bad image: {rgb_path}")
            continue

        depth = estimator.infer(bgr)
        save_raw_depth(depth, depth_path)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="为渲染训练样本生成原始 DA2 深度。")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="存在 *_depth.npy 时跳过该样本",
    )
    args = parser.parse_args()

    case_dir = case_config.case_dir()
    train_dir = os.path.join(case_dir, "sample", "train")
    val_dir = os.path.join(case_dir, "sample", "val")

    print("=== 04. Depth Anything on training samples ===")
    print(f"  frame: {case_config.PATIENT_ID}/{case_config.FRAME_ID}")
    print("  output: raw DA2 float32, no post-processing")

    estimator = get_estimator()
    n_train = process_split(train_dir, estimator, args.skip_existing)
    n_val = process_split(val_dir, estimator, args.skip_existing)
    print(f"[OK] wrote depth: train={n_train}, val={n_val}")
    print("[Next] python 05_depth_anything_intraop.py")


if __name__ == "__main__":
    main()
