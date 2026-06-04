"""为刚性偏移训练渲染可复现的显式 A/B 配对。

base_to_target 配对用于学习从初始位姿开始的粗修正。
local_refine 配对用于学习采样目标位姿附近的小范围修正。
"""

import json
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config
from shared.pose_render import PoseSampleRenderer, euler_to_matrix


class Config:
    """Script 03 采样配置。

    平移范围单位为 mm，旋转范围单位为 degree。修改这些参数会改变训练
    数据分布，必须与生成的 checkpoint 一同记录。
    """

    PATIENT_ID = case_config.PATIENT_ID
    FRAME_ID = case_config.FRAME_ID
    CASE_ROOT = case_config.CASE_ROOT

    # 显式配对总数，以及可复现且保证非空的 train/validation 划分。
    NUM_PAIRS = int(os.environ.get("AR_NUM_SAMPLES", "5000"))
    TRAIN_RATIO = float(os.environ.get("AR_TRAIN_RATIO", "0.9"))
    SEED = case_config.SEED

    # 两类配对共同使用的宽范围目标位姿分布。
    GLOBAL_TRANS_RANGE_MM = float(os.environ.get("AR_TRANS_RANGE_MM", "50.0"))
    GLOBAL_ROT_RANGE_DEG = float(os.environ.get("AR_ROT_RANGE_DEG", "20.0"))

    # 仅用于 local_refine 配对的 A 到 B 小范围扰动。
    LOCAL_TRANS_RANGE_MM = float(os.environ.get("AR_LOCAL_TRANS_RANGE_MM", "5.0"))
    LOCAL_ROT_RANGE_DEG = float(os.environ.get("AR_LOCAL_ROT_RANGE_DEG", "3.0"))
    LOCAL_REFINE_RATIO = float(os.environ.get("AR_LOCAL_REFINE_RATIO", "0.7"))

    # 保存配对前拒绝不可用的渲染结果。
    MIN_VISIBLE_TYPES = 2
    MIN_MASK_RATIO = 0.02
    MAX_MASK_RATIO = 0.95
    RENDER_CONTOUR_THICKNESS = case_config.CONTOUR_THICKNESS

    # 设为 0 时使用 NUM_PAIRS * 50 次尝试上限；失败会明确报错。
    MAX_ATTEMPTS = int(os.environ.get("AR_MAX_ATTEMPTS", "0"))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sample_delta(trans_range_mm, rot_range_deg):
    rx, ry, rz = np.random.uniform(-rot_range_deg, rot_range_deg, 3)
    tx, ty, tz = np.random.uniform(-trans_range_mm, trans_range_mm, 3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = euler_to_matrix(rx, ry, rz)
    T[:3, 3] = [tx, ty, tz]
    return T


def sample_global_delta(cfg):
    return sample_delta(cfg.GLOBAL_TRANS_RANGE_MM, cfg.GLOBAL_ROT_RANGE_DEG)


def sample_local_delta(cfg):
    return sample_delta(cfg.LOCAL_TRANS_RANGE_MM, cfg.LOCAL_ROT_RANGE_DEG)


def render_state(renderer, delta_T, total_pixels, cfg):
    contours, mask, rgb_bgr, T_render, visible, _ = renderer.render(delta_T)
    mask_ratio = np.count_nonzero(mask) / total_pixels
    valid = (
        len(visible) >= cfg.MIN_VISIBLE_TYPES
        and cfg.MIN_MASK_RATIO <= mask_ratio <= cfg.MAX_MASK_RATIO
    )
    state = {
        "contours": contours,
        "mask": mask,
        "rgb_bgr": rgb_bgr,
        "Delta_T": delta_T,
        "T_render": T_render,
        "visible_types": visible,
        "mask_ratio": float(mask_ratio),
    }
    return valid, state


def save_state(images_dir, labels, prefix, state):
    if prefix not in labels:
        np.save(
            os.path.join(images_dir, f"{prefix}_contours.npy"),
            state["contours"].astype(np.float32) / 255.0,
        )
        cv2.imwrite(os.path.join(images_dir, f"{prefix}_mask.png"), state["mask"])
        cv2.imwrite(os.path.join(images_dir, f"{prefix}_rgb.png"), state["rgb_bgr"])

    labels[prefix] = {
        "Delta_T": state["Delta_T"].tolist(),
        "T_render": state["T_render"].tolist(),
        "visible_types": state["visible_types"],
        "mask_ratio": state["mask_ratio"],
    }


def save_pair(split_dir, labels, pairs, pair_id, pair_type, state_A, state_B):
    images_dir = os.path.join(split_dir, "images")
    if pair_type == "base_to_target":
        prefix_A = "base"
        prefix_B = f"{pair_id}_B"
    else:
        prefix_A = f"{pair_id}_A"
        prefix_B = f"{pair_id}_B"

    save_state(images_dir, labels, prefix_A, state_A)
    save_state(images_dir, labels, prefix_B, state_B)
    pairs[pair_id] = {
        "pair_type": pair_type,
        "A": prefix_A,
        "B": prefix_B,
        "Delta_A": labels[prefix_A]["Delta_T"],
        "Delta_B": labels[prefix_B]["Delta_T"],
    }


def make_pair(renderer, base_state, total_pixels, cfg):
    pair_type = "local_refine" if np.random.rand() < cfg.LOCAL_REFINE_RATIO else "base_to_target"

    if pair_type == "base_to_target":
        valid_B, state_B = render_state(renderer, sample_global_delta(cfg), total_pixels, cfg)
        if not valid_B:
            return None
        return pair_type, base_state, state_B

    delta_B = sample_global_delta(cfg)
    delta_A = delta_B @ sample_local_delta(cfg)
    valid_A, state_A = render_state(renderer, delta_A, total_pixels, cfg)
    if not valid_A:
        return None
    valid_B, state_B = render_state(renderer, delta_B, total_pixels, cfg)
    if not valid_B:
        return None
    return pair_type, state_A, state_B


def main():
    cfg = Config()
    np.random.seed(cfg.SEED)
    if cfg.NUM_PAIRS < 2:
        raise ValueError("AR_NUM_SAMPLES must be at least 2 so train and val are both non-empty.")
    train_pair_count = int(round(cfg.NUM_PAIRS * cfg.TRAIN_RATIO))
    train_pair_count = min(max(train_pair_count, 1), cfg.NUM_PAIRS - 1)
    split_plan = np.array(
        ["train"] * train_pair_count + ["val"] * (cfg.NUM_PAIRS - train_pair_count),
        dtype=object,
    )
    np.random.default_rng(cfg.SEED).shuffle(split_plan)

    case_dir = case_config.case_dir(cfg.CASE_ROOT)
    image_path = os.path.join(case_dir, "image", "lap_undist.png")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    height, width = img.shape[:2]
    total_pixels = width * height

    renderer = PoseSampleRenderer(
        os.path.join(case_dir, "models", "Liver_obj.ply"),
        os.path.join(case_dir, "models", "Liver_vertices_centered_raw_order.npy"),
        os.path.join(case_dir, "contours", "model_contours.json"),
        os.path.join(case_dir, "camera", "camera_new.txt"),
        os.path.join(case_dir, "T_view.txt"),
        width,
        height,
        contour_thickness=cfg.RENDER_CONTOUR_THICKNESS,
    )

    train_dir = os.path.join(case_dir, "sample", "train")
    val_dir = os.path.join(case_dir, "sample", "val")
    for split_dir in (train_dir, val_dir):
        ensure_dir(os.path.join(split_dir, "images"))

    valid_base, base_state = render_state(renderer, np.eye(4, dtype=np.float32), total_pixels, cfg)
    if not valid_base:
        raise RuntimeError("Base T_view pose is not visible enough for training pairs.")

    print("=== 03. Render explicit A/B pose pairs (RGB for DA2, no depth model) ===")
    print(f"  frame: {cfg.PATIENT_ID}/{cfg.FRAME_ID}")
    print(f"  seed: {cfg.SEED}")
    print(f"  pairs: {cfg.NUM_PAIRS}")
    print(
        "  pair mix: base_to_target=%.0f%%, local_refine=%.0f%%"
        % ((1.0 - cfg.LOCAL_REFINE_RATIO) * 100.0, cfg.LOCAL_REFINE_RATIO * 100.0)
    )
    print(
        "  local refine range: +/-%.1f mm, +/-%.1f deg"
        % (cfg.LOCAL_TRANS_RANGE_MM, cfg.LOCAL_ROT_RANGE_DEG)
    )

    split_data = {
        "train": {"dir": train_dir, "labels": {}, "pairs": {}},
        "val": {"dir": val_dir, "labels": {}, "pairs": {}},
    }
    max_attempts = cfg.MAX_ATTEMPTS if cfg.MAX_ATTEMPTS > 0 else cfg.NUM_PAIRS * 50
    accepted = 0
    rejected = 0
    attempts = 0
    pbar = tqdm(total=cfg.NUM_PAIRS, desc="Rendering pairs")

    while accepted < cfg.NUM_PAIRS and attempts < max_attempts:
        attempts += 1
        result = make_pair(renderer, base_state, total_pixels, cfg)
        if result is None:
            rejected += 1
            continue

        pair_type, state_A, state_B = result
        split = str(split_plan[accepted])
        pair_id = f"pair_{accepted:05d}"
        data = split_data[split]
        save_pair(data["dir"], data["labels"], data["pairs"], pair_id, pair_type, state_A, state_B)
        accepted += 1
        pbar.update(1)

    pbar.close()
    if accepted < cfg.NUM_PAIRS:
        raise RuntimeError(
            f"Only accepted {accepted}/{cfg.NUM_PAIRS} pairs after {attempts} attempts. "
            "Relax visibility filters or raise AR_MAX_ATTEMPTS."
        )

    for split, data in split_data.items():
        with open(os.path.join(data["dir"], "labels.json"), "w", encoding="utf-8") as f:
            json.dump(data["labels"], f, indent=2)
        with open(os.path.join(data["dir"], "pairs.json"), "w", encoding="utf-8") as f:
            json.dump(data["pairs"], f, indent=2)

    print(f"[OK] accepted pairs: {accepted}, rejected attempts: {rejected}")
    for split, data in split_data.items():
        print(f"[OK] {split}: states={len(data['labels'])}, pairs={len(data['pairs'])}")
    print("[Next] python 04_depth_anything_samples.py")


if __name__ == "__main__":
    main()
