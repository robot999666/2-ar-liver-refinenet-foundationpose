import os
import json
import logging
import shutil
import sys
import csv

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from learning.models.refine_network import RefineNet

from shared import case_config
from shared.da2_engine import get_estimator
from shared.depth_augment import depth_for_network_eval
from shared.pose_render import PoseSampleRenderer


# ============================================================
# 07_infer_export_stl.py
# ------------------------------------------------------------
# B: 真实轮廓 + mask + 05 保存的 raw DA2 术中深度
# A: 当前位姿渲染 RGB -> DA2 raw -> 送入网络前仅做确定性 [0,255] 映射
# 训练时的随机深度增强不在推理阶段使用
#
# 直接运行：python 07_infer_export_stl.py
# ============================================================


class Config:
    # Current best profile for Patient1/02 experiments:
    #   1) Use a larger first coarse step: 0.4 * 0.75 = 0.30 of the network offset.
    #   2) Continue refinement with small steps: 0.10 of each later network offset.
    #   3) Stop when the model's own residual proxy stops improving.
    # This selection rule does not use MATLAB/TRE/IC ground truth during inference.
    INFER_PROFILE = os.environ.get("AR_INFER_PROFILE", "coarse0p30_refine0p10_rawstop")
    PATIENT_ID = case_config.PATIENT_ID
    FRAME_ID = case_config.FRAME_ID
    CASE_ROOT = case_config.CASE_ROOT
    DATA_ROOT = case_config.DATA_ROOT

    WEIGHT_PATH = os.environ.get("AR_WEIGHT_PATH", os.path.join(case_config.result_dir(), "best.pth"))
    IMG_SIZE = case_config.IMG_SIZE
    TRANS_SCALE = 50.0
    MAX_ITER = int(os.environ.get("AR_MAX_ITER", "10"))
    STOP_TRANS_MM = float(os.environ.get("AR_STOP_TRANS_MM", "0.2"))
    STOP_ROT_DEG = float(os.environ.get("AR_STOP_ROT_DEG", "0.2"))
    STOP_STREAK = int(os.environ.get("AR_STOP_STREAK", "2"))
    TRANS_DAMPING = float(os.environ.get("AR_TRANS_DAMPING", "0.4"))
    ROT_DAMPING = float(os.environ.get("AR_ROT_DAMPING", "0.4"))
    FIRST_STEP_DAMPING = float(os.environ.get("AR_FIRST_STEP_DAMPING", "0.75"))
    REFINE_TRANS_DAMPING = float(os.environ.get("AR_REFINE_TRANS_DAMPING", "0.1"))
    REFINE_ROT_DAMPING = float(os.environ.get("AR_REFINE_ROT_DAMPING", "0.1"))
    MAX_STEP_TRANS_MM = float(os.environ.get("AR_MAX_STEP_TRANS_MM", "0.0"))
    MAX_STEP_ROT_DEG = float(os.environ.get("AR_MAX_STEP_ROT_DEG", "0.0"))
    DIVERGE_RAW_TRANS_MM = float(os.environ.get("AR_DIVERGE_RAW_TRANS_MM", "120.0"))
    DIVERGE_RAW_ROT_DEG = float(os.environ.get("AR_DIVERGE_RAW_ROT_DEG", "45.0"))
    DIVERGE_POSE_TRANS_MM = float(os.environ.get("AR_DIVERGE_POSE_TRANS_MM", "200.0"))
    STOP_DRIFT_MM = float(os.environ.get("AR_STOP_DRIFT_MM", "0.0"))
    RAW_SCORE_ROT_WEIGHT = float(os.environ.get("AR_RAW_SCORE_ROT_WEIGHT", "5.0"))
    RAW_WORSEN_PATIENCE = int(os.environ.get("AR_RAW_WORSEN_PATIENCE", "1"))
    RAW_WORSEN_MIN_DELTA = float(os.environ.get("AR_RAW_WORSEN_MIN_DELTA", "0.5"))
    SELECT_BEST_BY = os.environ.get("AR_SELECT_BEST_BY", "last").lower()
    ACCEPT_BY_SCORE = os.environ.get("AR_ACCEPT_BY_SCORE", "0").lower() in ("1", "true", "yes")
    SCORE_ACCEPT_TOL = float(os.environ.get("AR_SCORE_ACCEPT_TOL", "0.0"))
    SCORE_PATIENCE = int(os.environ.get("AR_SCORE_PATIENCE", "0"))
    CONTOUR_SCORE_MAX_DIST_PX = float(os.environ.get("AR_CONTOUR_SCORE_MAX_DIST_PX", "50.0"))
    CONTOUR_SCORE_MASK_WEIGHT = float(os.environ.get("AR_CONTOUR_SCORE_MASK_WEIGHT", "20.0"))
    EVAL_METHOD_NAME = os.environ.get("AR_EVAL_METHOD_NAME", "FoundationPose")
    EVAL_ROOT = os.environ.get("AR_EVAL_ROOT", os.path.join(case_config.result_dir(), "evaluation"))
    MATLAB_REGISTRATION_ROOT = os.environ.get(
        "AR_MATLAB_REGISTRATION_ROOT",
        os.path.join(DATA_ROOT, "Registration Methods"),
    )
    UPDATE_MODE = os.environ.get("AR_UPDATE_MODE", "right")
    PRED_SUBDIR = os.environ.get("AR_PRED_SUBDIR", "pred")
    EXPORT_EVAL_STL = os.environ.get("AR_EXPORT_EVAL_STL", "1").lower() in ("1", "true", "yes")
    ALLOW_ZERO_DEPTH = os.environ.get("AR_ALLOW_ZERO_DEPTH", "0").lower() in ("1", "true", "yes")


class DummyCfg:
    use_BN = True
    rot_rep = "6d"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_logger(log_path):
    logger = logging.getLogger("InferExportSTL")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def make_mask_from_silhouette(contour_map):
    sil = (contour_map > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(sil)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return (filled > 0).astype(np.float32)


def _binary_chamfer_score(src, dst, max_dist_px):
    src = src.astype(bool)
    dst = dst.astype(bool)
    if not np.any(src) and not np.any(dst):
        return 0.0
    if not np.any(src) or not np.any(dst):
        return float(max_dist_px)

    src_inv = (~src).astype(np.uint8)
    dst_inv = (~dst).astype(np.uint8)
    dist_to_dst = cv2.distanceTransform(dst_inv, cv2.DIST_L2, 3)
    dist_to_src = cv2.distanceTransform(src_inv, cv2.DIST_L2, 3)
    src_to_dst = float(np.mean(np.clip(dist_to_dst[src], 0.0, max_dist_px)))
    dst_to_src = float(np.mean(np.clip(dist_to_src[dst], 0.0, max_dist_px)))
    return 0.5 * (src_to_dst + dst_to_src)


def compute_contour_score(rendered_contours, rendered_mask, target_contours, target_mask, max_dist_px, mask_weight):
    channel_scores = []
    for ch in range(3):
        channel_scores.append(
            _binary_chamfer_score(
                rendered_contours[ch] > 0,
                target_contours[ch] > 0,
                max_dist_px,
            )
        )

    rendered_mask = rendered_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    union = np.logical_or(rendered_mask, target_mask)
    if np.any(union):
        iou = float(np.count_nonzero(np.logical_and(rendered_mask, target_mask)) / np.count_nonzero(union))
    else:
        iou = 0.0

    contour_score = float(np.mean(channel_scores))
    score = contour_score + float(mask_weight) * (1.0 - iou)
    parts = {
        "score": score,
        "contour_score": contour_score,
        "mask_iou": iou,
        "silhouette_score": float(channel_scores[0]),
        "ridge_score": float(channel_scores[1]),
        "ligament_score": float(channel_scores[2]),
    }
    return score, parts


def pack_modalities(contours, depth, mask):
    w, h = Config.IMG_SIZE
    contours_rs = np.stack(
        [cv2.resize(contours[i], (w, h), interpolation=cv2.INTER_NEAREST) for i in range(3)], axis=0
    )
    depth_rs = cv2.resize(depth.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_rs = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    x = np.concatenate([contours_rs, depth_rs[None], mask_rs[None]], axis=0)
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


def compute_rotation_matrix_from_6d(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = F.normalize(x_raw, p=2, dim=-1)
    z_dot_x = (y_raw * x).sum(dim=-1, keepdim=True)
    y = F.normalize(y_raw - z_dot_x * x, p=2, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


class InferenceRenderer:
    def __init__(self, case_dir, width, height, da2_estimator):
        self.width = width
        self.height = height
        self.da2 = da2_estimator
        self.pose_renderer = PoseSampleRenderer(
            os.path.join(case_dir, "models", "Liver_obj.ply"),
            os.path.join(case_dir, "models", "Liver_vertices_centered_raw_order.npy"),
            os.path.join(case_dir, "contours", "model_contours.json"),
            os.path.join(case_dir, "camera", "camera_new.txt"),
            os.path.join(case_dir, "T_view.txt"),
            width,
            height,
        )

    def render_network_input(self, T_obj_to_cam):
        contours, mask, rgb_bgr, _, _, _ = self.pose_renderer.render_at_pose(T_obj_to_cam)
        depth_raw = self.da2.infer(rgb_bgr)
        mask_f = (mask > 127).astype(np.float32)
        depth_raw[mask_f <= 0] = 0.0
        depth_net = depth_for_network_eval(depth_raw, mask_f)
        return pack_modalities(
            contours.astype(np.float32) / 255.0,
            depth_net,
            mask_f,
        )

    def render_observation(self, T_obj_to_cam):
        contours, mask, _, _, _, _ = self.pose_renderer.render_at_pose(T_obj_to_cam)
        return contours, (mask > 127).astype(np.float32)


def load_real_B(case_dir, image_shape, intraop_depth_path, allow_zero_depth=False, logger=None, return_observation=False):
    h, w = image_shape
    contours = np.load(os.path.join(case_dir, "contours", "gt_multicontour.npy")).astype(np.float32)
    if contours.shape[1:] != (h, w):
        contours = np.stack(
            [cv2.resize(contours[i], (w, h), interpolation=cv2.INTER_NEAREST) for i in range(3)], axis=0
        )

    sil = (contours[0] > 0).astype(np.uint8)
    mask = make_mask_from_silhouette(sil)

    if os.path.exists(intraop_depth_path):
        depth_raw = np.load(intraop_depth_path).astype(np.float32)
        if depth_raw.shape != (h, w):
            depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        msg = f"Missing intra-op depth: {intraop_depth_path}. Run 05_depth_anything_intraop.py first."
        if not allow_zero_depth:
            raise FileNotFoundError(msg)
        msg = f"[WARN] {msg} AR_ALLOW_ZERO_DEPTH is enabled, using zeros."
        if logger is None:
            print(msg)
        else:
            logger.warning(msg)
        depth_raw = np.zeros((h, w), dtype=np.float32)

    depth_raw[mask <= 0] = 0.0
    depth_net = depth_for_network_eval(depth_raw, mask)
    tensor = pack_modalities(contours, depth_net, mask)
    if return_observation:
        return tensor, contours, mask
    return tensor


def rotation_angle_deg(R):
    trace = np.trace(R)
    val = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def make_transform(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def damp_offset(R, t, trans_factor, rot_factor):
    trans_factor = float(np.clip(trans_factor, 0.0, 1.0))
    rot_factor = float(np.clip(rot_factor, 0.0, 1.0))
    rotvec, _ = cv2.Rodrigues(R.astype(np.float64))
    R_damped, _ = cv2.Rodrigues(rotvec * rot_factor)
    t_damped = t.astype(np.float32) * trans_factor
    return R_damped.astype(np.float32), t_damped.astype(np.float32)


def clip_damped_offset(R, t, max_trans_mm, max_rot_deg):
    clipped = False
    t_out = t.astype(np.float32).copy()
    trans_norm = float(np.linalg.norm(t_out))
    if max_trans_mm > 0 and trans_norm > max_trans_mm:
        t_out *= float(max_trans_mm / max(trans_norm, 1e-8))
        clipped = True

    R_out = R.astype(np.float32)
    rot_deg = rotation_angle_deg(R_out)
    if max_rot_deg > 0 and rot_deg > max_rot_deg:
        rotvec, _ = cv2.Rodrigues(R_out.astype(np.float64))
        R_out, _ = cv2.Rodrigues(rotvec * float(max_rot_deg / max(rot_deg, 1e-8)))
        R_out = R_out.astype(np.float32)
        clipped = True

    return R_out, t_out, clipped


def raw_update_score(raw_trans_mm, raw_rot_deg, rot_weight):
    return float(raw_trans_mm + raw_rot_deg * rot_weight)


def write_history_csv(path, history):
    fieldnames = [
        "iter",
        "accepted",
        "raw_trans_mm",
        "raw_rot_deg",
        "raw_score",
        "damped_trans_mm",
        "damped_rot_deg",
        "trans_factor",
        "rot_factor",
        "step_clipped",
        "pose_drift_mm",
        "contour_score",
        "mask_iou",
        "silhouette_score",
        "ridge_score",
        "ligament_score",
        "is_best_contour",
        "stop_trigger",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            parts = row.get("contour_parts") or {}
            writer.writerow(
                {
                    "iter": row.get("iter"),
                    "accepted": int(bool(row.get("accepted", True))),
                    "raw_trans_mm": row.get("raw_trans_mm"),
                    "raw_rot_deg": row.get("raw_rot_deg"),
                    "raw_score": row.get("raw_score"),
                    "damped_trans_mm": row.get("damped_trans_mm"),
                    "damped_rot_deg": row.get("damped_rot_deg"),
                    "trans_factor": row.get("trans_factor"),
                    "rot_factor": row.get("rot_factor"),
                    "step_clipped": int(bool(row.get("step_clipped", False))),
                    "pose_drift_mm": row.get("pose_drift_mm"),
                    "contour_score": row.get("contour_score"),
                    "mask_iou": parts.get("mask_iou"),
                    "silhouette_score": parts.get("silhouette_score"),
                    "ridge_score": parts.get("ridge_score"),
                    "ligament_score": parts.get("ligament_score"),
                    "is_best_contour": int(bool(row.get("is_best_contour", False))),
                    "stop_trigger": row.get("stop_trigger", ""),
                }
            )


def export_transformed_mesh(mesh_path, T_obj_to_cam, out_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    verts = np.asarray(mesh.vertices).astype(np.float64)
    verts_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    verts_cam = (T_obj_to_cam @ verts_h.T).T[:, :3]
    mesh.vertices = o3d.utility.Vector3dVector(verts_cam)
    ensure_dir(os.path.dirname(out_path))
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    ext = os.path.splitext(out_path)[1].lower()
    ok = o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=(ext != ".stl"))
    if not ok:
        raise RuntimeError(f"Failed to write mesh: {out_path}")


def apply_update(T_curr, T_offset, mode):
    if mode == "right":
        return T_curr @ T_offset
    if mode == "left":
        return T_offset @ T_curr
    if mode == "inverse_right":
        return T_curr @ np.linalg.inv(T_offset)
    raise ValueError(f"Unknown UPDATE_MODE: {mode}")


def save_debug_state(debug_dir, iter_idx, T_pose, T_offset, tumour_path):
    name = f"iter_{iter_idx:02d}"
    np.savetxt(os.path.join(debug_dir, f"{name}_pose.txt"), T_pose, fmt="%.10f")
    if T_offset is not None:
        np.savetxt(os.path.join(debug_dir, f"{name}_offset.txt"), T_offset, fmt="%.10f")
    export_transformed_mesh(tumour_path, T_pose, os.path.join(debug_dir, f"{name}_tumour.stl"))


def main():
    cfg = Config()
    case_dir = case_config.case_dir(cfg.CASE_ROOT)
    image_path = os.path.join(case_dir, "image", "lap_undist.png")
    tview_path = os.path.join(case_dir, "T_view.txt")
    liver_path = os.path.join(case_dir, "models", "Liver_obj.ply")
    tumour_path = os.path.join(case_dir, "models", "Tumour_obj.ply")
    intraop_depth = os.path.join(case_config.depth_dir(cfg.CASE_ROOT), "da2_intraop.npy")
    pred_dir = os.path.join(case_dir, cfg.PRED_SUBDIR)
    debug_dir = os.path.join(pred_dir, "debug_iterations")
    ensure_dir(pred_dir)
    if os.path.isdir(debug_dir):
        shutil.rmtree(debug_dir)
    ensure_dir(debug_dir)
    logger = setup_logger(os.path.join(pred_dir, "infer.log"))

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RefineNet(cfg=DummyCfg(), c_in=5).to(device)
    model.load_state_dict(torch.load(cfg.WEIGHT_PATH, map_location=device))
    model.eval()

    da2 = get_estimator()
    renderer = InferenceRenderer(case_dir, w, h, da2)
    B_tensor, B_contours, B_mask = load_real_B(
        case_dir,
        (h, w),
        intraop_depth,
        allow_zero_depth=cfg.ALLOW_ZERO_DEPTH,
        logger=logger,
        return_observation=True,
    )
    B_tensor = B_tensor.to(device)

    T_curr = np.loadtxt(tview_path).astype(np.float32)
    T_init = T_curr.copy()

    logger.info("=== 07. Inference + STL export ===")
    logger.info(f"  infer profile: {cfg.INFER_PROFILE}")
    logger.info(f"  frame: {cfg.PATIENT_ID}/{cfg.FRAME_ID}")
    logger.info(f"  B depth: {intraop_depth}")
    logger.info(f"  allow zero depth: {cfg.ALLOW_ZERO_DEPTH}")
    logger.info(f"  weight: {cfg.WEIGHT_PATH}")
    logger.info(f"  UPDATE_MODE: {cfg.UPDATE_MODE}")
    logger.info(
        "  damping: first_base_trans=%.3f, first_base_rot=%.3f, first_step=%.3f, refine_trans=%.3f, refine_rot=%.3f, max_iter=%d",
        cfg.TRANS_DAMPING,
        cfg.ROT_DAMPING,
        cfg.FIRST_STEP_DAMPING,
        cfg.REFINE_TRANS_DAMPING,
        cfg.REFINE_ROT_DAMPING,
        cfg.MAX_ITER,
    )
    logger.info(
        "  step clip: max_trans=%.3f mm, max_rot=%.3f deg (0 disables)",
        cfg.MAX_STEP_TRANS_MM,
        cfg.MAX_STEP_ROT_DEG,
    )
    logger.info(
        "  stop: raw_trans<%.3f mm and raw_rot<%.3f deg for %d consecutive step(s)",
        cfg.STOP_TRANS_MM,
        cfg.STOP_ROT_DEG,
        cfg.STOP_STREAK,
    )
    logger.info(
        "  diverge: raw_trans>%.1f mm or raw_rot>%.1f deg or pose_drift>%.1f mm",
        cfg.DIVERGE_RAW_TRANS_MM,
        cfg.DIVERGE_RAW_ROT_DEG,
        cfg.DIVERGE_POSE_TRANS_MM,
    )
    logger.info(
        "  extra stops: stop_drift=%.1f mm, raw_worsen_patience=%d, raw_score_rot_weight=%.2f",
        cfg.STOP_DRIFT_MM,
        cfg.RAW_WORSEN_PATIENCE,
        cfg.RAW_SCORE_ROT_WEIGHT,
    )
    logger.info(
        "  selection: best_by=%s, accept_by_score=%s, score_patience=%d",
        cfg.SELECT_BEST_BY,
        cfg.ACCEPT_BY_SCORE,
        cfg.SCORE_PATIENCE,
    )
    logger.info(f"  log: {os.path.join(pred_dir, 'infer.log')}")

    save_debug_state(debug_dir, 0, T_curr, None, tumour_path)
    contours0, mask0 = renderer.render_observation(T_curr)
    best_score, best_parts = compute_contour_score(
        contours0,
        mask0,
        B_contours,
        B_mask,
        cfg.CONTOUR_SCORE_MAX_DIST_PX,
        cfg.CONTOUR_SCORE_MASK_WEIGHT,
    )
    best_iter = 0
    best_pose = T_curr.copy()
    best_offset = None
    current_score = best_score
    no_score_improve_count = 0
    history = [
        {
            "iter": 0,
            "raw_trans_mm": 0.0,
            "raw_rot_deg": 0.0,
            "raw_score": 0.0,
            "damped_trans_mm": 0.0,
            "damped_rot_deg": 0.0,
            "pose_drift_mm": 0.0,
            "contour_score": float(best_score),
            "contour_parts": best_parts,
            "is_best_contour": True,
            "accepted": True,
            "step_clipped": False,
            "stop_trigger": "",
        }
    ]
    logger.info(
        "Iter 00: contour_score=%.4f mask_iou=%.4f",
        best_parts["score"],
        best_parts["mask_iou"],
    )
    small_step_count = 0
    best_raw_score = None
    raw_worsen_count = 0
    stop_reason = None

    with torch.no_grad():
        for it in range(cfg.MAX_ITER):
            A_tensor = renderer.render_network_input(T_curr).to(device)
            out = model(A_tensor, B_tensor)
            R = compute_rotation_matrix_from_6d(out["rot"]).squeeze(0).cpu().numpy()
            t = out["trans"].squeeze(0).cpu().numpy() * cfg.TRANS_SCALE

            iter_idx = it + 1
            raw_trans_norm = float(np.linalg.norm(t))
            raw_rot_deg = rotation_angle_deg(R)
            raw_score = raw_update_score(raw_trans_norm, raw_rot_deg, cfg.RAW_SCORE_ROT_WEIGHT)
            if (not np.isfinite(raw_trans_norm)) or (not np.isfinite(raw_rot_deg)):
                stop_reason = "diverged_nan_offset"
                logger.warning("[STOP] Iter %02d produced non-finite offset.", iter_idx)
                break
            if raw_trans_norm > cfg.DIVERGE_RAW_TRANS_MM or raw_rot_deg > cfg.DIVERGE_RAW_ROT_DEG:
                stop_reason = "diverged_raw_offset"
                logger.warning(
                    "[STOP] Iter %02d raw offset too large: trans=%.4f mm, rot=%.4f deg",
                    iter_idx,
                    raw_trans_norm,
                    raw_rot_deg,
                )
                break

            if iter_idx == 1:
                trans_factor = cfg.TRANS_DAMPING * cfg.FIRST_STEP_DAMPING
                rot_factor = cfg.ROT_DAMPING * cfg.FIRST_STEP_DAMPING
            else:
                trans_factor = cfg.REFINE_TRANS_DAMPING
                rot_factor = cfg.REFINE_ROT_DAMPING
            R_damped, t_damped = damp_offset(R, t, trans_factor, rot_factor)
            R_damped, t_damped, step_clipped = clip_damped_offset(
                R_damped,
                t_damped,
                cfg.MAX_STEP_TRANS_MM,
                cfg.MAX_STEP_ROT_DEG,
            )
            T_offset = make_transform(R_damped, t_damped)
            damped_trans_norm = float(np.linalg.norm(t_damped))
            damped_rot_deg = rotation_angle_deg(R_damped)

            T_next = apply_update(T_curr, T_offset, cfg.UPDATE_MODE)
            pose_drift = float(np.linalg.norm(T_next[:3, 3] - T_init[:3, 3]))
            if not np.all(np.isfinite(T_next)):
                stop_reason = "diverged_nan_pose"
                logger.warning("[STOP] Iter %02d produced non-finite pose.", iter_idx)
                break
            if pose_drift > cfg.DIVERGE_POSE_TRANS_MM:
                save_debug_state(debug_dir, iter_idx, T_next, T_offset, tumour_path)
                stop_reason = "diverged_pose"
                logger.warning(
                    "[STOP] Iter %02d pose drift too large: drift=%.4f mm",
                    iter_idx,
                    pose_drift,
                )
                break
            if cfg.STOP_DRIFT_MM > 0 and pose_drift > cfg.STOP_DRIFT_MM:
                save_debug_state(debug_dir, iter_idx, T_next, T_offset, tumour_path)
                history.append(
                    {
                        "iter": iter_idx,
                        "raw_trans_mm": raw_trans_norm,
                        "raw_rot_deg": raw_rot_deg,
                        "raw_score": raw_score,
                        "damped_trans_mm": damped_trans_norm,
                        "damped_rot_deg": damped_rot_deg,
                        "trans_factor": float(np.clip(trans_factor, 0.0, 1.0)),
                        "rot_factor": float(np.clip(rot_factor, 0.0, 1.0)),
                        "pose_drift_mm": pose_drift,
                        "contour_score": None,
                        "contour_parts": {},
                        "is_best_contour": False,
                        "accepted": False,
                        "step_clipped": bool(step_clipped),
                        "stop_trigger": "stop_drift",
                    }
                )
                stop_reason = "stop_drift"
                logger.info(
                    "[STOP] Iter %02d exceeded stop drift: drift=%.4f mm > %.4f mm",
                    iter_idx,
                    pose_drift,
                    cfg.STOP_DRIFT_MM,
                )
                break

            contours_next, mask_next = renderer.render_observation(T_next)
            contour_score, contour_parts = compute_contour_score(
                contours_next,
                mask_next,
                B_contours,
                B_mask,
                cfg.CONTOUR_SCORE_MAX_DIST_PX,
                cfg.CONTOUR_SCORE_MASK_WEIGHT,
            )
            is_best_contour = contour_score < best_score
            if is_best_contour:
                best_score = contour_score
                best_parts = contour_parts
                best_iter = iter_idx
                best_pose = T_next.copy()
                best_offset = T_offset.copy()
                no_score_improve_count = 0
            else:
                no_score_improve_count += 1

            accepted = True
            if cfg.ACCEPT_BY_SCORE and contour_score > current_score + cfg.SCORE_ACCEPT_TOL:
                accepted = False

            save_debug_state(debug_dir, iter_idx, T_next, T_offset, tumour_path)
            history.append(
                {
                    "iter": iter_idx,
                    "raw_trans_mm": raw_trans_norm,
                    "raw_rot_deg": raw_rot_deg,
                    "raw_score": raw_score,
                    "damped_trans_mm": damped_trans_norm,
                    "damped_rot_deg": damped_rot_deg,
                    "trans_factor": float(np.clip(trans_factor, 0.0, 1.0)),
                    "rot_factor": float(np.clip(rot_factor, 0.0, 1.0)),
                    "pose_drift_mm": pose_drift,
                    "contour_score": float(contour_score),
                    "contour_parts": contour_parts,
                    "is_best_contour": bool(is_best_contour),
                    "accepted": bool(accepted),
                    "step_clipped": bool(step_clipped),
                    "stop_trigger": "",
                }
            )
            logger.info(
                (
                    "Iter %02d: raw_trans=%.4f mm raw_rot=%.4f deg | "
                    "raw_score=%.4f | damped_trans=%.4f mm damped_rot=%.4f deg%s | drift=%.4f mm | "
                    "contour_score=%.4f mask_iou=%.4f%s"
                ),
                iter_idx,
                raw_trans_norm,
                raw_rot_deg,
                raw_score,
                damped_trans_norm,
                damped_rot_deg,
                " clipped" if step_clipped else "",
                pose_drift,
                contour_parts["score"],
                contour_parts["mask_iou"],
                " [best]" if is_best_contour else "",
            )

            if not accepted:
                stop_reason = "score_rejected"
                logger.info(
                    "[STOP] Iter %02d rejected by contour score: current=%.4f candidate=%.4f",
                    iter_idx,
                    current_score,
                    contour_score,
                )
                break

            T_curr = T_next
            current_score = contour_score

            if raw_trans_norm < cfg.STOP_TRANS_MM and raw_rot_deg < cfg.STOP_ROT_DEG:
                small_step_count += 1
            else:
                small_step_count = 0
            if small_step_count >= cfg.STOP_STREAK:
                stop_reason = "converged_small_offset"
                logger.info("[OK] convergence threshold met.")
                break

            if best_raw_score is None or raw_score < best_raw_score - cfg.RAW_WORSEN_MIN_DELTA:
                best_raw_score = raw_score
                raw_worsen_count = 0
            else:
                raw_worsen_count += 1
            if cfg.RAW_WORSEN_PATIENCE > 0 and raw_worsen_count >= cfg.RAW_WORSEN_PATIENCE:
                stop_reason = "raw_score_no_improve"
                history[-1]["stop_trigger"] = "raw_score_no_improve"
                logger.info(
                    "[OK] raw update score did not improve for %d step(s). best=%.4f current=%.4f",
                    cfg.RAW_WORSEN_PATIENCE,
                    best_raw_score,
                    raw_score,
                )
                break
            if cfg.SCORE_PATIENCE > 0 and no_score_improve_count >= cfg.SCORE_PATIENCE:
                stop_reason = "contour_score_patience"
                history[-1]["stop_trigger"] = "contour_score_patience"
                logger.info(
                    "[OK] contour score did not improve for %d step(s).",
                    cfg.SCORE_PATIENCE,
                )
                break

    if stop_reason is None:
        stop_reason = "max_iter_reached"
        logger.info("[OK] max_iter reached.")

    with open(os.path.join(debug_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    write_history_csv(os.path.join(debug_dir, "history.csv"), history)

    if cfg.SELECT_BEST_BY == "last":
        accepted_history = [row for row in history if row.get("accepted", True)]
        selected_iter = int(accepted_history[-1]["iter"])
        selected_pose = T_curr
        selected_score = float(accepted_history[-1].get("contour_score", best_score) or best_score)
        selected_reason = "last"
    elif cfg.SELECT_BEST_BY == "contour":
        selected_iter = best_iter
        selected_pose = best_pose
        selected_score = best_score
        selected_reason = "lowest_contour_score"
    else:
        raise ValueError(f"Unknown AR_SELECT_BEST_BY={cfg.SELECT_BEST_BY!r}; use 'contour' or 'last'.")

    selection = {
        "selected_iter": int(selected_iter),
        "selected_reason": selected_reason,
        "selected_contour_score": float(selected_score),
        "best_contour_iter": int(best_iter),
        "best_contour_score": float(best_score),
        "best_contour_parts": best_parts,
        "stop_reason": stop_reason,
        "select_best_by": cfg.SELECT_BEST_BY,
        "accept_by_score": bool(cfg.ACCEPT_BY_SCORE),
        "config": {
            "max_iter": cfg.MAX_ITER,
            "update_mode": cfg.UPDATE_MODE,
            "trans_damping": cfg.TRANS_DAMPING,
            "rot_damping": cfg.ROT_DAMPING,
            "first_step_damping": cfg.FIRST_STEP_DAMPING,
            "refine_trans_damping": cfg.REFINE_TRANS_DAMPING,
            "refine_rot_damping": cfg.REFINE_ROT_DAMPING,
            "max_step_trans_mm": cfg.MAX_STEP_TRANS_MM,
            "max_step_rot_deg": cfg.MAX_STEP_ROT_DEG,
            "stop_drift_mm": cfg.STOP_DRIFT_MM,
            "raw_worsen_patience": cfg.RAW_WORSEN_PATIENCE,
            "raw_worsen_min_delta": cfg.RAW_WORSEN_MIN_DELTA,
        },
    }
    with open(os.path.join(debug_dir, "best_selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    np.savetxt(os.path.join(pred_dir, "last_pose.txt"), T_curr, fmt="%.10f")
    np.savetxt(os.path.join(pred_dir, "contour_best_pose.txt"), best_pose, fmt="%.10f")
    if best_offset is not None:
        np.savetxt(os.path.join(pred_dir, "contour_best_offset.txt"), best_offset, fmt="%.10f")
    np.savetxt(os.path.join(pred_dir, "final_pose.txt"), selected_pose, fmt="%.10f")
    tumour_pred = os.path.join(pred_dir, "tumour_pred_cam.stl")
    export_transformed_mesh(tumour_path, selected_pose, tumour_pred)
    export_transformed_mesh(liver_path, selected_pose, os.path.join(pred_dir, "liver_pred_cam.ply"))

    eval_stl = None
    matlab_eval_stl = None
    if cfg.EXPORT_EVAL_STL:
        eval_dir = os.path.join(cfg.EVAL_ROOT, "Registration Methods", cfg.EVAL_METHOD_NAME, cfg.PATIENT_ID)
        ensure_dir(eval_dir)
        eval_stl = os.path.join(eval_dir, f"{cfg.FRAME_ID}.stl")
        shutil.copyfile(tumour_pred, eval_stl)

        matlab_eval_dir = os.path.join(cfg.MATLAB_REGISTRATION_ROOT, cfg.EVAL_METHOD_NAME, cfg.PATIENT_ID)
        ensure_dir(matlab_eval_dir)
        matlab_eval_stl = os.path.join(matlab_eval_dir, f"{cfg.FRAME_ID}.stl")
        shutil.copyfile(tumour_pred, matlab_eval_stl)

    logger.info(f"[OK] stop_reason: {stop_reason}")
    logger.info(
        "[OK] selected_iter: %02d (%s), contour_score=%.4f",
        selected_iter,
        selected_reason,
        selected_score,
    )
    logger.info(f"[OK] final_pose: {os.path.join(pred_dir, 'final_pose.txt')}")
    if cfg.EXPORT_EVAL_STL:
        logger.info(f"[OK] Evaluation STL: {eval_stl}")
        logger.info(f"[OK] MATLAB evaluation STL: {matlab_eval_stl}")
    else:
        logger.info("[OK] Evaluation STL export disabled for this run.")


if __name__ == "__main__":
    main()
