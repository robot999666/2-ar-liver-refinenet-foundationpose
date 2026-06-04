import importlib.util
import json
import os
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
RUNTIME_ROOT = os.path.join(PROJECT_ROOT, "tests", "_runtime")
os.makedirs(RUNTIME_ROOT, exist_ok=True)

from shared.intraop_masks import mask_fingerprints, save_mask_bundle


SCRIPT_PATH = os.path.join(PROJECT_ROOT, "scripts", "07_infer_export_stl.py")
SPEC = importlib.util.spec_from_file_location("infer_script", SCRIPT_PATH)
INFER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INFER)


@contextmanager
def temporary_case():
    path = os.path.join(RUNTIME_ROOT, uuid.uuid4().hex)
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path)


class SafeConfig:
    SELECT_BEST_BY = "last"
    ACCEPT_BY_SCORE = False
    SCORE_PATIENCE = 0
    RAW_WORSEN_PATIENCE = 0
    ALLOW_UNVALIDATED_SCORE_CONTROL = False


def build_case(root):
    contour_dir = os.path.join(root, "contours")
    depth_dir = os.path.join(root, "depth")
    os.makedirs(contour_dir)
    os.makedirs(depth_dir)
    np.save(os.path.join(contour_dir, "gt_multicontour.npy"), np.zeros((3, 100, 100), np.float32))

    full = np.zeros((100, 100), np.uint8)
    full[20:80, 20:80] = 255
    depth_valid = np.zeros((100, 100), np.uint8)
    depth_valid[25:75, 25:75] = 255
    save_mask_bundle(root, np.zeros((100, 100, 3), np.uint8), full, depth_valid, {})

    yy, xx = np.indices((100, 100))
    depth = (yy + xx + 1).astype(np.float32)
    depth[depth_valid <= 0] = 0
    depth_path = os.path.join(depth_dir, "da2_intraop.npy")
    np.save(depth_path, depth)
    with open(os.path.join(depth_dir, "da2_intraop_quality.json"), "w", encoding="utf-8") as file:
        json.dump({"mask_fingerprints": mask_fingerprints(full, depth_valid)}, file)
    return depth_path, full, depth_valid


class InferSafetyTests(unittest.TestCase):
    def test_default_controls_are_allowed(self):
        INFER.validate_inference_control_config(SafeConfig)

    def test_proxy_selection_is_blocked(self):
        class Unsafe(SafeConfig):
            SELECT_BEST_BY = "contour"

        with self.assertRaises(ValueError):
            INFER.validate_inference_control_config(Unsafe)

    def test_explicit_ablation_can_enable_proxy_control(self):
        class Ablation(SafeConfig):
            SELECT_BEST_BY = "contour"
            ALLOW_UNVALIDATED_SCORE_CONTROL = True

        INFER.validate_inference_control_config(Ablation)

    def test_missing_masks_fail_before_tensor_creation(self):
        with temporary_case() as case_dir:
            os.makedirs(os.path.join(case_dir, "contours"))
            np.save(
                os.path.join(case_dir, "contours", "gt_multicontour.npy"),
                np.zeros((3, 100, 100), np.float32),
            )
            with self.assertRaises(Exception):
                INFER.load_real_B(
                    case_dir,
                    (100, 100),
                    os.path.join(case_dir, "depth", "da2_intraop.npy"),
                    return_observation=True,
                )

    def test_missing_depth_is_hard_error(self):
        with temporary_case() as case_dir:
            depth_path, _, _ = build_case(case_dir)
            os.remove(depth_path)
            with self.assertRaises(FileNotFoundError):
                INFER.load_real_B(case_dir, (100, 100), depth_path, return_observation=True)

    def test_stale_depth_mask_fingerprint_is_rejected(self):
        with temporary_case() as case_dir:
            depth_path, _, _ = build_case(case_dir)
            quality_path = os.path.join(case_dir, "depth", "da2_intraop_quality.json")
            with open(quality_path, "w", encoding="utf-8") as file:
                json.dump({"mask_fingerprints": {"wrong": "value"}}, file)
            with self.assertRaises(ValueError):
                INFER.load_real_B(case_dir, (100, 100), depth_path, return_observation=True)

    def test_unmasked_depth_is_rejected(self):
        with temporary_case() as case_dir:
            depth_path, _, _ = build_case(case_dir)
            depth = np.load(depth_path)
            depth[0, 0] = 10
            np.save(depth_path, depth)
            with self.assertRaises(Exception):
                INFER.load_real_B(case_dir, (100, 100), depth_path, return_observation=True)

    def test_network_mask_uses_full_liver_semantics(self):
        with temporary_case() as case_dir:
            depth_path, full, depth_valid = build_case(case_dir)
            _, depth_net, network_mask, _, depth_stats = INFER.load_real_B(
                case_dir,
                (100, 100),
                depth_path,
                return_observation=True,
            )
            self.assertTrue(np.array_equal(network_mask > 0, full > 0))
            self.assertEqual(np.count_nonzero(depth_net[depth_valid <= 0]), 0)
            self.assertEqual(depth_stats["nonzero_outside_depth_valid_mask"], 0)


if __name__ == "__main__":
    unittest.main()
