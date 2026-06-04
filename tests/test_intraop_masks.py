import json
import os
import shutil
import sys
import unittest
import uuid
from contextlib import contextmanager

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
RUNTIME_ROOT = os.path.join(PROJECT_ROOT, "tests", "_runtime")
os.makedirs(RUNTIME_ROOT, exist_ok=True)

from shared.intraop_masks import (
    MaskValidationError,
    depth_quality,
    mask_paths,
    save_mask_bundle,
    validate_case_masks,
    validate_depth_quality,
    validate_mask_arrays,
)


@contextmanager
def temporary_case():
    path = os.path.join(RUNTIME_ROOT, uuid.uuid4().hex)
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def valid_masks(shape=(100, 100)):
    full = np.zeros(shape, dtype=np.uint8)
    full[20:80, 20:80] = 255
    depth_valid = np.zeros(shape, dtype=np.uint8)
    depth_valid[25:75, 25:75] = 255
    return full, depth_valid


class IntraopMaskTests(unittest.TestCase):
    def test_valid_arrays(self):
        full, depth_valid = valid_masks()
        _, _, quality = validate_mask_arrays(full, depth_valid)
        self.assertTrue(quality["valid"])
        self.assertEqual(quality["depth_valid_outside_full_pixels"], 0)

    def test_empty_full_mask_is_rejected(self):
        with self.assertRaises(MaskValidationError):
            validate_mask_arrays(np.zeros((100, 100), np.uint8), np.zeros((100, 100), np.uint8))

    def test_fragmented_full_mask_is_rejected(self):
        full = np.zeros((100, 100), np.uint8)
        full[10:40, 10:40] = 255
        full[60:90, 60:90] = 255
        with self.assertRaises(MaskValidationError):
            validate_mask_arrays(full, full)

    def test_depth_valid_outside_full_is_rejected(self):
        full, depth_valid = valid_masks()
        depth_valid[0:5, 0:5] = 255
        with self.assertRaises(MaskValidationError):
            validate_mask_arrays(full, depth_valid)

    def test_save_and_validate_bundle(self):
        with temporary_case() as case_dir:
            image = np.zeros((100, 100, 3), np.uint8)
            full, depth_valid = valid_masks()
            paths, _ = save_mask_bundle(case_dir, image, full, depth_valid, {"source": "test"})
            loaded_full, loaded_depth, quality = validate_case_masks(case_dir, (100, 100))
            self.assertTrue(quality["valid"])
            self.assertTrue(np.array_equal(loaded_full, full))
            self.assertTrue(np.array_equal(loaded_depth, depth_valid))
            for name in ("annotation", "quality", "overlay"):
                self.assertTrue(os.path.exists(paths[name]))

    def test_manual_png_edit_breaks_provenance(self):
        with temporary_case() as case_dir:
            image = np.zeros((100, 100, 3), np.uint8)
            full, depth_valid = valid_masks()
            save_mask_bundle(case_dir, image, full, depth_valid, {})
            full[30, 30] = 0
            cv2.imwrite(mask_paths(case_dir)["full_liver_mask"], full)
            with self.assertRaises(MaskValidationError):
                validate_case_masks(case_dir, (100, 100))

    def test_missing_overlay_is_rejected(self):
        with temporary_case() as case_dir:
            image = np.zeros((100, 100, 3), np.uint8)
            full, depth_valid = valid_masks()
            save_mask_bundle(case_dir, image, full, depth_valid, {})
            os.remove(mask_paths(case_dir)["overlay"])
            with self.assertRaises(MaskValidationError):
                validate_case_masks(case_dir, (100, 100))

    def test_valid_masked_depth(self):
        full, depth_valid = valid_masks()
        raw = np.ones((100, 100), np.float32)
        masked = raw.copy()
        masked[depth_valid <= 0] = 0
        stats = depth_quality(raw, masked, full, depth_valid)
        validate_depth_quality(stats)
        self.assertTrue(stats["valid"])

    def test_depth_outside_valid_mask_is_rejected(self):
        full, depth_valid = valid_masks()
        depth = np.ones((100, 100), np.float32)
        stats = depth_quality(depth, depth, full, depth_valid)
        with self.assertRaises(MaskValidationError):
            validate_depth_quality(stats)

    def test_low_depth_coverage_is_rejected(self):
        full, depth_valid = valid_masks()
        depth = np.zeros((100, 100), np.float32)
        depth[25:50, 25:75] = 1
        stats = depth_quality(depth, depth, full, depth_valid)
        with self.assertRaises(MaskValidationError):
            validate_depth_quality(stats)


if __name__ == "__main__":
    unittest.main()
