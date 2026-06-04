import importlib.util
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

from shared.intraop_masks import save_mask_bundle


SCRIPT_PATH = os.path.join(PROJECT_ROOT, "scripts", "08_batch_patient_frames.py")
SPEC = importlib.util.spec_from_file_location("batch_script", SCRIPT_PATH)
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


@contextmanager
def temporary_case():
    path = os.path.join(RUNTIME_ROOT, uuid.uuid4().hex)
    os.makedirs(path)
    try:
        yield path
    finally:
        shutil.rmtree(path)


class BatchMaskPreflightTests(unittest.TestCase):
    def test_valid_case_passes(self):
        with temporary_case() as case_dir:
            image_dir = os.path.join(case_dir, "image")
            os.makedirs(image_dir)
            image = np.zeros((100, 100, 3), np.uint8)
            image_path = os.path.join(image_dir, "lap_undist.png")
            cv2.imwrite(image_path, image)
            full = np.zeros((100, 100), np.uint8)
            full[20:80, 20:80] = 255
            depth_valid = np.zeros((100, 100), np.uint8)
            depth_valid[25:75, 25:75] = 255
            save_mask_bundle(case_dir, image, full, depth_valid, {})
            quality, summary = BATCH.preflight_case_masks(case_dir)
            self.assertTrue(quality["valid"])
            self.assertIn("full_ratio", summary)

    def test_missing_image_fails(self):
        with temporary_case() as case_dir:
            with self.assertRaises(FileNotFoundError):
                BATCH.preflight_case_masks(case_dir)

    def test_parse_frames(self):
        self.assertEqual(BATCH.parse_frames("2-4,06"), ["02", "03", "04", "06"])


if __name__ == "__main__":
    unittest.main()
