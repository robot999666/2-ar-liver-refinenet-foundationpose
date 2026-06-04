import importlib.util
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

    def test_parse_all_discovers_numeric_png_frames(self):
        with temporary_case() as data_root:
            lap_dir = os.path.join(data_root, "Dataset", "Patient3", "Lap Images")
            os.makedirs(lap_dir)
            for name in ("10.png", "02.png", "notes.txt", "03.jpg"):
                with open(os.path.join(lap_dir, name), "wb") as file:
                    file.write(b"")
            self.assertEqual(
                BATCH.parse_frames("all", data_root, "Patient3"),
                ["02", "10"],
            )

    def test_copy_shared_initial_pose_records_reference(self):
        with temporary_case() as case_root:
            source_dir = os.path.join(case_root, "Patient3", "02")
            target_dir = os.path.join(case_root, "Patient3", "03")
            os.makedirs(source_dir)
            os.makedirs(target_dir)
            pose = np.eye(4, dtype=np.float32)
            np.savetxt(os.path.join(source_dir, "T_view.txt"), pose)
            with open(os.path.join(source_dir, "T_view_meta.json"), "w", encoding="utf-8") as file:
                json.dump({"params": {"TX": 1.0}}, file)

            BATCH.copy_shared_initial_pose("Patient3", "02", "03", case_root=case_root)

            copied = np.loadtxt(os.path.join(target_dir, "T_view.txt"))
            self.assertTrue(np.allclose(copied, pose))
            with open(os.path.join(target_dir, "T_view_meta.json"), "r", encoding="utf-8") as file:
                metadata = json.load(file)
            self.assertEqual(metadata["frame_id"], "03")
            self.assertEqual(metadata["shared_initial_pose"]["source_frame_id"], "02")


if __name__ == "__main__":
    unittest.main()
