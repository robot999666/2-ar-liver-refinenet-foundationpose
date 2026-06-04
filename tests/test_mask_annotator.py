import importlib.util
import os
import unittest

import cv2
import numpy as np


SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "scripts",
    "01b_annotate_intraop_masks.py",
)
SPEC = importlib.util.spec_from_file_location("mask_annotator_script", SCRIPT_PATH)
ANNOTATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANNOTATOR)


class MaskAnnotatorTests(unittest.TestCase):
    def test_full_and_depth_masks_have_distinct_semantics(self):
        annotator = ANNOTATOR.MaskAnnotator(np.zeros((100, 120, 3), np.uint8))
        annotator.full_polygon = [(10, 10), (100, 10), (100, 90), (10, 90)]
        annotator.occlusion_polygons = [[(40, 30), (70, 30), (70, 60), (40, 60)]]
        full, depth_valid = annotator.build_masks()
        self.assertGreater(np.count_nonzero(full), np.count_nonzero(depth_valid))
        self.assertEqual(depth_valid[45, 55], 0)
        self.assertGreater(full[45, 55], 0)

    def test_information_panel_does_not_accept_points(self):
        annotator = ANNOTATOR.MaskAnnotator(np.zeros((100, 120, 3), np.uint8))
        annotator.on_mouse(cv2.EVENT_LBUTTONDOWN, 130, 20, None, None)
        self.assertEqual(annotator.full_polygon, [])

    def test_render_has_fixed_information_panel(self):
        annotator = ANNOTATOR.MaskAnnotator(np.zeros((100, 120, 3), np.uint8))
        rendered = annotator.render()
        self.assertEqual(rendered.shape, (100, 120 + ANNOTATOR.INFO_PANEL_WIDTH, 3))


if __name__ == "__main__":
    unittest.main()

