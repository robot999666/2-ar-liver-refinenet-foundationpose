"""交互式标注真实图像中的完整肝脏区域和器械遮挡区域。"""

import argparse
import json
import os
import sys
import textwrap

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config
from shared.intraop_masks import MaskValidationError, mask_paths, quality_summary, save_mask_bundle


WINDOW_NAME = "Intra-op liver mask annotation"
INFO_PANEL_WIDTH = 360


def polygon_mask(shape, polygon):
    mask = np.zeros(shape, dtype=np.uint8)
    if len(polygon) >= 3:
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def polygons_mask(shape, polygons):
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
    return mask


class MaskAnnotator:
    def __init__(self, image):
        self.image = image
        self.shape = image.shape[:2]
        self.stage = "full_liver"
        self.full_polygon = []
        self.occlusion_polygons = []
        self.current_polygon = []
        self.cursor = None
        self.message = "Draw full liver boundary; Enter closes it."
        self.saved = False

    def on_mouse(self, event, x, y, _flags, _param):
        if x < 0 or y < 0 or x >= self.shape[1] or y >= self.shape[0]:
            return
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            target = self.full_polygon if self.stage == "full_liver" else self.current_polygon
            target.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            target = self.full_polygon if self.stage == "full_liver" else self.current_polygon
            if target:
                target.pop()

    def close_current(self):
        if self.stage == "full_liver":
            if len(self.full_polygon) < 3:
                self.message = "Full liver polygon needs at least 3 points."
                return
            self.stage = "occlusions"
            self.message = "Draw occlusion polygons; Enter closes one; S saves."
            return
        if len(self.current_polygon) < 3:
            self.message = "Occlusion polygon needs at least 3 points."
            return
        self.occlusion_polygons.append(list(self.current_polygon))
        self.current_polygon.clear()
        self.message = f"Closed occlusion #{len(self.occlusion_polygons)}. Draw another or press S."

    def reset_current(self):
        if self.stage == "full_liver":
            self.full_polygon.clear()
        else:
            self.current_polygon.clear()
        self.message = "Current polygon reset."

    def undo_polygon(self):
        if self.stage == "occlusions" and self.occlusion_polygons:
            self.occlusion_polygons.pop()
            self.message = "Removed last closed occlusion polygon."

    def build_masks(self):
        full = polygon_mask(self.shape, self.full_polygon)
        occlusion = polygons_mask(self.shape, self.occlusion_polygons)
        if len(self.current_polygon) >= 3:
            occlusion = cv2.bitwise_or(occlusion, polygon_mask(self.shape, self.current_polygon))
        depth_valid = cv2.bitwise_and(full, cv2.bitwise_not(occlusion))
        return full, depth_valid

    def render(self):
        image_canvas = self.image.copy()
        full = polygon_mask(self.shape, self.full_polygon)
        closed_occ = polygons_mask(self.shape, self.occlusion_polygons)
        current_occ = polygon_mask(self.shape, self.current_polygon)

        tint = image_canvas.copy()
        tint[full > 0] = (60, 150, 60)
        tint[(closed_occ > 0) | (current_occ > 0)] = (40, 40, 220)
        image_canvas = cv2.addWeighted(image_canvas, 0.72, tint, 0.28, 0.0)

        self._draw_polyline(image_canvas, self.full_polygon, (0, 255, 0), self.stage != "full_liver")
        for polygon in self.occlusion_polygons:
            self._draw_polyline(image_canvas, polygon, (0, 0, 255), True)
        self._draw_polyline(image_canvas, self.current_polygon, (0, 0, 255), False)

        panel = np.full((self.shape[0], INFO_PANEL_WIDTH, 3), 28, dtype=np.uint8)
        lines = [
            f"Stage: {self.stage}",
            "",
            "Left click: add point",
            "Right click: undo point",
            "Enter: close polygon",
            "R: reset current",
            "U: remove last occlusion",
            "B: back to full liver",
            "S: validate and save",
            "Esc/Q: cancel",
            "",
        ]
        lines.extend(textwrap.wrap(self.message, width=42) or [""])
        y = 20
        for line in lines:
            cv2.putText(
                panel,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += 18
        return np.hstack([image_canvas, panel])

    @staticmethod
    def _draw_polyline(canvas, points, color, closed):
        if not points:
            return
        pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], closed, color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, 2, color, -1, cv2.LINE_AA)


def load_existing_annotation(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="交互式标注一帧图像的 full_liver_mask 和器械遮挡区域。"
    )
    parser.add_argument("--force", action="store_true", help="允许覆盖已有标注")
    args = parser.parse_args()

    case_dir = case_config.case_dir()
    image_path = os.path.join(case_dir, "image", "lap_undist.png")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Run 01_prepare_case.py first. Missing: {image_path}")

    paths = mask_paths(case_dir)
    existing = load_existing_annotation(paths["annotation"])
    if existing and not args.force:
        raise FileExistsError(
            f"Annotation already exists: {paths['annotation']}. "
            "Use --force only after reviewing the existing overlay."
        )

    annotator = MaskAnnotator(image)
    if existing:
        annotator.full_polygon = [tuple(x) for x in existing.get("full_liver_polygon_xy", [])]
        annotator.occlusion_polygons = [
            [tuple(x) for x in polygon] for polygon in existing.get("occlusion_polygons_xy", [])
        ]
        annotator.stage = "occlusions"
        annotator.message = "Loaded existing annotation. Review and press S to save."

    print("=== Interactive intra-operative mask annotation ===")
    print(f"Frame: {case_config.PATIENT_ID}/{case_config.FRAME_ID}")
    print("Draw the COMPLETE liver boundary. Do not derive it from the open silhouette.")
    print("Then draw instrument occlusions, which are removed only from depth_valid_mask.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, annotator.on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW_NAME, annotator.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13):
                annotator.close_current()
            elif key in (ord("r"), ord("R")):
                annotator.reset_current()
            elif key in (ord("u"), ord("U")):
                annotator.undo_polygon()
            elif key in (ord("b"), ord("B")):
                annotator.stage = "full_liver"
                annotator.current_polygon.clear()
                annotator.message = "Back to full liver boundary."
            elif key in (ord("s"), ord("S")):
                if annotator.stage != "occlusions":
                    annotator.message = "Close the full liver polygon with Enter before saving."
                    continue
                if len(annotator.current_polygon) >= 3:
                    annotator.close_current()
                full, depth_valid = annotator.build_masks()
                annotation = {
                    "patient_id": case_config.PATIENT_ID,
                    "frame_id": case_config.FRAME_ID,
                    "image_path": image_path,
                    "full_liver_polygon_xy": [list(x) for x in annotator.full_polygon],
                    "occlusion_polygons_xy": [
                        [list(x) for x in polygon] for polygon in annotator.occlusion_polygons
                    ],
                }
                try:
                    saved_paths, quality = save_mask_bundle(case_dir, image, full, depth_valid, annotation)
                except MaskValidationError as exc:
                    annotator.message = str(exc)
                    print(f"[INVALID] {exc}")
                    continue
                print(f"[OK] {quality_summary(quality)}")
                for name, path in saved_paths.items():
                    if name != "root":
                        print(f"[OK] {name}: {path}")
                annotator.saved = True
                break
            elif key in (27, ord("q"), ord("Q")):
                break
    finally:
        cv2.destroyAllWindows()

    if not annotator.saved:
        raise RuntimeError("Annotation cancelled; no masks were written.")


if __name__ == "__main__":
    main()
