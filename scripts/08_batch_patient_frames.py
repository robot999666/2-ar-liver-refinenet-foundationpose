"""Batch preprocessing, inference, and post-hoc Python TRE/IC evaluation.

Every selected frame uses the canonical result/<patient_id>/<frame_id> layout.
Evaluation runs only after script 07 chooses its final pose, so TRE/IC cannot
influence inference selection. Script 01b remains an intentional manual step;
frames without a validated mask bundle stop during mask preflight.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from shared import case_config


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def frame_id(number):
    return f"{int(number):02d}"


def parse_frames(text):
    frames = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            frames.extend(frame_id(value) for value in range(int(start), int(end) + 1))
        else:
            frames.append(frame_id(chunk))
    if not frames:
        raise ValueError("No frames were selected.")
    return list(dict.fromkeys(frames))


def frame_paths(patient_id, frame, case_root=None, result_root=None):
    case_dir = case_config.case_dir(case_root, patient_id, frame)
    result_dir = case_config.result_dir(result_root, patient_id, frame)
    return {
        "case": case_dir,
        "result": result_dir,
        "logs": os.path.join(result_dir, "logs"),
        "inference": os.path.join(result_dir, "inference"),
        "debug": os.path.join(result_dir, "inference", "debug_iterations"),
        "evaluation": os.path.join(result_dir, "evaluation"),
        "mask_qa": os.path.join(result_dir, "mask_qa"),
        "lap_image": os.path.join(case_dir, "image", "lap_undist.png"),
        "initial_pose": os.path.join(case_dir, "T_view.txt"),
        "depth": os.path.join(case_dir, "depth", "da2_intraop.npy"),
        "final_stl": os.path.join(result_dir, "inference", "tumour_pred_cam.stl"),
    }


def required_source_inputs(data_root, patient_id, frame):
    return {
        "lap_image": os.path.join(data_root, "Dataset", patient_id, "Lap Images", f"{frame}.png"),
        "camera": os.path.join(data_root, "Dataset", patient_id, "Lap Images", "CameraParameters"),
        "calibration": os.path.join(
            data_root, "Dataset", patient_id, "LUS Calibration and Pose", f"{frame}.mat"
        ),
        "segmentation": os.path.join(
            data_root, "Dataset", patient_id, "LUS Segmentation", "json", f"{frame}.json"
        ),
        "contours": os.path.join(
            data_root,
            "contours_and_models",
            patient_id,
            "Annotations",
            frame,
            case_config.ANNOTATOR,
            "contours.xml",
        ),
        "liver": os.path.join(data_root, "contours_and_models", patient_id, "Liver.obj"),
        "tumour": os.path.join(data_root, "contours_and_models", patient_id, "Tumour.obj"),
    }


def missing_source_inputs(data_root, patient_id, frame):
    return {
        name: path
        for name, path in required_source_inputs(data_root, patient_id, frame).items()
        if not os.path.exists(path)
    }


def run_step(script_name, env, log_path):
    ensure_dir(os.path.dirname(log_path))
    command = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", script_name)]
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode


def preflight_case_masks(case_dir, image_path=None):
    import cv2

    from shared.intraop_masks import quality_summary, validate_case_masks

    image_path = image_path or os.path.join(case_dir, "image", "lap_undist.png")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read prepared frame image: {image_path}")
    _, _, quality = validate_case_masks(case_dir, expected_shape=image.shape[:2])
    return quality, quality_summary(quality)


def write_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def write_rows(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_selection(debug_dir):
    path = os.path.join(debug_dir, "best_selection.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing inference selection metadata: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_debug(debug_dir, data_root, patient_id, frame, out_csv):
    from shared.evaluation import eval_debug_dir, write_csv

    rows = eval_debug_dir(
        debug_dir,
        data_root,
        patient_id,
        frame,
        sample_count=50,
        icp_iterations=10,
        oncologic_margin=10.0,
    )
    if not rows:
        raise ValueError(f"No iteration STL files found for evaluation: {debug_dir}")
    write_csv(out_csv, rows)
    return rows


def final_row_from_eval(frame, eval_rows, selection):
    selected_iter = int(selection["selected_iter"])
    selected_eval = next(
        (row for row in eval_rows if int(row["iter"]) == selected_iter),
        None,
    )
    if selected_eval is None:
        raise ValueError(f"Selected iteration {selected_iter} is missing from Python evaluation.")
    posthoc_best = min(eval_rows, key=lambda row: float(row["TRE_mm"]))
    return {
        "frame": frame,
        "status": "ok",
        "selected_iter": selected_iter,
        "stop_reason": selection.get("stop_reason", ""),
        "selected_TRE_mm": f"{float(selected_eval['TRE_mm']):.10f}",
        "selected_IC": int(selected_eval["IC"]),
        "posthoc_best_iter": int(posthoc_best["iter"]),
        "posthoc_best_TRE_mm": f"{float(posthoc_best['TRE_mm']):.10f}",
        "posthoc_best_IC": int(posthoc_best["IC"]),
        "selected_is_posthoc_best": int(selected_iter == int(posthoc_best["iter"])),
    }


def write_final_evaluation(path, patient_id, frame, stl_path, row):
    text = (
        "Evaluation: Python TRE/IC implementation\n"
        "Inference selection: last safe accepted pose; evaluation is post-hoc only\n"
        f"Patient: {patient_id}\n"
        f"Frame: {frame}\n"
        f"STL: {stl_path}\n"
        f"TRE_mm: {row['selected_TRE_mm']}\n"
        f"IC: {'Pass' if int(row['selected_IC']) == 1 else 'Failed'}\n"
        f"IC_numeric: {row['selected_IC']}\n"
        f"Selected_iter: {row['selected_iter']}\n"
        f"Stop_reason: {row['stop_reason']}\n"
        f"Posthoc_best_iter: {row['posthoc_best_iter']}\n"
        f"Posthoc_best_TRE_mm: {row['posthoc_best_TRE_mm']}\n"
    )
    write_text(path, text)


def evaluate_frame(paths, data_root, patient_id, frame):
    ensure_dir(paths["evaluation"])
    eval_csv = os.path.join(paths["evaluation"], "debug_iterations.csv")
    eval_rows = evaluate_debug(paths["debug"], data_root, patient_id, frame, eval_csv)
    selection = read_selection(paths["debug"])
    frame_row = final_row_from_eval(frame, eval_rows, selection)

    selection_report = {
        "inference_selection": selection,
        "posthoc_evaluation": frame_row,
        "warning": (
            "TRE/IC and posthoc_best are evaluation outputs only. "
            "They are not read by scripts/07_infer_export_stl.py."
        ),
    }
    with open(os.path.join(paths["evaluation"], "selection.json"), "w", encoding="utf-8") as file:
        json.dump(selection_report, file, indent=2)
    write_final_evaluation(
        os.path.join(paths["evaluation"], "final_eval.txt"),
        patient_id,
        frame,
        paths["final_stl"],
        frame_row,
    )
    write_text(
        os.path.join(paths["logs"], "evaluation.log"),
        (
            f"[OK] selected_iter={frame_row['selected_iter']} "
            f"TRE_mm={frame_row['selected_TRE_mm']} IC={frame_row['selected_IC']}\n"
        ),
    )
    return frame_row, eval_rows


def make_environment(patient_id, frame, weight):
    env = os.environ.copy()
    env["PATIENT_ID"] = patient_id
    env["FRAME_ID"] = frame
    env["AR_WEIGHT_PATH"] = weight
    return env


def process_frame(args, frame):
    paths = frame_paths(args.patient_id, frame)
    for name in ("logs", "evaluation", "mask_qa"):
        ensure_dir(paths[name])

    missing = missing_source_inputs(case_config.DATA_ROOT, args.patient_id, frame)
    if missing:
        return {
            "frame": frame,
            "status": "skipped_missing_source",
            "missing": "; ".join(f"{name}={path}" for name, path in missing.items()),
        }, []

    env = make_environment(args.patient_id, frame, args.weight)
    if not args.skip_existing_case or not os.path.exists(paths["lap_image"]):
        return_code = run_step(
            "01_prepare_case.py",
            env,
            os.path.join(paths["logs"], "01_prepare_case.log"),
        )
        if return_code:
            return {"frame": frame, "status": "failed_01", "returncode": return_code}, []

    if not os.path.exists(paths["initial_pose"]):
        return_code = run_step(
            "02_setup_initial_pose.py",
            env,
            os.path.join(paths["logs"], "02_setup_initial_pose.log"),
        )
        if return_code:
            return {"frame": frame, "status": "failed_02", "returncode": return_code}, []

    try:
        quality, summary = preflight_case_masks(paths["case"], paths["lap_image"])
        write_text(os.path.join(paths["logs"], "mask_preflight.log"), f"[OK] {summary}\n")
        with open(os.path.join(paths["mask_qa"], "validated_quality.json"), "w", encoding="utf-8") as file:
            json.dump(quality, file, indent=2)
    except Exception as exc:
        write_text(os.path.join(paths["logs"], "mask_preflight.log"), f"[FAIL] {exc}\n")
        return {
            "frame": frame,
            "status": "needs_mask_annotation",
            "missing": str(exc),
        }, []

    if not args.skip_existing_depth or not os.path.exists(paths["depth"]):
        return_code = run_step(
            "05_depth_anything_intraop.py",
            env,
            os.path.join(paths["logs"], "05_depth_anything_intraop.log"),
        )
        if return_code:
            return {"frame": frame, "status": "failed_05", "returncode": return_code}, []

    if not args.skip_existing_infer or not os.path.exists(paths["final_stl"]):
        return_code = run_step(
            "07_infer_export_stl.py",
            env,
            os.path.join(paths["logs"], "07_infer_export_stl_driver.log"),
        )
        if return_code:
            return {"frame": frame, "status": "failed_07", "returncode": return_code}, []

    try:
        return evaluate_frame(paths, case_config.DATA_ROOT, args.patient_id, frame)
    except Exception as exc:
        write_text(os.path.join(paths["logs"], "evaluation.log"), f"[FAIL] {exc}\n")
        return {"frame": frame, "status": "failed_evaluation", "missing": str(exc)}, []


def main():
    parser = argparse.ArgumentParser(
        description="Batch run scripts 01/02/05/07 and Python TRE/IC evaluation."
    )
    parser.add_argument("--patient-id", default="Patient1")
    parser.add_argument("--frames", default="02-10")
    parser.add_argument("--weight", default=case_config.refinenet_weight_path())
    parser.add_argument("--skip-existing-depth", action="store_true")
    parser.add_argument("--skip-existing-case", action="store_true")
    parser.add_argument("--skip-existing-infer", action="store_true")
    args = parser.parse_args()

    frames = parse_frames(args.frames)
    if not os.path.exists(args.weight):
        raise FileNotFoundError(f"RefineNet weight does not exist: {args.weight}")

    batch_dir = os.path.join(
        case_config.RESULT_ROOT,
        args.patient_id,
        f"batch_{frames[0]}_{frames[-1]}",
    )
    ensure_dir(batch_dir)
    summary_rows = []
    all_debug_rows = []

    for frame in frames:
        print(f"=== Batch frame {args.patient_id}/{frame} ===")
        row, eval_rows = process_frame(args, frame)
        summary_rows.append(row)
        for eval_row in eval_rows:
            all_debug_rows.append(
                {
                    "frame": frame,
                    "iter": eval_row["iter"],
                    "TRE_mm": f"{float(eval_row['TRE_mm']):.10f}",
                    "IC": eval_row["IC"],
                    "file": eval_row["file"],
                    "script_selected": int(
                        row.get("status") == "ok"
                        and int(eval_row["iter"]) == int(row["selected_iter"])
                    ),
                }
            )
        print(f"[{row['status']}] {frame}")

    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    tre_values = [float(row["selected_TRE_mm"]) for row in ok_rows]
    ic_values = [int(row["selected_IC"]) for row in ok_rows]
    average = {
        "num_requested_frames": len(frames),
        "num_ok_frames": len(ok_rows),
        "mean_selected_TRE_mm": float(sum(tre_values) / len(tre_values)) if tre_values else None,
        "ic_pass_count": int(sum(ic_values)),
        "ic_pass_rate": float(sum(ic_values) / len(ic_values)) if ic_values else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "The average uses only script-selected final outputs. "
            "Post-hoc best iterations are reported but never used for inference selection."
        ),
    }
    summary_fields = [
        "frame",
        "status",
        "selected_iter",
        "stop_reason",
        "selected_TRE_mm",
        "selected_IC",
        "posthoc_best_iter",
        "posthoc_best_TRE_mm",
        "posthoc_best_IC",
        "selected_is_posthoc_best",
        "returncode",
        "missing",
    ]
    write_rows(os.path.join(batch_dir, "summary.csv"), summary_rows, summary_fields)
    write_rows(
        os.path.join(batch_dir, "debug_all_frames.csv"),
        all_debug_rows,
        ["frame", "iter", "TRE_mm", "IC", "file", "script_selected"],
    )
    write_rows(
        os.path.join(batch_dir, "incomplete_frames.csv"),
        [row for row in summary_rows if row.get("status") != "ok"],
        ["frame", "status", "returncode", "missing"],
    )
    with open(os.path.join(batch_dir, "average.json"), "w", encoding="utf-8") as file:
        json.dump(average, file, indent=2)
    print(f"[OK] batch results: {batch_dir}")


if __name__ == "__main__":
    main()
