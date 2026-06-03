import argparse
import csv
import json
import os
import shutil
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
            frames.extend(frame_id(i) for i in range(int(start), int(end) + 1))
        else:
            frames.append(frame_id(chunk))
    return frames


def required_inputs(data_root, patient_id, frame):
    return {
        "lap_image": os.path.join(data_root, "Dataset", patient_id, "Lap Images", f"{frame}.png"),
        "camera": os.path.join(data_root, "Dataset", patient_id, "Lap Images", "CameraParameters"),
        "calibration": os.path.join(data_root, "Dataset", patient_id, "LUS Calibration and Pose", f"{frame}.mat"),
        "seg_json": os.path.join(data_root, "Dataset", patient_id, "LUS Segmentation", "json", f"{frame}.json"),
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


def missing_inputs(data_root, patient_id, frame):
    paths = required_inputs(data_root, patient_id, frame)
    return {name: path for name, path in paths.items() if not os.path.exists(path)}


def run_step(script, env, cwd, log_path):
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, "scripts", script)]
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return proc.returncode


def copy_if_exists(src, dst):
    if os.path.exists(src):
        ensure_dir(os.path.dirname(dst))
        shutil.copyfile(src, dst)


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def evaluate_debug(debug_dir, data_root, patient_id, frame, eval_out):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_debug_tre_python",
        os.path.join(PROJECT_ROOT, "scripts", "_eval_debug_tre_python.py"),
    )
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    rows = eval_mod.eval_debug_dir(
        debug_dir,
        data_root,
        patient_id,
        frame,
        sample_count=50,
        icp_iterations=10,
        oncologic_margin=10.0,
    )
    eval_mod.write_csv(eval_out, rows)
    return rows


def read_selection(debug_dir):
    selection_path = os.path.join(debug_dir, "best_selection.json")
    if not os.path.exists(selection_path):
        return {}
    with open(selection_path, "r", encoding="utf-8") as f:
        return json.load(f)


def final_row_from_eval(frame, eval_rows, selection):
    selected_iter = int(selection.get("selected_iter", -1))
    selected_eval = next((row for row in eval_rows if int(row["iter"]) == selected_iter), None)
    best_eval = min(eval_rows, key=lambda row: float(row["TRE_mm"])) if eval_rows else None
    return {
        "frame": frame,
        "status": "ok",
        "selected_iter": selected_iter,
        "stop_reason": selection.get("stop_reason", ""),
        "selected_TRE_mm": f"{float(selected_eval['TRE_mm']):.10f}" if selected_eval else "",
        "selected_IC": selected_eval.get("IC", "") if selected_eval else "",
        "posthoc_best_iter": best_eval.get("iter", "") if best_eval else "",
        "posthoc_best_TRE_mm": f"{float(best_eval['TRE_mm']):.10f}" if best_eval else "",
        "posthoc_best_IC": best_eval.get("IC", "") if best_eval else "",
        "selected_is_posthoc_best": int(selected_eval is not None and best_eval is not None and int(selected_eval["iter"]) == int(best_eval["iter"])),
    }


def write_final_txt(path, patient_id, frame, method, stl_path, row):
    ic_text = "Pass" if str(row.get("selected_IC")) == "1" else "Failed"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Method: {method}\n")
        f.write(f"Patient: {patient_id}\n")
        f.write(f"Frame: {frame}\n")
        f.write(f"STL: {stl_path}\n")
        f.write(f"TRE_mm: {row.get('selected_TRE_mm', '')}\n")
        f.write(f"IC: {ic_text}\n")
        f.write(f"IC_numeric: {row.get('selected_IC', '')}\n")
        f.write(f"Selected_iter: {row.get('selected_iter', '')}\n")
        f.write(f"Stop_reason: {row.get('stop_reason', '')}\n")
        f.write("Inference_selection: script_selected_raw_score_stop\n")
        f.write("Evaluation_source: Python TRE/IC port of MATLAB debug metric\n")


def sync_src_data_eval_outputs(data_root, method, patient_id, frame, debug_eval_path, final_txt_src, summary_row):
    eval_tag = f"{method}_{patient_id}_{frame}"
    frame_eval_dir = os.path.join(data_root, "Evaluation_Results", eval_tag)
    ensure_dir(frame_eval_dir)
    final_txt_dst = os.path.join(frame_eval_dir, f"{eval_tag}.txt")
    debug_csv_dst = os.path.join(frame_eval_dir, f"{eval_tag}_debug_iterations.csv")
    shutil.copyfile(final_txt_src, final_txt_dst)
    shutil.copyfile(debug_eval_path, debug_csv_dst)
    meta_path = os.path.join(frame_eval_dir, f"{eval_tag}_selection.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary_row, f, indent=2)


def write_batch_eval_outputs(data_root, method, patient_id, frame_start, frame_end, summary_csv, debug_csv, average_json):
    eval_tag = f"{method}_{patient_id}_{frame_start}_{frame_end}_batch"
    batch_eval_dir = os.path.join(data_root, "Evaluation_Results", eval_tag)
    ensure_dir(batch_eval_dir)
    shutil.copyfile(summary_csv, os.path.join(batch_eval_dir, f"{eval_tag}_summary.csv"))
    shutil.copyfile(debug_csv, os.path.join(batch_eval_dir, f"{eval_tag}_debug_all_frames.csv"))
    shutil.copyfile(average_json, os.path.join(batch_eval_dir, f"{eval_tag}_average.json"))


def main():
    parser = argparse.ArgumentParser(description="Batch preprocess, infer, and post-hoc evaluate Patient frames.")
    parser.add_argument("--patient-id", default="Patient1")
    parser.add_argument("--frames", default="02-10")
    parser.add_argument("--weight", default=os.path.join(PROJECT_ROOT, "result", "Patient1", "02", "best.pth"))
    parser.add_argument("--method", default="FoundationPose")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--skip-existing-depth", action="store_true")
    parser.add_argument("--skip-existing-case", action="store_true")
    parser.add_argument("--skip-existing-infer", action="store_true")
    args = parser.parse_args()

    frames = parse_frames(args.frames)
    batch_tag = f"batch_{frames[0]}_{frames[-1]}"
    output_root = args.output_root or os.path.join(PROJECT_ROOT, "result", args.patient_id, batch_tag)
    logs_dir = os.path.join(output_root, "logs")
    final_eval_dir = os.path.join(output_root, "final_eval")
    debug_eval_dir = os.path.join(output_root, "debug_eval")
    ensure_dir(logs_dir)
    ensure_dir(final_eval_dir)
    ensure_dir(debug_eval_dir)

    summary_rows = []
    debug_all_rows = []
    skipped_rows = []

    for frame in frames:
        print(f"=== Batch frame {args.patient_id}/{frame} ===")
        missing = missing_inputs(case_config.DATA_ROOT, args.patient_id, frame)
        if missing:
            row = {
                "frame": frame,
                "status": "skipped_missing_inputs",
                "missing": "; ".join(f"{k}={v}" for k, v in missing.items()),
            }
            summary_rows.append(row)
            skipped_rows.append(row)
            print(f"[SKIP] {frame}: missing {', '.join(missing)}")
            continue

        env = os.environ.copy()
        env["PATIENT_ID"] = args.patient_id
        env["FRAME_ID"] = frame
        env["AR_WEIGHT_PATH"] = args.weight
        env["AR_EVAL_METHOD_NAME"] = args.method

        case_dir = os.path.join(case_config.CASE_ROOT, args.patient_id, frame)
        pred_dir = os.path.join(case_dir, "pred")
        debug_dir = os.path.join(pred_dir, "debug_iterations")

        if not args.skip_existing_case or not os.path.exists(os.path.join(case_dir, "image", "lap_undist.png")):
            rc = run_step("01_prepare_case.py", env, PROJECT_ROOT, os.path.join(logs_dir, f"{frame}_01_prepare_case.log"))
            if rc != 0:
                summary_rows.append({"frame": frame, "status": "failed_01", "returncode": rc})
                print(f"[FAIL] {frame}: 01 returned {rc}")
                continue

        if not os.path.exists(os.path.join(case_dir, "T_view.txt")):
            rc = run_step("02_setup_initial_pose.py", env, PROJECT_ROOT, os.path.join(logs_dir, f"{frame}_02_setup_initial_pose.log"))
            if rc != 0:
                summary_rows.append({"frame": frame, "status": "failed_02", "returncode": rc})
                print(f"[FAIL] {frame}: 02 returned {rc}")
                continue

        depth_path = os.path.join(case_dir, "depth", "da2_intraop.npy")
        if not args.skip_existing_depth or not os.path.exists(depth_path):
            rc = run_step("05_depth_anything_intraop.py", env, PROJECT_ROOT, os.path.join(logs_dir, f"{frame}_05_depth_anything_intraop.log"))
            if rc != 0:
                summary_rows.append({"frame": frame, "status": "failed_05", "returncode": rc})
                print(f"[FAIL] {frame}: 05 returned {rc}")
                continue

        if not args.skip_existing_infer or not os.path.exists(os.path.join(pred_dir, "tumour_pred_cam.stl")):
            rc = run_step("07_infer_export_stl.py", env, PROJECT_ROOT, os.path.join(logs_dir, f"{frame}_07_infer_export_stl.log"))
            if rc != 0:
                summary_rows.append({"frame": frame, "status": "failed_07", "returncode": rc})
                print(f"[FAIL] {frame}: 07 returned {rc}")
                continue

        debug_eval_path = os.path.join(debug_eval_dir, f"{frame}_debug_eval.csv")
        eval_rows = evaluate_debug(debug_dir, case_config.DATA_ROOT, args.patient_id, frame, debug_eval_path)
        selection = read_selection(debug_dir)
        frame_row = final_row_from_eval(frame, eval_rows, selection)
        summary_rows.append(frame_row)

        for eval_row in eval_rows:
            debug_all_rows.append(
                {
                    "frame": frame,
                    "iter": eval_row["iter"],
                    "TRE_mm": f"{float(eval_row['TRE_mm']):.10f}",
                    "IC": eval_row.get("IC", ""),
                    "file": eval_row["file"],
                    "script_selected": int(int(eval_row["iter"]) == int(frame_row["selected_iter"])),
                }
            )

        stl_path = os.path.join(case_config.DATA_ROOT, "Registration Methods", args.method, args.patient_id, f"{frame}.stl")
        final_txt = os.path.join(final_eval_dir, f"{frame}_final_eval.txt")
        write_final_txt(final_txt, args.patient_id, frame, args.method, stl_path, frame_row)
        sync_src_data_eval_outputs(case_config.DATA_ROOT, args.method, args.patient_id, frame, debug_eval_path, final_txt, frame_row)

        copy_if_exists(os.path.join(pred_dir, "infer.log"), os.path.join(logs_dir, f"{frame}_infer.log"))
        copy_if_exists(os.path.join(debug_dir, "history.csv"), os.path.join(debug_eval_dir, f"{frame}_inference_history.csv"))
        print(
            f"[OK] {frame}: selected iter_{int(frame_row['selected_iter']):02d}, "
            f"TRE={float(frame_row['selected_TRE_mm']):.4f}, IC={frame_row['selected_IC']}"
        )

    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    if ok_rows:
        tre_values = [float(row["selected_TRE_mm"]) for row in ok_rows]
        ic_values = [int(row["selected_IC"]) for row in ok_rows]
        avg = {
            "num_ok_frames": len(ok_rows),
            "mean_selected_TRE_mm": float(sum(tre_values) / len(tre_values)),
            "ic_pass_count": int(sum(ic_values)),
            "ic_pass_rate": float(sum(ic_values) / len(ic_values)),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "note": "Averages use script-selected final outputs only; posthoc_best is reported separately and is not used for inference selection.",
        }
    else:
        avg = {
            "num_ok_frames": 0,
            "mean_selected_TRE_mm": None,
            "ic_pass_count": 0,
            "ic_pass_rate": None,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
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
    summary_csv = os.path.join(output_root, "summary.csv")
    debug_all_csv = os.path.join(output_root, "debug_all_frames.csv")
    average_json = os.path.join(output_root, "average.json")
    write_csv(summary_csv, summary_rows, summary_fields)
    write_csv(debug_all_csv, debug_all_rows, ["frame", "iter", "TRE_mm", "IC", "file", "script_selected"])
    write_csv(os.path.join(output_root, "skipped_frames.csv"), skipped_rows, ["frame", "status", "missing"])
    with open(average_json, "w", encoding="utf-8") as f:
        json.dump(avg, f, indent=2)
    write_batch_eval_outputs(case_config.DATA_ROOT, args.method, args.patient_id, frames[0], frames[-1], summary_csv, debug_all_csv, average_json)

    print(f"[OK] batch output: {output_root}")
    print(f"[OK] summary: {summary_csv}")
    print(f"[OK] debug all: {debug_all_csv}")
    print(f"[OK] average: {average_json}")


if __name__ == "__main__":
    main()
