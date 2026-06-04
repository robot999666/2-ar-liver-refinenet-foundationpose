"""Shared configuration and canonical project paths.

All core scripts import this module. Keep patient/frame selection, root
directories, image resolution, random seed, and model-weight resolution here
so that scripts 01-08 agree on the same filesystem contract.

Environment variables always override the documented defaults. Paths may be
absolute or relative to the process working directory, although absolute paths
are recommended for cloud jobs.
"""

import os


def _project_default():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------
# Source and generated directories use English identifiers:
#   src_data/.../<patient_id>/...
#   data_case/<patient_id>/<frame_id>/...
#   result/<patient_id>/<frame_id>/...
PATIENT_ID = os.environ.get("PATIENT_ID", "Patient1")
FRAME_ID = os.environ.get("FRAME_ID", "02")


# ---------------------------------------------------------------------------
# Canonical roots
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("AR_PROJECT_ROOT", _project_default())
CASE_ROOT = os.environ.get("CASE_ROOT", os.path.join(PROJECT_ROOT, "data_case"))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(PROJECT_ROOT, "src_data"))
RESULT_ROOT = os.environ.get("RESULT_ROOT", os.path.join(PROJECT_ROOT, "result"))
WEIGHTS_ROOT = os.environ.get("WEIGHTS_ROOT", os.path.join(PROJECT_ROOT, "weights"))

# Name of the contour annotator subdirectory in the published source data.
ANNOTATOR = os.environ.get("AR_ANNOTATOR", "koo")


# ---------------------------------------------------------------------------
# Shared image/model-input contract
# ---------------------------------------------------------------------------
# Scripts 01-07 must use this exact working resolution. Changing it invalidates
# previously generated contours, masks, depth arrays, samples, and checkpoints.
WORK_WIDTH = int(os.environ.get("AR_WORK_WIDTH", "480"))
WORK_HEIGHT = int(os.environ.get("AR_WORK_HEIGHT", "270"))
IMG_SIZE = (WORK_WIDTH, WORK_HEIGHT)

# GT and rendered contour line width. Keep this identical across preprocessing,
# sample rendering, training, and inference to avoid an input-domain mismatch.
CONTOUR_THICKNESS = int(os.environ.get("AR_CONTOUR_THICKNESS", "2"))

# Used by sample generation, split assignment, training, workers, and Open3D
# sampling. Deterministic GPU behavior can still depend on CUDA/library builds.
SEED = int(os.environ.get("AR_SEED", "42"))


# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------
DA2_PROJECT_DIR = os.environ.get("DA2_PROJECT_DIR", os.path.join(PROJECT_ROOT, "src"))
DA2_ENCODER = os.environ.get("DA2_ENCODER", "vitl")
DA2_INPUT_SIZE = int(os.environ.get("DA2_INPUT_SIZE", "518"))
DA2_CHECKPOINT_PATH = os.environ.get(
    "DA2_CHECKPOINT_PATH",
    os.path.join(WEIGHTS_ROOT, f"depth_anything_v2_{DA2_ENCODER}.pth"),
)


# ---------------------------------------------------------------------------
# RefineNet checkpoint fallback
# ---------------------------------------------------------------------------
# AR_WEIGHT_PATH has highest priority. Otherwise inference uses a newly trained
# per-frame checkpoint when available, then falls back to this curated model.
REFINENET_PRETRAINED_PATH = os.environ.get(
    "AR_PRETRAINED_WEIGHT_PATH",
    os.path.join(WEIGHTS_ROOT, "refinenet_patient1_02_best.pth"),
)


def case_dir(case_root=None, patient_id=None, frame_id=None):
    """Return data_case/<patient_id>/<frame_id>."""
    return os.path.join(
        case_root or CASE_ROOT,
        patient_id or PATIENT_ID,
        frame_id or FRAME_ID,
    )


def depth_dir(case_root=None, patient_id=None, frame_id=None):
    return os.path.join(case_dir(case_root, patient_id, frame_id), "depth")


def image_dir(case_root=None, patient_id=None, frame_id=None):
    return os.path.join(case_dir(case_root, patient_id, frame_id), "image")


def contour_dir(case_root=None, patient_id=None, frame_id=None):
    return os.path.join(case_dir(case_root, patient_id, frame_id), "contours")


def mask_dir(case_root=None, patient_id=None, frame_id=None):
    return os.path.join(case_dir(case_root, patient_id, frame_id), "masks")


def result_dir(result_root=None, patient_id=None, frame_id=None):
    """Return result/<patient_id>/<frame_id>."""
    return os.path.join(
        result_root or RESULT_ROOT,
        patient_id or PATIENT_ID,
        frame_id or FRAME_ID,
    )


def logs_dir(result_root=None, patient_id=None, frame_id=None):
    """Canonical location for training, inference, and evaluation logs."""
    return os.path.join(result_dir(result_root, patient_id, frame_id), "logs")


def checkpoints_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "checkpoints")


def training_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "training")


def inference_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "inference")


def evaluation_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "evaluation")


def mask_qa_dir(result_root=None, patient_id=None, frame_id=None):
    return os.path.join(result_dir(result_root, patient_id, frame_id), "mask_qa")


def refinenet_weight_path(result_root=None, patient_id=None, frame_id=None):
    """Resolve the inference checkpoint using the documented priority order."""
    explicit = os.environ.get("AR_WEIGHT_PATH")
    if explicit:
        return explicit
    trained = os.path.join(
        checkpoints_dir(result_root, patient_id, frame_id),
        "best.pth",
    )
    return trained if os.path.exists(trained) else REFINENET_PRETRAINED_PATH
