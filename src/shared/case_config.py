import os

# Patient / frame defaults for scripts 01-07.
PATIENT_ID = os.environ.get("PATIENT_ID", "Patient1")
FRAME_ID = os.environ.get("FRAME_ID", "02")

PROJECT_ROOT = os.environ.get(
    "AR_PROJECT_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")),
)

CASE_ROOT = os.environ.get("CASE_ROOT", os.path.join(PROJECT_ROOT, "data_case"))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(PROJECT_ROOT, "src_data"))
RESULT_ROOT = os.environ.get("RESULT_ROOT", os.path.join(PROJECT_ROOT, "result"))
WEIGHTS_ROOT = os.environ.get("WEIGHTS_ROOT", os.path.join(PROJECT_ROOT, "weights"))
ANNOTATOR = "koo"

WORK_WIDTH = 480
WORK_HEIGHT = 270
IMG_SIZE = (WORK_WIDTH, WORK_HEIGHT)
CONTOUR_THICKNESS = int(os.environ.get("AR_CONTOUR_THICKNESS", "2"))
SEED = int(os.environ.get("AR_SEED", "42"))

# Depth Anything V2 source package and checkpoint.
DA2_PROJECT_DIR = os.environ.get("DA2_PROJECT_DIR", os.path.join(PROJECT_ROOT, "src"))
DA2_ENCODER = os.environ.get("DA2_ENCODER", "vitl")
DA2_INPUT_SIZE = int(os.environ.get("DA2_INPUT_SIZE", "518"))
DA2_CHECKPOINT_PATH = os.environ.get(
    "DA2_CHECKPOINT_PATH",
    os.path.join(WEIGHTS_ROOT, f"depth_anything_v2_{DA2_ENCODER}.pth"),
)
REFINENET_PRETRAINED_PATH = os.environ.get(
    "AR_PRETRAINED_WEIGHT_PATH",
    os.path.join(WEIGHTS_ROOT, "refinenet_patient1_02_best.pth"),
)


def case_dir(case_root=None):
    return os.path.join(case_root or CASE_ROOT, PATIENT_ID, FRAME_ID)


def depth_dir(case_root=None):
    return os.path.join(case_dir(case_root), "depth")


def image_dir(case_root=None):
    return os.path.join(case_dir(case_root), "image")


def contour_dir(case_root=None):
    return os.path.join(case_dir(case_root), "contours")


def mask_dir(case_root=None):
    return os.path.join(case_dir(case_root), "masks")


def result_dir(result_root=None):
    return os.path.join(result_root or RESULT_ROOT, PATIENT_ID, FRAME_ID)


def logs_dir(result_root=None):
    return os.path.join(result_dir(result_root), "logs")


def checkpoints_dir(result_root=None):
    return os.path.join(result_dir(result_root), "checkpoints")


def refinenet_weight_path(result_root=None):
    explicit = os.environ.get("AR_WEIGHT_PATH")
    if explicit:
        return explicit
    trained = os.path.join(checkpoints_dir(result_root), "best.pth")
    return trained if os.path.exists(trained) else REFINENET_PRETRAINED_PATH


def training_dir(result_root=None):
    return os.path.join(result_dir(result_root), "training")


def inference_dir(result_root=None):
    return os.path.join(result_dir(result_root), "inference")


def evaluation_dir(result_root=None):
    return os.path.join(result_dir(result_root), "evaluation")


def mask_qa_dir(result_root=None):
    return os.path.join(result_dir(result_root), "mask_qa")
