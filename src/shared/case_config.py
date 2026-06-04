"""共享配置和项目标准路径。

所有核心脚本都会导入本模块。病例/帧选择、根目录、图像分辨率、随机种子和
模型权重解析应统一保留在这里，确保 scripts 01-08 使用相同的文件系统契约。

环境变量始终覆盖文档中的默认值。路径可以是绝对路径，也可以相对于进程工作
目录；云端任务推荐使用绝对路径。
"""

import os


def _project_default():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# 病例和帧选择
# ---------------------------------------------------------------------------
# 源数据目录和生成目录使用英文标识：
#   src_data/.../<patient_id>/...
#   data_case/<patient_id>/<frame_id>/...
#   result/<patient_id>/<frame_id>/...
PATIENT_ID = os.environ.get("PATIENT_ID", "Patient1")
FRAME_ID = os.environ.get("FRAME_ID", "02")


# ---------------------------------------------------------------------------
# 标准根目录
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("AR_PROJECT_ROOT", _project_default())
CASE_ROOT = os.environ.get("CASE_ROOT", os.path.join(PROJECT_ROOT, "data_case"))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(PROJECT_ROOT, "src_data"))
RESULT_ROOT = os.environ.get("RESULT_ROOT", os.path.join(PROJECT_ROOT, "result"))
WEIGHTS_ROOT = os.environ.get("WEIGHTS_ROOT", os.path.join(PROJECT_ROOT, "weights"))

# 发布源数据中轮廓标注者子目录的名称。
ANNOTATOR = os.environ.get("AR_ANNOTATOR", "koo")


# ---------------------------------------------------------------------------
# 共享图像和模型输入契约
# ---------------------------------------------------------------------------
# Scripts 01-07 必须使用完全相同的工作分辨率。修改后，之前生成的轮廓、
# 掩码、深度数组、样本和 checkpoint 均不再兼容。
WORK_WIDTH = int(os.environ.get("AR_WORK_WIDTH", "480"))
WORK_HEIGHT = int(os.environ.get("AR_WORK_HEIGHT", "270"))
IMG_SIZE = (WORK_WIDTH, WORK_HEIGHT)

# GT 和渲染轮廓的共同线宽。预处理、样本渲染、训练和推理必须保持一致，
# 避免输入 domain gap。
CONTOUR_THICKNESS = int(os.environ.get("AR_CONTOUR_THICKNESS", "2"))

# 用于样本生成、数据划分、训练、worker 和 Open3D 采样。GPU 的确定性行为
# 仍可能受到 CUDA 和依赖库版本影响。
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
# RefineNet checkpoint 回退路径
# ---------------------------------------------------------------------------
# AR_WEIGHT_PATH 优先级最高。未设置时，推理优先使用当前帧新训练的
# checkpoint；若不存在，再回退到此固定模型。
REFINENET_PRETRAINED_PATH = os.environ.get(
    "AR_PRETRAINED_WEIGHT_PATH",
    os.path.join(WEIGHTS_ROOT, "refinenet_patient1_02_best.pth"),
)


def case_dir(case_root=None, patient_id=None, frame_id=None):
    """返回 data_case/<patient_id>/<frame_id>。"""
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
    """返回 result/<patient_id>/<frame_id>。"""
    return os.path.join(
        result_root or RESULT_ROOT,
        patient_id or PATIENT_ID,
        frame_id or FRAME_ID,
    )


def logs_dir(result_root=None, patient_id=None, frame_id=None):
    """返回训练、推理和评估日志的标准目录。"""
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
    """按照文档约定的优先级解析推理 checkpoint。"""
    explicit = os.environ.get("AR_WEIGHT_PATH")
    if explicit:
        return explicit
    trained = os.path.join(
        checkpoints_dir(result_root, patient_id, frame_id),
        "best.pth",
    )
    return trained if os.path.exists(trained) else REFINENET_PRETRAINED_PATH
