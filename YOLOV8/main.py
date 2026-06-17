import sys
from pathlib import Path
from ultralytics import YOLO

# Thêm thư mục gốc vào sys.path để có thể import utils
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from utils import ID_TO_LABEL, PROJECT_ROOT

# Load an official or custom model
model_path = PROJECT_ROOT / "YOLOV8" / "yolov8m" / "yolov8m_final.pt"
model = YOLO(str(model_path))

if hasattr(model, 'model'):
    model.model.names = ID_TO_LABEL


# Perform tracking with the model
source_path = PROJECT_ROOT / "demo.png"
results = model.track(source=str(source_path), show=True, save=True, project=".", tracker="bytetrack.yaml")