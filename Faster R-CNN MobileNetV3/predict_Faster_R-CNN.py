import sys
from pathlib import Path
from PIL import Image, ImageDraw
import torchvision.transforms as T
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import ImageFont

# Thêm thư mục gốc vào sys.path để có thể import utils
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from utils import ID_TO_LABEL, PROJECT_ROOT


def get_faster_rcnn_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# Thiết lập mô hình
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_faster_rcnn_model(num_classes=50)
model.load_state_dict(torch.load(str(PROJECT_ROOT / "Faster R-CNN MobileNetV3" / "faster_epoch_final.pth"), map_location=device))
model.to(device)


# Hàm dự đoán và lưu hình ảnh với bounding boxes và nhãn
def predict_and_save(image_path, model, device, transform, output_path, id_to_label, threshold=0.5):
    original_image = Image.open(image_path)
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    pred_labels = prediction[0]['labels'].cpu().numpy()
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    high_confidence_predictions = pred_scores >= threshold
    final_boxes = pred_boxes[high_confidence_predictions]
    final_labels = pred_labels[high_confidence_predictions]
    final_scores = pred_scores[high_confidence_predictions]

    draw = ImageDraw.Draw(original_image)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(final_boxes, final_labels, final_scores):
        label_name = id_to_label.get(label, 'Unknown')
        text = f"{label_name}, Score: {score:.2f}"

        # Tính kích thước của văn bản
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=5)
        draw.rectangle([(box[0], box[1] - text_height - 4), (box[0] + text_width + 4, box[1])], fill="white")
        draw.text((box[0] + 2, box[1] - text_height - 2), text, fill="black", font=font)

    original_image.save(output_path)


# Các thông số và đối tượng cần thiết
transform = transforms.Compose([transforms.ToTensor()])
image_path = str(PROJECT_ROOT / "demo.png")  # Thay thế bằng đường dẫn của hình ảnh muốn dự đoán
output_path = str(PROJECT_ROOT / "demo_output.png")

# Gọi hàm dự đoán và lưu hình ảnh
predict_and_save(image_path, model, device, transform, output_path, ID_TO_LABEL, threshold=0.5)

