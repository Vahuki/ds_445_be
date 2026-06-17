import sys
from pathlib import Path
import cv2
import torch
import torchvision.transforms as T
import time
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
model.eval()
def get_faster_rcnn_model(num_classes):
    # Load a model pre-trained on COCO
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
def load_model(model_path, num_classes):
    model = get_faster_rcnn_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model, device

def predict_on_frame(model, device, frame):
    # Transform the frame
    transform = T.Compose([T.ToTensor()])
    frame = transform(frame).to(device)
    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model([frame])
    return predictions[0]
def process_and_save_video(model, device, input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error opening input video file")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    total_frames = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict on the frame
        start_time = time.time()
        predictions = predict_on_frame(model, device, rgb_frame)
        total_time += time.time() - start_time

        # Process predictions (e.g., draw bounding boxes)
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if score < 0.2:
                continue
            x1, y1, x2, y2 = map(int, box)
            label_name = ID_TO_LABEL.get(label.item(), 'Unknown')
            text = f'{label_name}: {score:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Write frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate and return average FPS
    avg_fps = total_frames / total_time

    # Open the saved video and add FPS text
    cap = cv2.VideoCapture(output_video_path)
    if not cap.isOpened():
        print("Error opening saved video file")
        return

    # Re-initialize video writer
    out = cv2.VideoWriter(output_video_path.replace('.mp4', '_fps.mp4'), fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Put FPS text on frame
        cv2.putText(frame, f'Avg FPS: {avg_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame to new output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return avg_fps

# Example usage

# Initialize model
num_classes = 50
model_path = str(PROJECT_ROOT / "Faster R-CNN MobileNetV3" / "faster_epoch_final.pth")
model, device = load_model(model_path, num_classes)

# Process and save video
input_video_path = str(PROJECT_ROOT / "demo_video.mp4")  # Update with your video path
output_video_path = str(PROJECT_ROOT / "video_output.mp4")  # Update with your desired output path
avg_fps = process_and_save_video(model, device, input_video_path, output_video_path)
print(f"FPS trung bình: {avg_fps}")