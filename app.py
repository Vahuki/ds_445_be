from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np
from utils import PROJECT_ROOT
import os
from pathlib import Path
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Traffic Sign Recognition API",
    description="API nhận diện biển báo giao thông tại Việt Nam sử dụng YOLOv8",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
model_path = PROJECT_ROOT / "YOLOV8" / "yolov8m" / "yolov8m_final.pt"

# Disable weights_only globally for PyTorch 2.6+ to load YOLOv8 models safely
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

try:
    import torch
    import ultralytics.utils.loss
    # Alias DFLoss to BboxLoss to fix unpickling issue with older models
    if not hasattr(ultralytics.utils.loss, 'DFLoss'):
        class DummyDFLoss: pass
        ultralytics.utils.loss.DFLoss = DummyDFLoss
        
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Traffic Sign Recognition API.",
        "endpoints": {
            "GET /labels": "Xem danh sách các loại biển báo được hỗ trợ",
            "POST /predict/json": "Dự đoán biển báo và trả về tọa độ (JSON)",
            "POST /predict/image": "Dự đoán biển báo và trả về ảnh đã vẽ bounding box (Image)",
            "POST /report": "Báo cáo nhận diện sai và lưu vào CSDL (Active Learning)"
        }
    }

# Tạo thư mục lưu trữ report nếu chưa có
REPORT_DIR = PROJECT_ROOT / "dataset" / "reports"
(REPORT_DIR / "images").mkdir(parents=True, exist_ok=True)
(REPORT_DIR / "labels").mkdir(parents=True, exist_ok=True)

@app.post("/report")
async def report_incorrect_detection(
    file: UploadFile = File(...),
    label: str = Form(...),
    x_center: float = Form(...),
    y_center: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
):
    """Lưu báo cáo nhận diện sai từ người dùng để phục vụ Active Learning"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_filename = f"report_{timestamp}.jpg"
        txt_filename = f"report_{timestamp}.txt"
        
        img_path = REPORT_DIR / "images" / img_filename
        txt_path = REPORT_DIR / "labels" / txt_filename
        
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save(img_path, format="JPEG")
        
        # Tìm class_id dựa trên tên nhãn
        class_id = 0
        if model is not None:
            for k, v in model.names.items():
                if v == label or v.lower() == label.lower():
                    class_id = k
                    break
                    
        # Lưu nhãn theo định dạng YOLO: class_id x_center y_center width height
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            
        return {"status": "success", "message": "Báo cáo đã được ghi nhận."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/labels")
def get_labels():
    """Trả về danh sách tất cả các nhãn (biển báo) mà mô hình có thể nhận diện."""
    if model is None:
        return {"error": "Model not loaded."}
    return {"total_classes": len(model.names), "labels": model.names}

@app.post("/predict/json")
async def predict_sign_json(file: UploadFile = File(...), conf: float = Form(0.5)):
    """Upload ảnh, nhận diện biển báo và trả về kết quả dưới dạng JSON. Tham số 'conf' để điều chỉnh độ nhạy (mặc định 0.5)."""
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded properly."})
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 1. Run Traffic Sign Model
        results = model(source=image, conf=conf)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id, "Unknown")
                
                detections.append({
                    "bounding_box": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)},
                    "confidence": round(confidence, 4),
                    "class_id": cls_id,
                    "label": label
                })
                
        return JSONResponse(content={"filename": file.filename, "total_detected": len(detections), "conf_threshold": conf, "detections": detections})
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Nhận frame video dưới dạng byte hoặc chuỗi base64
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
                
            if "bytes" in message:
                image_bytes = message["bytes"]
            elif "text" in message:
                import base64
                text_data = message["text"]
                if "," in text_data:
                    text_data = text_data.split(",")[1]
                image_bytes = base64.b64decode(text_data)
            else:
                continue
            
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                conf = 0.6 # Tăng conf để lọc nhiễu, loại bỏ nhận diện sai
                
                # 1. Run Traffic Sign Model
                results = model(source=image, conf=conf)
                
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = model.names.get(cls_id, "Unknown")
                        
                        detections.append({
                            "bounding_box": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)},
                            "confidence": round(confidence, 4),
                            "class_id": cls_id,
                            "label": label
                        })
                
                # Gửi trả cấu trúc tương thích với parseDetections
                await websocket.send_json({"detections": detections, "error": None})
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                await websocket.send_json({"error": "Failed to process frame"})
                
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")

@app.post("/predict/image")
async def predict_sign_image(file: UploadFile = File(...), conf: float = Form(0.1)):
    """Upload ảnh, nhận diện biển báo và trả về trực tiếp ảnh đã được vẽ bounding box. Tham số 'conf' để chỉnh độ nhạy (mặc định 0.1)."""
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded properly."})
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # YOLOv8 trả về class Results có phương thức plot()
        results = model.predict(source=image, save=False, conf=conf)
        
        # Lấy mảng numpy (BGR) từ frame đã vẽ bbox
        res_plotted = results[0].plot()
        
        # Encode lại thành chuẩn JPEG
        _, encoded_img = cv2.imencode('.jpg', res_plotted)
        
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
