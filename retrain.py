from roboflow import Roboflow
from ultralytics import YOLO
import os

def main():
    # ==========================================
    # BƯỚC 1: TẢI DATASET TỪ ROBOFLOW
    # ==========================================
    # TODO: Thay thế 'YOUR_API_KEY' bằng API Key của bạn từ Roboflow
    # Hướng dẫn lấy key: Đăng nhập Roboflow -> Settings -> API -> Copy Private API Key
    API_KEY = "nfzKcHjhVMXhCzfxsstI"
    
    # Thông tin project trên Roboflow (Ví dụ project gộp nhóm biển báo)
    # Bạn có thể tìm các project dạng "Zalo AI Traffic Sign" trên Roboflow Universe
    WORKSPACE_NAME = "vh-kien" 
    PROJECT_NAME = "vietnam-traffic-sign-altsi-qytwg"
    VERSION = 1

    print("Đang kết nối tải dataset từ Roboflow...")
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
        version = project.version(VERSION)
        
        # Tải dataset định dạng chuẩn YOLOv8
        dataset = version.download("yolov8")
        print(f"Dataset đã tải xong tại: {dataset.location}")
    except Exception as e:
        print(f"Lỗi khi tải dataset: {e}")
        print("Vui lòng kiểm tra lại API_KEY, WORKSPACE_NAME và PROJECT_NAME.")
        return

    # ==========================================
    # BƯỚC 2: TIẾN HÀNH TRAINING YOLOV8
    # ==========================================
    print("\nBắt đầu quá trình training model YOLOv8...")
    
    # Khởi tạo model từ pre-trained weights cỡ trung (m)
    model = YOLO("yolov8m.pt") 
    
    # Lấy đường dẫn file data.yaml trong dataset vừa tải
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        print(f"Không tìm thấy file {data_yaml_path}. Quá trình train bị hủy.")
        return

    # Cấu hình tham số training
    results = model.train(
        data=data_yaml_path,
        epochs=50,                  # Số vòng lặp huấn luyện (có thể tăng lên 100 nếu cần)
        imgsz=640,                  # Kích thước ảnh đầu vào
        batch=16,                   # Số lượng ảnh trong một mẻ (tuỳ thuộc vào VRAM GPU)
        device="cuda",              # Sử dụng GPU (sửa thành 'cpu' nếu máy không có card rời)
        project="Traffic_Sign_Retrain", # Thư mục lưu kết quả
        name="yolov8_superclasses", # Tên folder run
        patience=10,                # Dừng sớm nếu model không cải thiện sau 10 epoch
        save=True,                  # Lưu lại weight tốt nhất
        # Data Augmentation cơ bản
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, # Biến đổi màu sắc, độ sáng
        degrees=10.0,               # Xoay ảnh ngẫu nhiên 10 độ
        translate=0.1,              # Dịch chuyển ảnh
        scale=0.5,                  # Phóng to/thu nhỏ
        flipud=0.0, fliplr=0.5      # Lật ảnh ngang
    )

    print("\nQuá trình training đã hoàn tất!")
    print(f"File trọng số (weights) tốt nhất được lưu tại: Traffic_Sign_Retrain/yolov8_superclasses/weights/best.pt")
    print("Vui lòng cập nhật đường dẫn này vào file app.py để API sử dụng model mới.")

if __name__ == "__main__":
    main()
