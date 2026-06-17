
## ✨ Cập nhật mới nhất (Production-Ready API)
Dự án đã được tái cấu trúc và nâng cấp để sẵn sàng triển khai thực tế (Production-ready):
- **FastAPI REST API**: Cung cấp các endpoint `/predict/json` và `/predict/image` với khả năng tuỳ chỉnh `conf` threshold.
- **Dockerized**: Đóng gói toàn bộ ứng dụng và môi trường bằng Docker.
- **Quản lý môi trường chuẩn**: Cung cấp `requirements.txt` thay cho các thư viện cài đặt thủ công.
- **Tự động hoá Retrain**: Tích hợp script tải dữ liệu từ Roboflow Universe và tự động retrain YOLOv8 với Data Augmentation.

## 1. Tổng Quan Về đề tài
*   Đặt Vấn Đề: Trong khuôn khổ của sự phát triển nhanh chóng của công nghệ và đô thị hóa, việc đảm bảo an toàn giao thông trở thành một thách thức lớn. Biển báo giao thông đóng vai trò quan trọng trong việc hướng dẫn và bảo vệ người tham gia giao thông. 

*   Tầm Quan Trọng của Đề Tài: Nhận diện biển báo giao thông không chỉ cần thiết cho việc tuân thủ luật lệ giao thông mà còn là một yếu tố cốt lõi trong việc phát triển xe tự hành và các hệ thống hỗ trợ lái xe hiện đại.

*   Mục tiêu của Đồ Án: Mục tiêu của đồ án này là phát triển một hệ thống nhận diện biển báo giao thông chính xác và kịp thời sử dụng công nghệ deep learning, đặc biệt tập trung vào dữ liệu từ môi trường giao thông Việt Nam.

*   Ý Nghĩa Ứng Dụng:Ứng dụng của hệ thống này không chỉ giới hạn trong việc nâng cao an toàn giao thông mà còn mở rộng sang các lĩnh vực như hỗ trợ lái xe tự động và quản lý giao thông thông minh.

Khu vực được lựa chọn để thu thập data: Hầu hết các quận ở thành phô Hồ Chí Minh

**Input và Output bài toán (Cập nhật)**
*   **Input**:
    *   **Ảnh (Image)**: Định dạng JPG/PNG thông qua upload API.
    *   **Video / Webcam**: Xử lý luồng (stream) bằng cách cắt frame và gửi qua API, hoặc chạy trực tiếp bằng script local.
*   **Output**:
    *   **Ảnh kết quả (Visual)**: Trả về ảnh đã vẽ sẵn Bounding box và tên biển báo.
    *   **Dữ liệu dạng JSON (Data)**: Trả về toạ độ Bounding box (x, y, w, h), độ tin cậy (Confidence score) và Tên nhóm biển báo (Prohibitory, Warning...).


## 2. Xây dựng Bộ dữ liệu

Để giải quyết bài toán phát hiện biển báo giao thông trong nhiều điều kiện phức tạp (như chói nắng, bóng râm) và hạn chế hiện tượng Imbalanced data (mất cân bằng dữ liệu), dự án sử dụng bộ dữ liệu chất lượng cao từ **Roboflow Universe** bao gồm hơn 9.600 ảnh.

### 2.1. Cách tiếp cận Siêu lớp (Super-classes)
Bộ dữ liệu được tổ chức lại bằng cách gộp các biển báo thành các Siêu lớp (Super-classes) thay vì từng class chi tiết, bao gồm:
*   Biển cấm (Prohibitory)
*   Biển nguy hiểm (Warning)
*   Biển hiệu lệnh (Mandatory)
*   Và một số nhóm biển báo thông dụng khác.

Cách tiếp cận này giúp mô hình YOLOv8 học được đặc trưng tổng quát của từng nhóm biển báo dễ dàng hơn và giảm thiểu đáng kể tỷ lệ nhận diện nhầm lẫn ở các điều kiện thực tế so với việc chia nhỏ thành quá nhiều class.

### 2.2. Tự động hoá Data Pipeline
Toàn bộ quá trình làm việc với dữ liệu đã được tự động hoá giúp dễ dàng nâng cấp mô hình trong tương lai:
*   **Tải dữ liệu:** Tự động pull dữ liệu ở định dạng YOLOv8 trực tiếp từ nền tảng Roboflow.
*   **Tăng cường dữ liệu (Data Augmentation):** Áp dụng các kỹ thuật như Xoay (Rotation), Đổi tỷ lệ (Scaling), và Thay đổi màu sắc/độ sáng (HSV shift) để làm phong phú dữ liệu huấn luyện.
*   **Huấn luyện:** Bạn có thể chạy toàn bộ quá trình tải dữ liệu và huấn luyện lại mô hình (retrain) chỉ thông qua một lệnh duy nhất với script `retrain.py` có sẵn trong mã nguồn.

## 3. Huấn luyện model
### 3.1. Faster R-CNN
Thuật toán  [Faser R-CNN ]( https://arxiv.org/abs/1506.01497) được phát triển bởi Shaoqing Ren và cộng sự.

Faster RCNN là một mô hình phát hiện đối tượng hai giai đoạn. Đầu tiên, nó tạo ra các đề xuất vùng mà có thể chứa đối tượng, sau đó sử dụng một mạng phân loại để xác định lớp và vị trí chính xác của đối tượng và nó hoạt động theo các bước sau:
![Mô hình Faster R-CNN](https://media.geeksforgeeks.org/wp-content/uploads/20230823154315/Region-Proposal-Network-RPN-2.png)
*   Rút Trích Đặc Trưng: Đầu tiên, một hình ảnh được đưa qua mạng CNN (ví dụ: VGG hoặc ResNet) để rút trích đặc trưng.

*   Region Proposal Network (RPN): Sau đó, một mạng con gọi là Region Proposal Network sử dụng đặc trưng này để xác định các vùng (regions) mà có thể chứa đối tượng. RPN tạo ra các đề xuất về vị trí và kích thước của hộp giới hạn tiềm năng.

*   ROI Pooling: Các hộp đề xuất từ RPN sau đó được đưa qua một quá trình gọi là ROI Pooling, nơi mỗi hộp được chuyển hóa thành một kích thước cố định để có thể xử lý được.

*   Phân Loại và Định Vị: Cuối cùng, các đặc trưng từ ROI Pooling được sử dụng bởi hai lớp đầu ra: một lớp phân loại để xác định lớp của đối tượng trong mỗi hộp, và một lớp regression để tinh chỉnh vị trí của hộp đề xuất. 

### 3.2. YOLO
Thuật Toán  [YOLO ]( https://arxiv.org/abs/1506.02640) được phát triển bởi Joseph Redmon và cộng sự.
YOLO là một mô hình phát hiện đối tượng nhanh và hiệu quả, hoạt động theo các bước sau:
![Mô hình YOLO ](https://www.labellerr.com/blog/content/images/2023/01/yolo-algorithm-1.webp
)
*   Xử Lý Toàn Bộ Hình Ảnh: Đầu tiên, YOLO xem xét toàn bộ hình ảnh một cách tổng thể, không giống như các mô hình phát hiện đối tượng truyền thống.

*   Chia Hình Ảnh thành Lưới: Hình ảnh được chia thành một lưới có kích thước cố định (ví dụ: 13x13).

* Dự Đoán Đối Tượng và Lớp: Mỗi ô trong lưới đưa ra dự đoán về hộp giới hạn và xác suất lớp. Hộp giới hạn bao gồm thông tin về vị trí và kích thước của đối tượng tiềm năng, trong khi xác suất lớp biểu thị khả năng đối tượng thuộc về một lớp cụ thể.

* Lọc và Tinh Chỉnh: Cuối cùng, YOLO áp dụng các kỹ thuật như non-maximum suppression để loại bỏ các hộp giới hạn chồng chéo và giữ lại những hộp với xác suất cao nhất.


### 3.3. So Sánh hai mô hình:

Faster RCNN: 
YOLO: Nhanh hơn nhưng có thể kém chính xác hơn trong một số trường hợp. Phù hợp cho các ứng dụng cần tốc độ xử lý nhanh.

Faster RCNN thực hiện phát hiện đối tượng theo hai giai đoạn: đầu tiên là đề xuất các hộp giới hạn, sau đó là phân loại và tinh chỉnh. Điều này thường đảm bảo độ chính xác cao hơn nhưng tốc độ xử lý chậm hơn.Phức tạp hơn, chính xác hơn nhưng chậm hơn. Phù hợp cho các ứng dụng cần độ chính xác cao.

YOLO thực hiện tất cả trong một giai đoạn, xử lý nhanh hơn nhưng đôi khi kém chính xác hơn so với Faster RCNN, đặc biệt trong việc phát hiện các đối tượng nhỏ hoặc chồng chéo.

### Huấn luyện mô hình

Trong quá trình đào tạo của mình, tôi đã áp dụng Gradient Descent
optimizer và đào tạo tất cả các mô hình trong 100 epochs với bathch_size là 32. Chúng tôi đã duy trì các cài đặt tham số tương tự trong các model với nhau. Bằng cách sử dụng phương pháp này, chúng tôi đảm bảo tính nhất quán và cho phép so sánh trực tiếp.

#### Cách huấn luyện model 
Sau đây là cách tổ chức thư mục để huấn luyện model:

* DATASET 
    *    Train
          *   Images
          *   Labels
    *    Val
          *   Images
          *   Labels

source code train model YOLO: [Here](https://docs.ultralytics.com/modes/train/#usage-examples)

hoặc bạn cũng có thể tải sẳn model tôi đã để ở phía trên.





## 4.1 Đánh giá Model
Để đánh giá một model Object Detection có tốt hay không thì các nhà khoa học thường sử dụng hay thông số chính đó là mAP và FPS. Đầu tiên để hiểu hai thông số này thì ta cần làm rõ các thông số liên quan.
#### 4.1.1 IOU
IOU (Intersection Over Union): IOU là thước đo mức độ chồng chéo giữa hai hình dạng. Trong bối cảnh phát hiện đối tượng trong thị giác máy tính, IOU thường được sử dụng để đánh giá mức độ chồng chéo giữa hai hình chữ nhật: hộp giới hạn được dự đoán và hộp giới hạn thực tế trên mặt đất.
Công thức:

#### 4.1.3. Presicion 
Presicion: đo lường mức độ mô hình có thể xác định nhãn hoặc lớp chính xác cho dữ liệu. độ chính xác được tính bằng cách chia số lượng kết quả dương tính thật (𝑇𝑃) cho tổng số lượng kết quả dương tính thật và dương tính giả (𝐹𝑃) 


#### 4.1.3. Recall
Recall: Đo lường mức độ một mô hình có thể xác định chính xác các điểm dữ liệu có liên quan hoặc lớp tích cực trong số tất cả các điểm dữ liệu thuộc lớp đó. Nó được tính bằng cách chia số 𝑇𝑃 cho tổng số dương tính thật (TP) và âm tính giả (𝐹𝑁)


#### 4.1.4. Precision-Recall Curve (PRC):
PRC là biểu diễn đồ họa minh họa mối quan hệ giữa Presicion và Recall ở các ngưỡng khác nhau của mô hình phát hiện đối tượng. Bằng cách điều chỉnh ngưỡng, PRC cho phép quan sát các mức độ Presicion và Recall khác nhau, dẫn đến một đường cong mô tả sự cân bằng giữa hai số liệu này.
#### 4.1.5 Average Precision (AP) và mean Average Precision (mAP)

Trong các bài toán Object Detection, số liệu AP được tính bằng diện tích dưới đường cong PRC. Giá trị này cho biết mức hiệu suất của mô hình trên một lớp tính năng cụ thể. AP cung cấp một giá trị duy nhất để đánh giá sự cân bằng giữa độ chính xác và khả năng thu hồi, đồng thời phản ánh mức độ chính xác của mô hình trong việc định vị đối tượng. Sau khi tính AP, chỉ số Mean Average Precision (mAP) được sử dụng để tính giá trị trung bình của AP.

#### 4.1.6 FPS

FPS cho biết số lượng hình ảnh mà mô hình có thể xử lý trong một giây. Chỉ số này rất quan trọng khi đánh giá khả năng xử lý thời gian thực của mô hình. FPS cao cho thấy model có khả năng xử lý nhanh, phù hợp với các ứng dụng cần phản hồi tức thời.

### 4.2 Đánh giá model

bảng kết quả huấn luyện model được đánh giá trên tập VAL 

Model | mAP50 | mAP50:95 | FPS  
--- | --- | --- | --- 
Faster R-CNN | 0.910 | 0.668 | 39 
YOLOv8n | 0.956 | 0.730 | 84.03
YOLOv8s | 0.981 | 0.790 | 74.07
YOLOv8m | 0.984 | 0.839 | 66.67

Bảng trên cung cấp sự so sánh toàn diện về các đối tượng khác nhau
các mô hình phát hiện, bao gồm các biến thể khác nhau của YOLOv8 và Faster R-CNN. ...kiến nó trở thành lựa chọn thiết thực hơn cho các ứng dụng phát hiện đối tượng theo thời gian thực yêu cầu xử lý hiệu quả.

---

## 5. Hướng dẫn Cài đặt & Sử dụng API (MỚI)

Dự án hiện đã hỗ trợ chạy dưới dạng một RESTful API bằng **FastAPI**, giúp dễ dàng tích hợp với các ứng dụng Web/Mobile.

### Cài đặt môi trường
Clone repo và cài đặt các thư viện cần thiết:
```bash
git clone https://github.com/your-username/Traffic-sign-VietNam-recognition.git
cd Traffic-sign-VietNam-recognition
pip install -r requirements.txt
```

### Chạy API Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Sau đó truy cập [http://localhost:8000/docs](http://localhost:8000/docs) để xem giao diện Swagger UI và test trực tiếp các API.
- `POST /predict/image`: Upload ảnh và nhận về ảnh đã vẽ Bounding Box. (Có thể chỉnh sửa tham số `conf` threshold).
- `POST /predict/json`: Upload ảnh và nhận về tọa độ dạng JSON.

### Triển khai bằng Docker
Dự án đã có sẵn `Dockerfile`. Để build và chạy bằng Docker:
```bash
docker build -t traffic-sign-api .
docker run -p 8000:8000 traffic-sign-api
```

## 6. Hướng dẫn Retrain Model bằng Roboflow
Nếu bạn muốn tự huấn luyện lại mô hình với dữ liệu mới từ Roboflow:
1. Mở file `retrain.py`.
2. Lấy Private API Key từ tài khoản Roboflow của bạn và điền vào biến `API_KEY`.
3. Điền `WORKSPACE_NAME` và `PROJECT_NAME` của dataset bạn muốn dùng.
4. Chạy lệnh:
```bash
python retrain.py
```
Script sẽ tự động tải dataset (chuẩn YOLOv8), áp dụng Data Augmentation và tiến hành huấn luyện. Trọng số (weights) tốt nhất sẽ được lưu tại `Traffic_Sign_Retrain/yolov8_superclasses/weights/best.pt`. Cập nhật lại đường dẫn này vào `app.py` để sử dụng!





