# NLPonSchool

Kho lưu trữ bài thực hành Machine Learning và khai phá dữ liệu của **Phạm Đào Thành Đạt — MSSV 21083541**. Nội dung hiện có tập trung vào hồi quy logistic, hồi quy tuyến tính có regularization, so sánh mô hình phân loại và sử dụng Weka trong Java.

## Tóm tắt nội dung

### 1. Notebook Python

| Bài thực hành | Nội dung chính |
| --- | --- |
| [Lab 01 — Sigmoid](C1_W3_Lab01_Sigmoid_function.ipynb) | Cài đặt hàm sigmoid bằng NumPy, tính giá trị đầu ra và vẽ đồ thị. |
| [Lab 02 — Decision Boundary](C1_W3_Lab02_Decision_Boundary.ipynb) | Minh họa ngưỡng phân loại và đường biên quyết định của hồi quy logistic trên dữ liệu hai đặc trưng. |
| [Lab 03 — Cost Function](C1_W3_Lab03_Cost_Function.ipynb) | Cài đặt hàm chi phí của hồi quy logistic bằng vòng lặp và phép toán vector; so sánh chi phí giữa các bộ trọng số. |
| [Lab 04 — Gradient Descent](C1_W3_Lab04_Gradient_Descent.ipynb) | Tính gradient của hồi quy logistic theo hai cách: vòng lặp và vector hóa. |
| [Lab 05 — One Vs All](C1_W3_Lab05_One_Vs_All.ipynb) | Nội dung hiện tại tạo dữ liệu phân loại nhị phân, so sánh Logistic Regression, Decision Tree và Random Forest, rồi thử kết hợp xác suất dự đoán. |
| [Lab 06 — Regularized Cost](C1_W3_Lab06_Regularized_Cost.ipynb) | Tính hàm chi phí hồi quy tuyến tính có regularization L2, không phạt hệ số bias. |
| [Lab 07 — Regularized Gradient Descent](C1_W3_Lab07_Regularized_Gradient_Descent.ipynb) | Huấn luyện hồi quy tuyến tính bằng gradient descent có regularization và lưu lịch sử chi phí. |
| [Lab 08 — Scikit-Learn](C1_W3_Lab08_Scikit_Learn.ipynb) | Huấn luyện Logistic Regression bằng scikit-learn, dự đoán, tính độ chính xác trên tập huấn luyện và vẽ đường biên quyết định. |

### 2. Dự án Java/Weka

| Dự án | Nội dung chính |
| --- | --- |
| [Clusster_DataMining](21083541_PhamDaoThanhDat/Clusster_DataMining) | Đọc dữ liệu ARFF, xây dựng mô hình `SimpleKMeans`, chọn khoảng cách Euclidean hoặc Manhattan và in cụm của từng mẫu. Chương trình mẫu sử dụng 4 cụm và khoảng cách Manhattan. |
| [J48_DataMining](21083541_PhamDaoThanhDat/J48_DataMining) | Đọc dữ liệu ARFF, chia 70% huấn luyện và 30% kiểm tra bằng `RemovePercentage`, xây dựng cây quyết định `J48` với thuộc tính cuối làm nhãn; in cây, kết quả đánh giá, ma trận nhầm lẫn và thống kê từng lớp. |

### 3. Tài liệu

- [Báo cáo phân cụm](21083541_PhamDaoThanhDat/21083541_PhamDaoThanhDat_Cluster_DataMining.docx).
- [Báo cáo J48](21083541_PhamDaoThanhDat/21083541_PhamDaoThanhDat_J48.docx).
- [Bản nén bài tập gốc](21083541_PhamDaoThanhDat.rar).

## Cấu trúc thư mục

```text
NLPonSchool/
├── README.md
├── .gitignore
├── C1_W3_Lab01_Sigmoid_function.ipynb
├── C1_W3_Lab02_Decision_Boundary.ipynb
├── C1_W3_Lab03_Cost_Function.ipynb
├── C1_W3_Lab04_Gradient_Descent.ipynb
├── C1_W3_Lab05_One_Vs_All.ipynb
├── C1_W3_Lab06_Regularized_Cost.ipynb
├── C1_W3_Lab07_Regularized_Gradient_Descent.ipynb
├── C1_W3_Lab08_Scikit_Learn.ipynb
├── 21083541_PhamDaoThanhDat.rar
└── 21083541_PhamDaoThanhDat/
    ├── Clusster_DataMining/
    │   └── src/kmean/
    ├── J48_DataMining/
    │   └── src/j48/
    ├── 21083541_PhamDaoThanhDat_Cluster_DataMining.docx
    └── 21083541_PhamDaoThanhDat_J48.docx
```

## Chạy các bài thực hành

### Notebook Python

Chuẩn bị Python 3 và cài các thư viện được sử dụng trong notebook:

```bash
git clone https://github.com/Lyxnpham2003/NLPonSchool.git
cd NLPonSchool
python -m pip install notebook numpy matplotlib pandas scikit-learn
python -m notebook
```

Mở notebook cần xem và chạy các ô lệnh theo thứ tự từ trên xuống. Có thể bắt đầu với Lab 01 rồi tiếp tục theo bảng nội dung.

**Phụ thuộc còn thiếu:** Lab 02, 03 và 04 import `plot_data` và `sigmoid` từ `lab_utils`. Repo hiện chưa có `lab_utils.py`; cần bổ sung file hỗ trợ tương ứng từ bộ bài thực hành gốc vào cùng thư mục notebook trước khi chạy các bài này.

### Java/Weka

Hai dự án được cấu hình cho **JavaSE-19** và **Weka 3.8.6** trong Eclipse.

1. Chuẩn bị JDK 19, Eclipse có hỗ trợ Java và Weka 3.8.6.
2. Trong Eclipse, chọn **File → Import → Existing Projects into Workspace**, rồi chọn thư mục `21083541_PhamDaoThanhDat` để import hai dự án.
3. Kiểm tra **Java Build Path** của từng dự án. File `.classpath` hiện trỏ đến `C:/Program Files/Weka-3-8-6/weka.jar`; sửa đường dẫn nếu Weka nằm ở vị trí khác.
4. Sửa đường dẫn dữ liệu trong hai lớp chạy mẫu nếu cần. Mã hiện đọc `C:\Program Files\Weka-3-8-6\data\weather.nominal.arff`; file ARFF này chưa được lưu riêng trong repo.
5. Chạy `kmean.testClusteringModel` hoặc `j48.testClassificationModel` bằng **Run As → Java Application** để xem kết quả trong console.

## Lưu ý khi đọc kết quả

- Lab 05 hiện là thử nghiệm so sánh ba mô hình trên dữ liệu nhị phân. Phần kết hợp dùng `argmax` để chọn **chỉ số mô hình** rồi ánh xạ thành nhãn `0`, `1`, `2`, trong khi dữ liệu chỉ có nhãn `0`, `1`; phần này cần điều chỉnh để đánh giá kết quả kết hợp đúng nghĩa hoặc triển khai One-vs-All.
- Đồ thị cuối Lab 07 sử dụng dữ liệu minh họa từ `log` và `sin`. Muốn xem quá trình tối ưu thực tế, hãy vẽ lịch sử `J_history` do hàm `gradient_descent` trả về.
- Lab 08 đo độ chính xác ngay trên tập huấn luyện; kết quả này chưa thể hiện khả năng dự đoán trên dữ liệu mới.
- Ví dụ J48 chia dữ liệu theo thứ tự có sẵn, chưa xáo trộn trước khi chia tập.
- `.gitignore` bỏ qua metadata Eclipse, checkpoint Jupyter, cache Python và file Java đã biên dịch.
