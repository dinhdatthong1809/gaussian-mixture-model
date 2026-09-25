## Chạy như nào ?

```bash
# 1. Tạo môi trường ảo (khuyến nghị, để không đụng Python hệ thống)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Cài thư viện
pip3 install -r requirements.txt

# 3. Chạy
python3 src/gmm_baihat.py

# 4. Xem vì sao chỉ dùng K-Means + tính mean/std là chưa đủ
python3 src/kmeans_vs_gmm.py
```

## Web app trực quan hoá GMM

App tĩnh chạy thẳng trong trình duyệt, không cần cài gì thêm:

```bash
# Mở trực tiếp
xdg-open web/index.html        # macOS: open web/index.html

# Hoặc chạy qua server tĩnh
python3 -m http.server 8000    # rồi mở http://localhost:8000/web/
```

Giao diện có nút chuyển **English / Tiếng Việt** ở góc trên bên phải (mặc định
tiếng Anh, lựa chọn được nhớ lại ở lần mở sau). Trong app có thể chọn dữ liệu **1D hoặc 2D**, chọn cách biểu thị mô hình là
**khoanh vùng ellipse** hoặc **đồ thị đèo núi**, nhấp chuột để thêm điểm, và
chạy thuật toán EM từng bước (bước E / bước M / tự động chạy).
