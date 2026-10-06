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

# 5. Xem vì sao mỗi vòng lặp EM không bao giờ làm giảm log-likelihood
python3 src/elbo_em.py

# 6. GMM trên một tập con nhỏ của MNIST (lần đầu chạy sẽ tải ~11 MB về data/)
python3 src/gmm_mnist.py

# 7. Sinh bộ dữ liệu 600 bài hát (4 cụm) để AIC/BIC đủ dữ liệu chọn k
python3 src/tao_du_lieu.py
# rồi đổi DATA_FILE thành "BaiHatLon.csv" trong src/gmm_baihat.py và chạy lại
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
