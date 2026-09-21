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
