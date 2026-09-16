# Clustering: Gaussian Mixture Models & Expectation–Maximization

Đề tài thuyết trình gồm hai phần: **báo cáo lý thuyết** (tiếng Việt, có hình minh hoạ) và **ứng dụng Python** train/dự đoán bằng GMM.

## Nội dung

```
report/BAO_CAO_GMM.md      báo cáo chính: bài toán, toán học 1D → ma trận, EM, ưu/nhược điểm
report/figures/            7 hình minh hoạ (sinh lại bằng report/make_figures.py)
report/make_figures.py     script sinh toàn bộ hình
src/gmm_app.py             ứng dụng CLI: train / predict / eval / outliers
src/gmm_from_scratch.py    EM cài từ đầu bằng NumPy (đối chiếu với công thức trong báo cáo)
src/plots.py               tiện ích vẽ (ellipse, gán mềm)
data/old_faithful.csv      bộ dữ liệu ví dụ: 272 lần phun của mạch nước Old Faithful
data/new_samples.csv       vài mẫu mới để thử lệnh predict
```

## Bộ dữ liệu

**Old Faithful geyser** (Yellowstone, 272 quan sát) — bộ dữ liệu kinh điển để minh hoạ GMM vì nó có đúng hai cụm hình ellipse nghiêng:

| cột | ý nghĩa |
|---|---|
| `duration` | thời gian phun (phút) |
| `waiting` | thời gian chờ tới lần phun kế tiếp (phút) |
| `kind` | nhãn tham chiếu `short`/`long` — **không dùng khi train**, chỉ để chấm điểm |

Muốn đổi dữ liệu: truyền `--data <file.csv>`, miễn là file có hai cột `duration`, `waiting`.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

```bash
# 1) Huấn luyện, tự chọn số cụm K bằng BIC, kèm biểu đồ
python3 src/gmm_app.py train --k auto --plot

# ép K và đổi ràng buộc covariance
python3 src/gmm_app.py train --k 2 --cov diag

# 2) Dự đoán cụm + xác suất hậu nghiệm cho điểm mới
python3 src/gmm_app.py predict --input "4.1,82" "1.9,54" "3.0,68"
python3 src/gmm_app.py predict --csv data/new_samples.csv

# 3) Đánh giá mô hình đã lưu (log-likelihood, BIC, ARI so với nhãn thật)
python3 src/gmm_app.py eval

# 4) Phát hiện bất thường: các điểm có log p(x) thấp nhất
python3 src/gmm_app.py outliers --top 5

# 5) Sinh lại toàn bộ hình cho báo cáo
python3 report/make_figures.py

# 6) Chạy bản EM cài từ đầu
python3 src/gmm_from_scratch.py
```

## Kết quả trên Old Faithful

- BIC chọn **K = 2**, hội tụ sau 10 vòng EM.
- Cụm 0: π=0.356, duration ≈ 2.04 phút, waiting ≈ 54.5 phút — cụm 1: π=0.644, duration ≈ 4.29 phút, waiting ≈ 80.0 phút.
- **Adjusted Rand Index = 0.927** so với nhãn thật (chỉ 5/272 điểm lệch), dù mô hình hoàn toàn không thấy nhãn.
