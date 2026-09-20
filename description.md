Chủ đề:
Clustering: Gaussian Mixture Models and Expectation-Maximization method
Phân cụm: "Mô hình hỗn hợp phân phối chuẩn" và "phương pháp tối đa hoá kỳ vọng"

1 - Vấn đề và ngữ cảnh:
+ Đời thường:
  - Tôi có cái web nghe nhạc, người dùng càng nghe thì tôi càng có tiền quảng cáo
  - Làm chức năng gợi ý danh sách nhac tương tự

+ Toán học:
  - Phân cụm
  - Tìm tương đồng


2 - Giải pháp:
  Áp dụng mô hình học không giám sát để phân cụm dữ liệu
  Sữ dụng mô hình hỗn hợp phân phối chuẩn

phân phối chuẩn (aka Gaussian): cái chuông

Phương trình biểu diễn mô hình:


Mô hình núi đèo = (trọng số 1)(cái chuông 1) + (trọng số 2)(cái chuông 2) + ... + (trọng số k)(cái chuông k)

Công thức toán dữ liệu 1 chiều:

$$
p(x) = \sum_{i=1}^{k} \pi_i \cdot \frac{1}{\sqrt{2\pi\sigma_i^2}}e^{-\frac{(x-\mu_i)^2}{2\sigma_i^2}}
$$

Kiến thức cần có:
- Cái chuông chuẩn (Normal distribution) là gì ?
  + Công thức:
    $$
    \mathcal{N}(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    $$

Điều chỉnh 2 thứ:
- Tâm cái chuông (Trung bình cộng các điểm dữ liệu: $\mu$)
- Độ mập (Độ lệch chuẩn $\sigma^2$)

Các bước vẽ mô hình núi đèo:
- Tự đặt số lượng cụm k - để chọn k tốt thì xài phương pháp BIC (Bayesian Information Criterion) hoặc AIC (Akaike Information Criterion) để chọn k sao cho giá trị BIC hoặc AIC là thấp nhất
- Tìm tâm của mỗi cụm (xài luôn K-Means)
- Vẽ từng cái chuông dựa vào tâm, rồi úp lên dữ liệu sao cho khớp nhất
- Nối tất cả cái chuông tạo thành mô hình cuối cùng

2.1 - Thử với dữ liệu 1 chiều

2.1.1 - Cái chuông Phân phối chuẩn Guassian
Công thức cái chuông chuẩn:
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

Vẽ từng cái chuông úp lên dữ liệu sao cho khớp nhất

Toán học:
Điều chỉnh 2 thông số trong công thức:
- Tâm cái chuông (Trung bình cộng các điểm dữ liệu: $\mu$)
- Độ mập của cái chuông (Độ lệch chuẩn $\sigma^2$)

input, mô hình, output

Ta có 100 bộ truyện, mỗi bộ có số chương

| Tên truyện | Tổng số chương |
|


Đề tài liên quan:
Laplace Mixture Model