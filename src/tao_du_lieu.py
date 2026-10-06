"""Sinh thêm dữ liệu bài hát để AIC / BIC có đủ bằng chứng chọn k.

Bộ data/BaiHat.csv chỉ có 20 bài, quá ít nên AIC lẫn BIC đều chọn k = 1.
Script này sinh ra data/BaiHatLon.csv từ một hỗn hợp Gauss ĐÃ BIẾT TRƯỚC
tham số, để sau đó kiểm tra xem GMM + BIC có tìm lại đúng số cụm hay không.

Chạy: python3 src/tao_du_lieu.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "BaiHatLon.csv"

N_SONGS = 600
SEED = 7

# Tham số thật của 4 cụm: ballad -> nhạc nhẹ -> pop sôi động -> EDM
TRUE_WEIGHTS = np.array([0.25, 0.30, 0.28, 0.17])
TRUE_MEANS = np.array([2.2, 4.4, 6.6, 8.6])
TRUE_STDS = np.array([0.55, 0.60, 0.65, 0.50])
CLUSTER_NAMES = ["ballad", "nhạc nhẹ", "pop sôi động", "EDM"]

# Ghép từ để đặt tên bài hát cho giống dữ liệu thật
PREFIX = ["Giấc Mơ", "Đêm", "Nắng", "Mưa", "Hạ", "Thu", "Biển", "Phố", "Em",
          "Anh", "Tình", "Cơn Gió", "Ngôi Sao", "Bầu Trời", "Con Đường",
          "Lời Hứa", "Khoảnh Khắc", "Vũ Điệu", "Nhịp Tim", "Ánh Trăng"]
SUFFIX = ["Mùa Hạ", "Cuối Cùng", "Dịu Dàng", "Xa Xôi", "Rực Rỡ", "Bình Yên",
          "Vội Vã", "Chờ Đợi", "Không Tên", "Tháng Sáu", "Trong Mơ", "Lặng Lẽ",
          "Phiêu Lưu", "Bất Tận", "Ngọt Ngào", "Cháy Hết Mình", "Đầu Tiên",
          "Của Em", "Giữa Đêm", "Mãi Mãi"]


def make_names(n, rng):
    """Tạo n tên bài hát khác nhau."""
    names, used = [], set()
    while len(names) < n:
        name = f"{rng.choice(PREFIX)} {rng.choice(SUFFIX)}"
        if name in used:
            name = f"{name} {len([x for x in names if x.startswith(name)]) + 2}"
        used.add(name)
        names.append(name)
    return names


rng = np.random.default_rng(SEED)

# Mỗi bài hát: bốc một cụm theo pi, rồi lấy điểm sôi động từ chuông của cụm đó
components = rng.choice(len(TRUE_WEIGHTS), size=N_SONGS, p=TRUE_WEIGHTS)
scores = rng.normal(TRUE_MEANS[components], TRUE_STDS[components])
scores = np.round(np.clip(scores, 0, 10), 1)      # giữ 1 chữ số thập phân như dữ liệu gốc

df = pd.DataFrame({"tenBaiHat": make_names(N_SONGS, rng), "diemSoiDong": scores})
df.to_csv(OUT_PATH, index=False)

print(f"Đã sinh {len(df)} bài hát từ {len(TRUE_WEIGHTS)} cụm -> {OUT_PATH}")
print(f"\n{'cụm':>4} {'tên':>14} {'pi thật':>9} {'mu thật':>9} {'sigma thật':>11} {'số bài':>8}")
for k in range(len(TRUE_WEIGHTS)):
    n_k = int((components == k).sum())
    print(f"{k + 1:>4} {CLUSTER_NAMES[k]:>14} {TRUE_WEIGHTS[k]:>9.2f} "
          f"{TRUE_MEANS[k]:>9.2f} {TRUE_STDS[k]:>11.2f} {n_k:>8}")

print(f"\nĐiểm sôi động: min={df['diemSoiDong'].min():.1f}, "
      f"max={df['diemSoiDong'].max():.1f}, "
      f"trung bình={df['diemSoiDong'].mean():.2f}")
print(df.head(8).to_string(index=False))
print("\nGiờ đổi DATA_FILE trong src/gmm_baihat.py thành \"BaiHatLon.csv\" "
      "rồi chạy lại để xem AIC / BIC chọn k nào.")
