"""GMM 2 cụm trên cột diemSoiDong của data/BaiHat.csv."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent  # thư mục gốc của repo

# ---------- Phần 1: Huấn luyện GMM với 2 cụm ----------

df = pd.read_csv(ROOT / "data" / "BaiHat.csv")
X = df[["diemSoiDong"]].to_numpy()  # tập X 1 chiều, shape (n, 1)

gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
gmm.fit(X)

weights = gmm.weights_                      # trọng số pi_k
means = gmm.means_.ravel()                  # trung bình mu_k
stds = np.sqrt(gmm.covariances_.ravel())    # độ lệch chuẩn sigma_k

# Sắp cụm theo mean tăng dần để cụm 1 = "ít sôi động", cụm 2 = "sôi động"
order = np.argsort(means)
weights, means, stds = weights[order], means[order], stds[order]
remap = {old: new + 1 for new, old in enumerate(order)}
df["cum"] = [remap[l] for l in gmm.predict(X)]

for k in range(2):
    print(f"Cụm {k + 1}: pi={weights[k]:.3f}  mu={means[k]:.3f}  sigma={stds[k]:.3f}")
print(f"Hội tụ: {gmm.converged_} | log-likelihood trung bình: {gmm.score(X):.3f}")
print(df.sort_values("diemSoiDong").to_string(index=False))

# ---------- Phần 2: Visualize 2 chuông + GMM kết quả ----------

def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

grid = np.linspace(X.min() - 2, X.max() + 2, 500)
bell1 = weights[0] * normal_pdf(grid, means[0], stds[0])
bell2 = weights[1] * normal_pdf(grid, means[1], stds[1])
mixture = bell1 + bell2

fig, ax = plt.subplots(figsize=(10, 6))

# Mô hình GMM kết quả (tím) - đường dày, liền nét
ax.plot(grid, mixture, color="purple", lw=3.5, alpha=0.9,
        label="GMM (hỗn hợp 2 chuông)")

# Chuông 1 (đỏ, nét đứt) và chuông 2 (xanh, nét gạch-chấm)
ax.plot(grid, bell1, color="red", lw=2, ls="--",
        label=f"Chuông 1 (μ={means[0]:.2f}, σ={stds[0]:.2f}, π={weights[0]:.2f})")
ax.plot(grid, bell2, color="blue", lw=2, ls="-.",
        label=f"Chuông 2 (μ={means[1]:.2f}, σ={stds[1]:.2f}, π={weights[1]:.2f})")

# Đánh dấu tâm mỗi chuông
ax.axvline(means[0], color="red", lw=1, ls=":", alpha=0.6)
ax.axvline(means[1], color="blue", lw=1, ls=":", alpha=0.6)

# Các điểm dữ liệu trên trục X: mỗi chấm = 1 bài hát, tô theo cụm được gán
colors = np.where(df["cum"].to_numpy() == 1, "red", "blue")
ax.scatter(X.ravel(), np.zeros_like(X.ravel()), s=60, c=colors,
           edgecolors="black", linewidths=0.6, zorder=5, clip_on=False)

ax.set_xlabel("Điểm sôi động - Mỗi chấm tròn tương ứng điểm sôi động của một bài hát", fontsize=12)
ax.set_ylabel("Mật độ xác suất", fontsize=12)
ax.set_ylim(bottom=0)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out_path = ROOT / "gmm_baihat.png"
fig.savefig(out_path, dpi=150)
print(f"Đã lưu biểu đồ: {out_path}")
plt.show()
