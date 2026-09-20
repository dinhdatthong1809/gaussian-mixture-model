"""GMM 2 cụm trên cột diemSoiDong của data/BaiHat.csv."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ---------- Phần 1: Huấn luyện GMM với 2 cụm ----------

df = pd.read_csv("data/BaiHat.csv")
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

# Mô hình GMM kết quả (tím) - vẽ nền dày để thấy rõ dưới 2 chuông
ax.fill_between(grid, mixture, color="purple", alpha=0.20)
ax.plot(grid, mixture, color="purple", lw=5, alpha=0.9,
        label="GMM (hỗn hợp 2 chuông)")

# Chuông 1 (đỏ) và chuông 2 (xanh)
ax.fill_between(grid, bell1, color="red", alpha=0.35)
ax.plot(grid, bell1, color="red", lw=2, ls="--",
        label=f"Chuông 1 (μ={means[0]:.2f}, σ={stds[0]:.2f})")
ax.fill_between(grid, bell2, color="blue", alpha=0.35)
ax.plot(grid, bell2, color="blue", lw=2, ls="--",
        label=f"Chuông 2 (μ={means[1]:.2f}, σ={stds[1]:.2f})")

# Các điểm dữ liệu trên trục X: mỗi chấm = 1 bài hát
ax.scatter(X.ravel(), np.zeros_like(X.ravel()), s=60, color="black",
           zorder=5, clip_on=False, label="Mỗi chấm tương ứng 1 bài nhạc")

ax.set_xlabel("điểm sôi động", fontsize=12)
ax.set_ylabel("mật độ xác suất", fontsize=12)
ax.set_title("GMM 2 cụm trên điểm sôi động của các bài hát", fontsize=13)
ax.set_ylim(bottom=0)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("gmm_baihat.png", dpi=150)
plt.show()
