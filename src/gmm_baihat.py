"""GMM trên cột diemSoiDong của data/BaiHat.csv.

Phần 0: chọn số cụm k tốt nhất theo AIC / BIC.
Phần 1: huấn luyện mô hình.
Phần 2: visualize chính mô hình vừa huấn luyện.
Phần 3: đưa một điểm dữ liệu mới vào mô hình đó và xem đầu ra.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent  # thư mục gốc của repo

DATA_FILE = "BaiHat.csv"       # đổi thành "BaiHatLon.csv" (600 bài, sinh bởi src/tao_du_lieu.py)
N_CLUSTERS = 2                 # đổi số cụm ở đây; đặt None để tự lấy k tốt nhất theo BIC
K_RANGE = range(1, 9)          # dải k đem ra so sánh bằng AIC / BIC
TEST_POINTS = [5.8, 8.2]       # các điểm sôi động mới muốn thử với mô hình
COLORS = ["red", "blue", "green", "orange", "purple"]
LINE_STYLES = ["--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1, 1, 1))]


def normal_pdf(x, mu, sigma):
    """Hàm mật độ N(x; mu, sigma)."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


df = pd.read_csv(ROOT / "data" / DATA_FILE)
X = df[["diemSoiDong"]].to_numpy()  # tập X 1 chiều, shape (n, 1)


# ==========================================================
# Phần 0: Chọn số cụm k theo AIC / BIC
# ==========================================================
# Thêm cụm thì log-likelihood LUÔN tăng, nên không thể chọn k bằng likelihood.
# AIC và BIC cộng thêm một khoản phạt theo số tham số p = 3k - 1 (k mu + k sigma
# + (k-1) trọng số độc lập):
#     AIC = -2 * logLik + 2p
#     BIC = -2 * logLik + p * ln(n)
# Cả hai đều CÀNG NHỎ CÀNG TỐT. Với n >= 8 thì ln(n) > 2 nên BIC phạt nặng hơn
# AIC, tức BIC thường chọn mô hình gọn hơn.

scores = []
for k in K_RANGE:
    g = GaussianMixture(n_components=k, covariance_type="full",
                        n_init=10, random_state=0).fit(X)
    scores.append({"k": k, "p": 3 * k - 1, "logLik": g.score(X) * len(X),
                   "AIC": g.aic(X), "BIC": g.bic(X)})

scores = pd.DataFrame(scores)
best_aic = int(scores.loc[scores["AIC"].idxmin(), "k"])
best_bic = int(scores.loc[scores["BIC"].idxmin(), "k"])

print(f"--- Phần 0: Chọn k theo AIC / BIC ({DATA_FILE}, {len(X)} bài hát) ---")
print(scores.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))
print(f"k tốt nhất theo AIC: {best_aic} | k tốt nhất theo BIC: {best_bic}")

if N_CLUSTERS is None:
    N_CLUSTERS = best_bic
    print(f"N_CLUSTERS = None -> dùng k tốt nhất theo BIC = {N_CLUSTERS}")
elif N_CLUSTERS != best_bic:
    print(f"Lưu ý: đang chạy với k = {N_CLUSTERS} trong khi BIC chọn k = {best_bic}.")

# Vẽ đường AIC / BIC theo k, đánh dấu điểm cực tiểu
fig_k, ax_k = plt.subplots(figsize=(8, 4.5))
ax_k.plot(scores["k"], scores["AIC"], marker="o", color="#e08a1e", label="AIC")
ax_k.plot(scores["k"], scores["BIC"], marker="s", color="#0d7c86", label="BIC")
for name, k_best, color in [("AIC", best_aic, "#e08a1e"), ("BIC", best_bic, "#0d7c86")]:
    y_best = float(scores.loc[scores["k"] == k_best, name].iloc[0])
    ax_k.scatter([k_best], [y_best], s=180, facecolors="none",
                 edgecolors=color, linewidths=2.5, zorder=5)
    ax_k.annotate(f"{name} nhỏ nhất tại k={k_best}", xy=(k_best, y_best),
                  xytext=(12, -20 if name == "AIC" else 14),
                  textcoords="offset points", color=color, fontweight="bold")
ax_k.set_xlabel("số cụm k")
ax_k.set_ylabel("giá trị tiêu chí (càng nhỏ càng tốt)")
ax_k.set_title("Chọn số cụm bằng AIC / BIC")
ax_k.set_xticks(list(K_RANGE))
ax_k.legend()
ax_k.spines[["top", "right"]].set_visible(False)
fig_k.tight_layout()
k_path = ROOT / "gmm_baihat_aic_bic.png"
fig_k.savefig(k_path, dpi=150)
print(f"Đã lưu biểu đồ chọn k: {k_path}")


# ==========================================================
# Phần 1: Huấn luyện GMM
# ==========================================================

gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type="full", random_state=0)
gmm.fit(X)

weights = gmm.weights_                      # trọng số pi_k
means = gmm.means_.ravel()                  # trung bình mu_k
stds = np.sqrt(gmm.covariances_.ravel())    # độ lệch chuẩn sigma_k

# Sắp cụm theo mean tăng dần: cụm 1 = ít sôi động nhất
order = np.argsort(means)
weights, means, stds = weights[order], means[order], stds[order]
remap = {old: new + 1 for new, old in enumerate(order)}
df["cum"] = [remap[l] for l in gmm.predict(X)]

print(f"\n--- Phần 1: Huấn luyện GMM {N_CLUSTERS} cụm ---")
for k in range(N_CLUSTERS):
    print(f"Cụm {k + 1}: pi={weights[k]:.3f}  mu={means[k]:.3f}  sigma={stds[k]:.3f}")
print(f"Hội tụ: {gmm.converged_} | log-likelihood trung bình: {gmm.score(X):.3f}")
if len(df) <= 40:
    print(df.sort_values("diemSoiDong").to_string(index=False))
else:
    summary = (df.groupby("cum")["diemSoiDong"]
                 .agg(so_bai="size", nho_nhat="min", trung_binh="mean", lon_nhat="max")
                 .round(2))
    print(summary.to_string())
    print("(bảng chi tiết được rút gọn vì dữ liệu nhiều hơn 40 bài)")


# ==========================================================
# Phần 2: Visualize chính mô hình vừa huấn luyện
# ==========================================================

grid = np.linspace(X.min() - 2, X.max() + 2, 500)
bells = np.array([weights[k] * normal_pdf(grid, means[k], stds[k])
                  for k in range(N_CLUSTERS)])
mixture = bells.sum(axis=0)

fig, ax = plt.subplots(figsize=(11, 6.5))

# Mô hình GMM kết quả (tím) - đường dày, liền nét
ax.plot(grid, mixture, color="purple", lw=3.5, alpha=0.9,
        label=f"GMM (hỗn hợp {N_CLUSTERS} chuông)")

# Từng chuông thành phần + vạch dọc tại tâm mu
for k in range(N_CLUSTERS):
    color = COLORS[k % len(COLORS)]
    ax.plot(grid, bells[k], color=color, lw=2, ls=LINE_STYLES[k % len(LINE_STYLES)],
            label=f"Chuông {k + 1} (μ={means[k]:.2f}, σ={stds[k]:.2f}, π={weights[k]:.2f})")
    ax.axvline(means[k], color=color, lw=1, ls=":", alpha=0.6)

# Các điểm dữ liệu trên trục X: mỗi chấm = 1 bài hát, tô theo cụm được gán
dot_colors = [COLORS[(l - 1) % len(COLORS)] for l in df["cum"]]
ax.scatter(X.ravel(), np.zeros_like(X.ravel()), s=60, c=dot_colors,
           edgecolors="black", linewidths=0.6, zorder=5, clip_on=False)


# ==========================================================
# Phần 3: Test một điểm dữ liệu mới trên chính mô hình đó
# ==========================================================

print("\n--- Phần 3: Thử điểm dữ liệu mới ---")

for x_new in TEST_POINTS:
    # Mật độ có trọng số của từng cụm: pi_k * N(x | mu_k, sigma_k)
    comp = np.array([weights[k] * normal_pdf(x_new, means[k], stds[k])
                     for k in range(N_CLUSTERS)])
    px = comp.sum()               # p(x): mật độ của cả hỗn hợp tại x
    post = comp / px              # xác suất hậu nghiệm: x thuộc về cụm nào
    best = int(np.argmax(post)) + 1

    print(f"\nĐiểm sôi động mới x = {x_new}")
    print(f"  {'cụm':>4} {'pi_k':>7} {'N(x|k)':>9} {'pi_k*N':>9} {'hậu nghiệm':>12}")
    for k in range(N_CLUSTERS):
        print(f"  {k + 1:>4} {weights[k]:>7.3f} "
              f"{normal_pdf(x_new, means[k], stds[k]):>9.4f} "
              f"{comp[k]:>9.4f} {post[k]:>11.1%}")
    print(f"  -> p(x) = {px:.4f}  |  log p(x) = {np.log(px):.4f}")
    print(f"  -> Dự đoán: CỤM {best} với xác suất {post[best - 1]:.1%}")

    # Đối chiếu với sklearn để chắc chắn công thức tay ở trên đúng
    assert np.allclose(post, gmm.predict_proba([[x_new]])[0][order])
    assert np.isclose(np.log(px), gmm.score_samples([[x_new]])[0])

    # Vẽ điểm mới lên biểu đồ: vạch dọc + ngôi sao tại p(x) + nhãn kết quả
    color = COLORS[(best - 1) % len(COLORS)]
    ax.axvline(x_new, color="black", lw=1.2, ls="--", alpha=0.5)
    ax.plot(x_new, px, marker="*", ms=20, color=color,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6)
    ax.annotate(f"x = {x_new}\ncụm {best} ({post[best - 1]:.1%})",
                xy=(x_new, px), xytext=(0, 28), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))

ax.plot([], [], marker="*", ms=14, ls="none", color="black",
        label="điểm dữ liệu mới đem test")

ax.set_xlabel("Điểm sôi động - Mỗi chấm tròn tương ứng điểm sôi động của một bài hát",
              fontsize=12)
ax.set_ylabel("Mật độ xác suất", fontsize=12)
ax.set_ylim(bottom=0)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out_path = ROOT / "gmm_baihat.png"
fig.savefig(out_path, dpi=150)
print(f"\nĐã lưu biểu đồ: {out_path}")
plt.show()
