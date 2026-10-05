"""GMM trên một tập con nhỏ của MNIST — cùng 3 phần như bài BaiHat.

Phần 1: huấn luyện GMM.
Phần 2: visualize chính mô hình vừa huấn luyện.
Phần 3: đưa một ảnh mới (chưa từng thấy khi train) vào mô hình và xem đầu ra.

Trước đó có phần 0 trả lời câu hỏi "có phải tách nền ảnh không?".

Lần chạy đầu sẽ tải MNIST (~11 MB) về data/mnist.npz rồi dùng lại cho các lần sau.
Chạy: python3 src/gmm_mnist.py
"""

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
MNIST_PATH = ROOT / "data" / "mnist.npz"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

DIGITS = [0, 1, 8]        # chỉ lấy vài chữ số cho dễ nhìn
N_PER_DIGIT = 150         # mỗi chữ số lấy bấy nhiêu ảnh -> tập train nhỏ
N_PCA = 2                 # số chiều PCA: train và vẽ trên cùng không gian này
N_CLUSTERS = len(DIGITS)
SEED = 0
COLORS = ["red", "blue", "green", "orange", "purple", "brown", "magenta",
          "olive", "cyan", "gray"]


def load_mnist():
    """Đọc MNIST từ data/mnist.npz, tải về nếu chưa có."""
    if not MNIST_PATH.exists():
        print(f"Đang tải MNIST về {MNIST_PATH} ...")
        MNIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(MNIST_URL) as resp:
            MNIST_PATH.write_bytes(resp.read())
    d = np.load(MNIST_PATH)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def take_subset(images, labels, digits, n_per_digit, seed=SEED):
    """Lấy n_per_digit ảnh cho mỗi chữ số trong digits."""
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == g), n_per_digit,
                                     replace=False) for g in digits])
    rng.shuffle(idx)
    return images[idx].reshape(len(idx), -1) / 255.0, labels[idx]


x_train, y_train, x_test, y_test = load_mnist()
X_raw, y_sub = take_subset(x_train, y_train, DIGITS, N_PER_DIGIT)


# ==========================================================
# Phần 0: Có phải tách nền ảnh không?
# ==========================================================
# Câu trả lời: KHÔNG cần tách nền — nền MNIST đã là 0 tuyệt đối và đồng nhất,
# không có gì để tách. Nhưng chính vì vậy mà rất nhiều pixel viền LUÔN bằng 0
# trong mọi ảnh, tức phương sai = 0. Với covariance_type="full", ma trận hiệp
# phương sai khi đó suy biến (không nghịch đảo được) và EM gãy ngay.
# Cách xử lý đúng không phải là "xoá nền" mà là bỏ các chiều vô dụng đó đi:
# giảm chiều bằng PCA (dùng ở đây), hoặc covariance_type="diag", hoặc reg_covar.

print("--- Phần 0: Nền ảnh ---")
print(f"Tập train: {X_raw.shape[0]} ảnh × {X_raw.shape[1]} pixel "
      f"(các chữ số {DIGITS})")
print(f"Tỉ lệ pixel bằng 0 (nền): {(X_raw == 0).mean():.1%}")
n_const = int((X_raw.var(axis=0) == 0).sum())
print(f"Pixel luôn bằng 0 ở mọi ảnh (phương sai = 0): {n_const}/{X_raw.shape[1]}")
try:
    GaussianMixture(n_components=N_CLUSTERS, covariance_type="full",
                    reg_covar=0, random_state=SEED, max_iter=20).fit(X_raw)
    print("Fit thẳng trên 784 pixel: không lỗi (bất ngờ).")
except ValueError as err:
    print(f"Fit thẳng trên 784 pixel -> ValueError: {str(err)[:90]}...")
    print("=> Không phải do 'còn nền', mà do các pixel hằng số làm ma trận suy biến.")


# ==========================================================
# Phần 1: Huấn luyện GMM (trên không gian PCA)
# ==========================================================

pca = PCA(n_components=N_PCA, random_state=SEED)
X = pca.fit_transform(X_raw)          # mỗi ảnh -> 1 điểm N_PCA chiều

gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type="full",
                      n_init=10, random_state=SEED)
gmm.fit(X)

weights = gmm.weights_
means = gmm.means_
covs = gmm.covariances_
labels = gmm.predict(X)               # cụm được gán cho từng ảnh

print(f"\n--- Phần 1: Huấn luyện GMM {N_CLUSTERS} cụm trên {N_PCA} thành phần PCA ---")
print(f"PCA giữ lại {pca.explained_variance_ratio_.sum():.1%} phương sai")
print(f"Hội tụ: {gmm.converged_} | log-likelihood trung bình: {gmm.score(X):.3f}")

# GMM là học không giám sát: nó không biết nhãn. Đối chiếu cụm với chữ số thật
# để xem mỗi cụm "bắt" được chữ số nào.
cluster_digit = {}
for k in range(N_CLUSTERS):
    members = y_sub[labels == k]
    vals, counts = np.unique(members, return_counts=True)
    cluster_digit[k] = int(vals[np.argmax(counts)])
    print(f"Cụm {k}: pi={weights[k]:.3f}  n={len(members):>4}  "
          f"chữ số chiếm đa số = {cluster_digit[k]}  "
          f"(độ thuần khiết {counts.max() / len(members):.1%})")
purity = np.mean([cluster_digit[l] == d for l, d in zip(labels, y_sub)])
print(f"Độ chính xác nếu coi mỗi cụm là một chữ số: {purity:.1%}")

# Nhiều chiều PCA hơn thì cụm sạch hơn, nhưng không vẽ được trên mặt phẳng
X40 = PCA(n_components=40, random_state=SEED).fit_transform(X_raw)
g40 = GaussianMixture(n_components=N_CLUSTERS, covariance_type="full",
                      n_init=10, random_state=SEED).fit(X40)
l40 = g40.predict(X40)
p40 = np.mean([int(np.bincount(y_sub[l40 == l]).argmax()) == d
               for l, d in zip(l40, y_sub)])
print(f"(Để so sánh: dùng 40 chiều PCA thì độ chính xác là {p40:.1%})")


# ==========================================================
# Phần 2: Visualize chính mô hình vừa huấn luyện
# ==========================================================

ncol = max(N_CLUSTERS, 3)
fig = plt.figure(figsize=(6 + 1.9 * ncol, 8))
gs = fig.add_gridspec(3, 3 + ncol, hspace=0.45, wspace=0.5)
ax = fig.add_subplot(gs[:, :3])

# Các ảnh train, tô theo cụm được gán
for k in range(N_CLUSTERS):
    m = labels == k
    ax.scatter(X[m, 0], X[m, 1], s=18, color=COLORS[k], alpha=0.55,
               label=f"Cụm {k} (≈ chữ số {cluster_digit[k]}, π={weights[k]:.2f})")

# Mỗi cụm: ellipse 1σ / 2σ / 3σ lấy từ ma trận hiệp phương sai
for k in range(N_CLUSTERS):
    vals, vecs = np.linalg.eigh(covs[k])
    angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    for n_sigma, alpha in [(1, 0.9), (2, 0.5), (3, 0.3)]:
        w_, h_ = 2 * n_sigma * np.sqrt(vals[::-1])
        ax.add_patch(plt.matplotlib.patches.Ellipse(
            means[k], w_, h_, angle=angle, facecolor="none",
            edgecolor=COLORS[k], lw=2 if n_sigma == 1 else 1.2, alpha=alpha))
    ax.plot(*means[k], marker="+", ms=14, mew=2.5, color=COLORS[k])

ax.set_xlabel("Thành phần chính 1 (PC1)")
ax.set_ylabel("Thành phần chính 2 (PC2)")
ax.set_title(f"GMM {N_CLUSTERS} cụm trên {len(X)} ảnh MNIST (chiếu xuống 2 chiều PCA)")
ax.legend(fontsize=9, loc="best")
ax.spines[["top", "right"]].set_visible(False)

# Tâm mỗi cụm trong không gian PCA, dựng ngược lại thành ảnh 28×28
for k in range(N_CLUSTERS):
    ax_k = fig.add_subplot(gs[0, 3 + k])
    ax_k.imshow(pca.inverse_transform(means[k:k + 1])[0].reshape(28, 28), cmap="gray_r")
    ax_k.set_title(f"Tâm cụm {k}", fontsize=10, color=COLORS[k])
    ax_k.set_xticks([]); ax_k.set_yticks([])
    for s in ax_k.spines.values():
        s.set_color(COLORS[k]); s.set_linewidth(2)


# ==========================================================
# Phần 3: Test một ảnh mới chưa từng thấy khi train
# ==========================================================

test_idx = int(np.flatnonzero(np.isin(y_test, DIGITS))[7])
img_new = x_test[test_idx]
true_digit = int(y_test[test_idx])
x_new = pca.transform(img_new.reshape(1, -1) / 255.0)[0]   # cùng phép chiếu PCA

# Mật độ có trọng số của từng cụm: pi_k * N(x | mu_k, Sigma_k)
comp = np.array([weights[k] * multivariate_normal(means[k], covs[k]).pdf(x_new)
                 for k in range(N_CLUSTERS)])
px = comp.sum()
post = comp / px
best = int(np.argmax(post))

print("\n--- Phần 3: Thử một ảnh mới (lấy từ tập test) ---")
print(f"Ảnh test #{test_idx}, chữ số thật = {true_digit}")
print(f"Toạ độ sau PCA: PC1={x_new[0]:.3f}, PC2={x_new[1]:.3f}")
print(f"  {'cụm':>4} {'pi_k':>7} {'N(x|k)':>10} {'pi_k*N':>10} {'hậu nghiệm':>12}")
for k in range(N_CLUSTERS):
    print(f"  {k:>4} {weights[k]:>7.3f} "
          f"{multivariate_normal(means[k], covs[k]).pdf(x_new):>10.6f} "
          f"{comp[k]:>10.6f} {post[k]:>11.1%}")
print(f"  -> p(x) = {px:.6f}  |  log p(x) = {np.log(px):.4f}")
print(f"  -> Dự đoán: CỤM {best} (≈ chữ số {cluster_digit[best]}) "
      f"với xác suất {post[best]:.1%}")
print(f"  -> Chữ số thật là {true_digit} => "
      f"{'ĐÚNG' if cluster_digit[best] == true_digit else 'SAI'}")

# Công thức tay ở trên phải khớp sklearn
assert np.allclose(post, gmm.predict_proba([x_new])[0])
assert np.isclose(np.log(px), gmm.score_samples([x_new])[0])

# Ảnh mới nằm ở đâu trên biểu đồ
ax.plot(x_new[0], x_new[1], marker="*", ms=26, color=COLORS[best],
        markeredgecolor="black", markeredgewidth=1.2, zorder=6,
        label="ảnh mới đem test")
ax.annotate(f"ảnh mới (số {true_digit})\n-> cụm {best} ({post[best]:.0%})",
            xy=x_new, xytext=(14, 18), textcoords="offset points",
            fontsize=10, fontweight="bold", color=COLORS[best],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLORS[best], alpha=0.9))
ax.legend(fontsize=9, loc="best")

ax_img = fig.add_subplot(gs[1, 3])
ax_img.imshow(img_new, cmap="gray_r")
ax_img.set_title(f"Ảnh mới (số {true_digit})", fontsize=10)
ax_img.set_xticks([]); ax_img.set_yticks([])

ax_bar = fig.add_subplot(gs[1, 4:])
ax_bar.bar(range(N_CLUSTERS), post, color=COLORS[:N_CLUSTERS])
ax_bar.set_title("Xác suất hậu nghiệm của ảnh mới", fontsize=10, pad=14)
ax_bar.set_xticks(range(N_CLUSTERS))
ax_bar.set_xticklabels([f"cụm {k}" for k in range(N_CLUSTERS)], fontsize=9)
ax_bar.set_ylim(0, 1.2)
for k in range(N_CLUSTERS):
    ax_bar.text(k, post[k] + 0.03, f"{post[k]:.0%}", ha="center", fontsize=9)
ax_bar.spines[["top", "right"]].set_visible(False)

out_path = ROOT / "gmm_mnist.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nĐã lưu biểu đồ: {out_path}")
plt.show()
