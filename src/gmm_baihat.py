"""GMM trên cột diemSoiDong của data/BaiHat.csv, với số cụm tuỳ chọn.

Chạy:
    python3 src/gmm_baihat.py                 # 2 cụm (mặc định)
    python3 src/gmm_baihat.py --clusters 4    # 4 cụm
    python3 src/gmm_baihat.py -k 2 3 4        # vẽ 3 biểu đồ để so sánh
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent  # thư mục gốc của repo

# Kiểu nét vẽ xoay vòng cho các chuông, để phân biệt được cả khi màu gần nhau
LINE_STYLES = ["--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1, 1, 1))]
MIXTURE_COLOR = "purple"


# ---------- Phần 1: Huấn luyện ----------

def load_data(csv_path=None):
    """Đọc dữ liệu, trả về (df, X) với X là tập 1 chiều shape (n, 1)."""
    df = pd.read_csv(csv_path or ROOT / "data" / "BaiHat.csv")
    return df, df[["diemSoiDong"]].to_numpy()


def fit_gmm(X, n_clusters, random_state=0):
    """Huấn luyện GMM n_clusters cụm, trả về tham số đã sắp theo mu tăng dần.

    Trả về dict: weights (pi), means (mu), stds (sigma), labels (1..n_clusters),
    converged, score (log-likelihood trung bình).
    """
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                          random_state=random_state)
    gmm.fit(X)

    weights = gmm.weights_
    means = gmm.means_.ravel()
    stds = np.sqrt(gmm.covariances_.ravel())

    # Sắp cụm theo mu tăng dần: cụm 1 = ít sôi động nhất
    order = np.argsort(means)
    remap = {old: new + 1 for new, old in enumerate(order)}
    return {
        "k": n_clusters,
        "weights": weights[order],
        "means": means[order],
        "stds": stds[order],
        "labels": np.array([remap[l] for l in gmm.predict(X)]),
        "converged": gmm.converged_,
        "score": gmm.score(X),
    }


def print_model(df, model):
    """In tham số từng cụm và bảng bài hát kèm cụm được gán."""
    print(f"\n=== GMM {model['k']} cụm ===")
    for k in range(model["k"]):
        print(f"Cụm {k + 1}: pi={model['weights'][k]:.3f}  "
              f"mu={model['means'][k]:.3f}  sigma={model['stds'][k]:.3f}")
    print(f"Hội tụ: {model['converged']} | "
          f"log-likelihood trung bình: {model['score']:.3f}")
    table = df.assign(cum=model["labels"]).sort_values("diemSoiDong")
    print(table.to_string(index=False))


# ---------- Phần 2: Visualize ----------

def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def cluster_colors(k, cmap_name="tab10"):
    """Bảng màu cho k cụm; k <= 2 giữ đỏ/xanh như trước cho quen mắt."""
    if k == 1:
        return ["red"]
    if k == 2:
        return ["red", "blue"]
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i % cmap.N) for i in range(k)]


def visualize_gmm(X, model, ax=None, show_dots=True, cmap_name="tab10"):
    """Vẽ từng chuông, đường mật độ GMM và các điểm dữ liệu — cho k cụm bất kỳ.

    X        : dữ liệu 1 chiều, shape (n, 1) hoặc (n,)
    model    : dict do fit_gmm trả về
    ax       : trục matplotlib để vẽ lên; None thì tạo figure mới
    show_dots: có vẽ các chấm dữ liệu trên trục X hay không
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    x = np.asarray(X).ravel()
    k = model["k"]
    weights, means, stds = model["weights"], model["means"], model["stds"]
    colors = cluster_colors(k, cmap_name)

    grid = np.linspace(x.min() - 2, x.max() + 2, 500)
    bells = np.array([w * normal_pdf(grid, m, s)
                      for w, m, s in zip(weights, means, stds)])
    mixture = bells.sum(axis=0)

    # Đường mật độ GMM: dày, liền nét, vẽ trước để nằm dưới các chuông
    ax.plot(grid, mixture, color=MIXTURE_COLOR, lw=3.5, alpha=0.9,
            label=f"GMM (hỗn hợp {k} chuông)")

    # Từng chuông: màu + kiểu nét riêng, kèm vạch dọc tại tâm mu
    for i in range(k):
        ax.plot(grid, bells[i], color=colors[i], lw=2,
                ls=LINE_STYLES[i % len(LINE_STYLES)],
                label=(f"Chuông {i + 1} (μ={means[i]:.2f}, "
                       f"σ={stds[i]:.2f}, π={weights[i]:.2f})"))
        ax.axvline(means[i], color=colors[i], lw=1, ls=":", alpha=0.6)

    # Mỗi chấm = 1 bài hát, tô theo cụm được gán
    if show_dots:
        dot_colors = [colors[l - 1] for l in model["labels"]]
        ax.scatter(x, np.zeros_like(x), s=60, c=dot_colors,
                   edgecolors="black", linewidths=0.6, zorder=5, clip_on=False)

    ax.set_xlabel("Điểm sôi động - Mỗi chấm tròn tương ứng điểm sôi động "
                  "của một bài hát", fontsize=12)
    ax.set_ylabel("Mật độ xác suất", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9 if k > 3 else 10)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", "--clusters", type=int, nargs="+", default=[2],
                        help="số cụm (có thể liệt kê nhiều giá trị để so sánh)")
    parser.add_argument("--seed", type=int, default=0, help="random_state của GMM")
    parser.add_argument("--out", default=None,
                        help="đường dẫn file ảnh (mặc định gmm_baihat.png ở gốc repo)")
    args = parser.parse_args()

    ks = args.clusters
    if any(k < 1 for k in ks):
        parser.error("số cụm phải >= 1")

    df, X = load_data()
    if max(ks) > len(X):
        parser.error(f"số cụm không được vượt quá số điểm dữ liệu ({len(X)})")

    fig, axes = plt.subplots(len(ks), 1, figsize=(10, 6 * len(ks)), squeeze=False)
    for ax, k in zip(axes.ravel(), ks):
        model = fit_gmm(X, k, random_state=args.seed)
        print_model(df, model)
        visualize_gmm(X, model, ax=ax)
        if len(ks) > 1:   # chỉ cần tiêu đề khi xếp nhiều biểu đồ để so sánh
            ax.set_title(f"GMM {k} cụm — log-likelihood TB {model['score']:.3f}",
                         fontsize=12)

    fig.tight_layout()
    out_path = Path(args.out) if args.out else ROOT / "gmm_baihat.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nĐã lưu biểu đồ: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
