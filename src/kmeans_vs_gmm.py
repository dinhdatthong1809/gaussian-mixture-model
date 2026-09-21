"""Vì sao K-Means + tính mean/std theo cụm KHÔNG thay thế được GMM.

Ý tưởng: K-Means gán cứng (hard assignment) mỗi điểm cho đúng 1 cụm, rồi ta
tính kỳ vọng / độ lệch chuẩn trên từng nhóm. Cách này cắt dữ liệu tại đường
biên giữa 2 tâm, nên mỗi nhóm bị CẮT CỤT (truncated) phần đuôi lấn sang cụm
kia. Hệ quả: sigma bị ước lượng nhỏ hơn thật, 2 mu bị đẩy xa nhau, và tỉ lệ
trộn pi bị lệch khi 2 cụm không cân bằng.

Chạy: python3 src/kmeans_vs_gmm.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
RNG_SEED = 0


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def mixture_pdf(x, weights, means, stds):
    return sum(w * normal_pdf(x, m, s) for w, m, s in zip(weights, means, stds))


def sort_params(weights, means, stds):
    """Sắp 2 cụm theo mu tăng dần để so sánh được với nhau."""
    order = np.argsort(means)
    return np.asarray(weights)[order], np.asarray(means)[order], np.asarray(stds)[order]


def fit_kmeans_then_moments(X, k=2, seed=RNG_SEED):
    """Cách 'ngây thơ': K-Means gán cứng -> tính mean/std trên từng nhóm."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    labels = km.labels_
    x = X.ravel()
    weights = np.array([(labels == j).mean() for j in range(k)])
    means = np.array([x[labels == j].mean() for j in range(k)])
    # ddof=1: ước lượng không chệch CHO NHÓM ĐÃ CẮT - nhưng nhóm đã bị cắt sẵn
    stds = np.array([x[labels == j].std(ddof=1) for j in range(k)])
    return (*sort_params(weights, means, stds), labels)


def fit_gmm(X, k=2, seed=RNG_SEED):
    gmm = GaussianMixture(n_components=k, covariance_type="full", n_init=10,
                      random_state=seed).fit(X)
    weights, means, stds = sort_params(
        gmm.weights_, gmm.means_.ravel(), np.sqrt(gmm.covariances_.ravel())
    )
    return weights, means, stds, gmm


def avg_log_likelihood(X, weights, means, stds):
    return np.log(mixture_pdf(X.ravel(), weights, means, stds)).mean()


def sample_mixture(weights, means, stds, n, seed):
    rng = np.random.default_rng(seed)
    comp = rng.choice(len(weights), size=n, p=weights)
    return rng.normal(means[comp], stds[comp]).reshape(-1, 1)


def main():
    # ===================================================================
    # Thí nghiệm 1: biết trước sự thật -> đo được sai số của từng cách
    # ===================================================================

    SCENARIOS = {
        "A. 2 cụm cân bằng, chồng lấn vừa": dict(
            weights=[0.50, 0.50], means=[4.5, 7.0], stds=[1.0, 1.0]
        ),
        "B. 2 cụm chồng lấn NHIỀU": dict(
            weights=[0.50, 0.50], means=[5.0, 6.5], stds=[1.2, 1.2]
        ),
        "C. Lệch tỉ lệ + lệch sigma": dict(
            weights=[0.25, 0.75], means=[3.5, 7.5], stds=[0.6, 1.5]
        ),
    }
    N_SAMPLES = 4000


    def report_scenario(name, truth, n=N_SAMPLES, seed=RNG_SEED):
        tw, tm, ts = sort_params(truth["weights"], truth["means"], truth["stds"])
        X = sample_mixture(tw, tm, ts, n, seed)

        kw, kmn, ksd, _ = fit_kmeans_then_moments(X)
        gw, gmn, gsd, _ = fit_gmm(X)

        rows = []
        for j in range(2):
            rows.append(
                {
                    "cụm": j + 1,
                    "pi thật": tw[j], "pi KMeans": kw[j], "pi GMM": gw[j],
                    "mu thật": tm[j], "mu KMeans": kmn[j], "mu GMM": gmn[j],
                    "sigma thật": ts[j], "sigma KMeans": ksd[j], "sigma GMM": gsd[j],
                    "sigma sai KMeans %": 100 * (ksd[j] - ts[j]) / ts[j],
                    "sigma sai GMM %": 100 * (gsd[j] - ts[j]) / ts[j],
                }
            )
        print(f"\n### {name}  (n={n})")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
        print(
            f"  log-likelihood TB  | sự thật: {avg_log_likelihood(X, tw, tm, ts):.4f}"
            f" | KMeans: {avg_log_likelihood(X, kw, kmn, ksd):.4f}"
            f" | GMM: {avg_log_likelihood(X, gw, gmn, gsd):.4f}"
        )
        return X, (tw, tm, ts), (kw, kmn, ksd), (gw, gmn, gsd)


    print("=" * 78)
    print("THÍ NGHIỆM 1 - Dữ liệu mô phỏng: đã biết tham số thật của 2 chuông")
    print("=" * 78)
    scenario_results = {name: report_scenario(name, truth) for name, truth in SCENARIOS.items()}

    # ===================================================================
    # Thí nghiệm 2: sai số có hệ thống hay chỉ do may rủi? -> lặp nhiều seed
    # ===================================================================

    print("\n" + "=" * 78)
    print("THÍ NGHIỆM 2 - Lặp 200 lần (mỗi lần 1 bộ dữ liệu mới): sai số trung bình")
    print("=" * 78)

    truth = SCENARIOS["A. 2 cụm cân bằng, chồng lấn vừa"]
    tw, tm, ts = sort_params(truth["weights"], truth["means"], truth["stds"])
    err_km, err_gmm = [], []
    for seed in range(200):
        X = sample_mixture(tw, tm, ts, 1000, seed + 100)
        _, _, ksd, _ = fit_kmeans_then_moments(X, seed=seed)
        _, _, gsd, _ = fit_gmm(X, seed=seed)
        err_km.append(100 * (ksd - ts) / ts)
        err_gmm.append(100 * (gsd - ts) / ts)
    err_km, err_gmm = np.array(err_km), np.array(err_gmm)
    print(f"Sai số sigma trung bình - KMeans+moment: {err_km.mean(axis=0).round(2)} %")
    print(f"Sai số sigma trung bình - GMM (EM)     : {err_gmm.mean(axis=0).round(2)} %")
    print(f"Số lần KMeans cho sigma NHỎ hơn thật   : {(err_km < 0).mean(axis=0) * 100} %")
    print("-> KMeans lệch âm gần như mọi lần: đây là sai số CÓ HỆ THỐNG, không phải nhiễu.")

    # ===================================================================
    # Thí nghiệm 3: dữ liệu thật BaiHat.csv - mất thông tin "không chắc chắn"
    # ===================================================================

    print("\n" + "=" * 78)
    print("THÍ NGHIỆM 3 - data/BaiHat.csv: gán cứng làm mất xác suất thuộc cụm")
    print("=" * 78)

    df = pd.read_csv(ROOT / "data" / "BaiHat.csv")
    X = df[["diemSoiDong"]].to_numpy()
    kw, kmn, ksd, klabels = fit_kmeans_then_moments(X)
    gw, gmn, gsd, gmm = fit_gmm(X)

    print(pd.DataFrame({
        "tham số": ["pi_1", "pi_2", "mu_1", "mu_2", "sigma_1", "sigma_2"],
        "KMeans+moment": [kw[0], kw[1], kmn[0], kmn[1], ksd[0], ksd[1]],
        "GMM (EM)": [gw[0], gw[1], gmn[0], gmn[1], gsd[0], gsd[1]],
    }).to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    print(f"\nlog-likelihood TB | KMeans: {avg_log_likelihood(X, kw, kmn, ksd):.4f}"
          f" | GMM: {avg_log_likelihood(X, gw, gmn, gsd):.4f}")

    order = np.argsort(gmm.means_.ravel())
    resp = gmm.predict_proba(X)[:, order]          # xác suất thuộc từng cụm
    km_order = np.argsort([X.ravel()[klabels == j].mean() for j in range(2)])
    km_map = {old: new + 1 for new, old in enumerate(km_order)}
    out = pd.DataFrame({
        "tenBaiHat": df["tenBaiHat"],
        "diemSoiDong": df["diemSoiDong"],
        "KMeans (cứng)": [f"cụm {km_map[l]} (100%)" for l in klabels],
        "GMM (mềm)": [f"cụm 1: {p[0]:.0%} | cụm 2: {p[1]:.0%}" for p in resp],
    }).sort_values("diemSoiDong")
    print("\n" + out.to_string(index=False))
    n_nhap_nhang = int(((resp.max(axis=1) < 0.85)).sum())
    print(f"\n{n_nhap_nhang} bài nằm ở vùng nhập nhằng (GMM cho xác suất < 85%),"
          " K-Means vẫn khẳng định 100% thuộc 1 cụm.")

    # ===================================================================
    # Biểu đồ: mật độ tái tạo từ 2 cách, so với mật độ thật
    # ===================================================================

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (name, (Xs, truth_p, km_p, gmm_p)) in zip(axes, scenario_results.items()):
        grid = np.linspace(Xs.min() - 1, Xs.max() + 1, 500)
        ax.hist(Xs.ravel(), bins=60, density=True, color="lightgray", label="dữ liệu")
        ax.plot(grid, mixture_pdf(grid, *truth_p), color="black", lw=2.5, label="mật độ THẬT")
        ax.plot(grid, mixture_pdf(grid, *km_p), color="orange", lw=2, ls="--",
                label="KMeans + mean/std")
        ax.plot(grid, mixture_pdf(grid, *gmm_p), color="purple", lw=2, ls="-.", label="GMM (EM)")
        for mu, sd, w in zip(km_p[1], km_p[2], km_p[0]):
            ax.plot(grid, w * normal_pdf(grid, mu, sd), color="orange", lw=1, alpha=0.5)
        boundary = (km_p[1][0] + km_p[1][1]) / 2
        ax.axvline(boundary, color="red", lw=1, ls=":", label="ranh giới cứng KMeans")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("điểm sôi động")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mật độ xác suất")
    axes[0].legend(fontsize=8)
    fig.suptitle("K-Means cắt dữ liệu tại ranh giới cứng -> chuông hẹp hơn thật, "
                 "mật độ tái tạo lệch khỏi sự thật", fontsize=12)
    fig.tight_layout()
    out_path = ROOT / "kmeans_vs_gmm.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nĐã lưu biểu đồ: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
