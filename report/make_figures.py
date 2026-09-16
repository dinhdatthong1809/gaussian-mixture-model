#!/usr/bin/env python3
"""Sinh toan bo hinh minh hoa cho bao cao: python3 report/make_figures.py"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from plots import PALETTE, FIGDIR, draw_ellipse, _full_cov  # noqa: E402
from gmm_from_scratch import GaussianMixtureScratch          # noqa: E402

os.makedirs(FIGDIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(os.path.join(ROOT, "data", "old_faithful.csv"))
X = df[["duration", "waiting"]].to_numpy(float)
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)


def save(fig, name):
    p = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    print("->", p)


# ---------------------------------------------- H1: vi sao can GMM (vs K-means)
def fig_why_gmm():
    rng = np.random.default_rng(3)
    A = rng.multivariate_normal([0, 0], [[6.0, 5.2], [5.2, 5.0]], 300)
    B = rng.multivariate_normal([5, -4], [[6.0, 5.2], [5.2, 5.0]], 300)
    Z = np.vstack([A, B])
    km = KMeans(2, n_init=10, random_state=0).fit(Z)
    gm = GaussianMixture(2, covariance_type="full", n_init=10, random_state=0).fit(Z)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, lab, title in [(axes[0], km.labels_, "K-means: biên là đường thẳng, cắt ngang cụm"),
                           (axes[1], gm.predict(Z), "GMM: ellipse ôm đúng hình dạng cụm")]:
        for k in (0, 1):
            ax.scatter(*Z[lab == k].T, s=8, color=PALETTE[k], edgecolor="none")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x1")
    for k in (0, 1):
        draw_ellipse(axes[1], gm.means_[k], gm.covariances_[k], PALETTE[k])
    axes[0].scatter(*km.cluster_centers_.T, marker="X", s=140, c="k")
    axes[0].set_ylabel("x2")
    fig.suptitle("Hai cụm Gaussian kéo dài, tương quan mạnh", fontsize=12)
    save(fig, "fig1_vi_sao_gmm.png")


# ------------------------------------------------- H2: hon hop 1 chieu (waiting)
def fig_mixture_1d():
    w = X[:, 1:2]
    g = GaussianMixture(2, random_state=0).fit(w)
    grid = np.linspace(w.min() - 8, w.max() + 8, 600).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(w, bins=30, density=True, color="0.82", edgecolor="w", label="dữ liệu (histogram)")
    total = np.zeros(len(grid))
    for k in range(2):
        pk = g.weights_[k] * np.exp(
            -0.5 * (grid[:, 0] - g.means_[k, 0]) ** 2 / g.covariances_[k, 0, 0]
        ) / np.sqrt(2 * np.pi * g.covariances_[k, 0, 0])
        total += pk
        ax.plot(grid, pk, color=PALETTE[k], lw=2,
                label=rf"$\pi_{k}\,\mathcal{{N}}(x\,|\,{g.means_[k,0]:.1f},\,"
                      rf"{np.sqrt(g.covariances_[k,0,0]):.1f}^2)$, $\pi_{k}$={g.weights_[k]:.2f}")
    ax.plot(grid, total, "k--", lw=2, label="hỗn hợp $p(x)=\\sum_k \\pi_k \\mathcal{N}_k$")
    ax.set_xlabel("waiting – thời gian chờ (phút)")
    ax.set_ylabel("mật độ")
    ax.set_title("Một Gauss không tả nổi dữ liệu 2 đỉnh — tổng hai Gauss thì được")
    ax.legend(fontsize=8)
    save(fig, "fig2_hon_hop_1d.png")


# ------------------------------------------------------- H3: EM qua tung vong lap
def fig_em_steps():
    g = GaussianMixtureScratch(n_components=2, random_state=7, max_iter=60).fit(Xs)
    picks = [0, 1, 3, min(len(g.snapshots_) - 1, 12)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)
    for ax, it in zip(axes, picks):
        snap = g.snapshots_[it]
        gm = GaussianMixtureScratch(2)
        gm.weights_, gm.means_, gm.covariances_, gm.K = (
            snap["weights"], snap["means"], snap["covariances"], 2)
        gamma = gm.predict_proba(Xs)
        c = np.clip(np.outer(gamma[:, 0], matplotlib.colors.to_rgb(PALETTE[0])) +
                    np.outer(gamma[:, 1], matplotlib.colors.to_rgb(PALETTE[1])), 0, 1)
        ax.scatter(Xs[:, 0], Xs[:, 1], c=c, s=12, edgecolor="none")
        for k in range(2):
            draw_ellipse(ax, snap["means"][k], snap["covariances"][k], PALETTE[k])
            ax.plot(*snap["means"][k], "k+", ms=10, mew=2)
        ax.set_title(f"vòng {it + 1}   log-lik = {snap['ll']:.1f}", fontsize=10)
        ax.set_xlabel("duration (chuẩn hoá)")
    axes[0].set_ylabel("waiting (chuẩn hoá)")
    fig.suptitle("EM hội tụ: màu điểm = trách nhiệm $\\gamma_{nk}$, ellipse = $\\mu_k,\\Sigma_k$",
                 fontsize=12)
    save(fig, "fig3_em_tung_vong.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(range(1, len(g.history_) + 1), g.history_, "o-", color=PALETTE[0], ms=4)
    ax.set_xlabel("vòng lặp EM")
    ax.set_ylabel("log-likelihood")
    ax.set_title("Log-likelihood không bao giờ giảm qua mỗi vòng EM")
    save(fig, "fig4_log_likelihood.png")


# --------------------------------------------------------- H5: cac loai covariance
def fig_cov_types():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    for ax, ct in zip(axes, ["spherical", "diag", "tied", "full"]):
        g = GaussianMixture(2, covariance_type=ct, n_init=10, random_state=0).fit(Xs)
        lab = g.predict(Xs)
        for k in range(2):
            ax.scatter(*Xs[lab == k].T, s=10, color=PALETTE[k], edgecolor="none")
            draw_ellipse(ax, g.means_[k], _full_cov(g, k), PALETTE[k])
        n_par = int(g._n_parameters())
        ax.set_title(f"{ct} — {n_par} tham số, BIC={g.bic(Xs):.0f}", fontsize=10)
        ax.set_xlabel("duration (chuẩn hoá)")
    axes[0].set_ylabel("waiting (chuẩn hoá)")
    fig.suptitle("Ràng buộc trên $\\Sigma_k$: càng tự do càng khớp, càng nhiều tham số", fontsize=12)
    save(fig, "fig5_loai_covariance.png")


# ----------------------------------------------------------------- H6: chon K/BIC
def fig_bic():
    ks = range(1, 9)
    bic, aic = [], []
    for k in ks:
        g = GaussianMixture(k, covariance_type="full", n_init=10, random_state=0).fit(Xs)
        bic.append(g.bic(Xs))
        aic.append(g.aic(Xs))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(ks, bic, "o-", color=PALETTE[0], label="BIC")
    ax.plot(ks, aic, "s--", color=PALETTE[1], label="AIC")
    kbest = list(ks)[int(np.argmin(bic))]
    ax.axvline(kbest, color="0.5", ls=":")
    ax.annotate(f"BIC nhỏ nhất tại K={kbest}", (kbest, min(bic)),
                textcoords="offset points", xytext=(14, 24),
                arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_xlabel("số cụm K")
    ax.set_ylabel("giá trị tiêu chí (nhỏ hơn = tốt hơn)")
    ax.set_title("Chọn K bằng BIC / AIC")
    ax.legend()
    save(fig, "fig6_chon_k_bic.png")


# -------------------------------------------- H7: gan mem + duong dong muc p(x)
def fig_soft_and_density():
    g = GaussianMixture(2, covariance_type="full", n_init=10, random_state=0).fit(Xs)
    proba = g.predict_proba(Xs)
    xx, yy = np.meshgrid(np.linspace(*ax_lim(Xs[:, 0]), 300),
                         np.linspace(*ax_lim(Xs[:, 1]), 300))
    Z = -g.score_samples(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)

    c = np.clip(np.outer(proba[:, 0], matplotlib.colors.to_rgb(PALETTE[0])) +
                np.outer(proba[:, 1], matplotlib.colors.to_rgb(PALETTE[1])), 0, 1)
    axes[0].scatter(Xs[:, 0], Xs[:, 1], c=c, s=16, edgecolor="none")
    amb = proba.max(1) < 0.9
    axes[0].scatter(*Xs[amb].T, s=90, facecolor="none", edgecolor="k", lw=1.2,
                    label=f"mơ hồ: max $\\gamma$ < 0.9 ({amb.sum()} điểm)")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].set_title("Gán mềm: màu pha theo $\\gamma_{nk}$", fontsize=11)

    cs = axes[1].contour(xx, yy, Z, levels=np.logspace(0.1, 1.4, 12), cmap="viridis_r")
    axes[1].scatter(Xs[:, 0], Xs[:, 1], s=8, c="0.3", edgecolor="none")
    fig.colorbar(cs, ax=axes[1], label="$-\\log p(x)$ (cao = bất thường)")
    axes[1].set_title("GMM là mô hình sinh: ước lượng cả mật độ $p(x)$", fontsize=11)
    for ax in axes:
        ax.set_xlabel("duration (chuẩn hoá)")
    axes[0].set_ylabel("waiting (chuẩn hoá)")
    save(fig, "fig7_gan_mem_va_mat_do.png")


def ax_lim(v, pad=0.35):
    return v.min() - pad, v.max() + pad


if __name__ == "__main__":
    fig_why_gmm()
    fig_mixture_1d()
    fig_em_steps()
    fig_cov_types()
    fig_bic()
    fig_soft_and_density()
