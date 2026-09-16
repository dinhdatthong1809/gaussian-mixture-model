"""Tien ich ve hinh cho GMM (dung chung cho app va cho bao cao)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "report", "figures")
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#9D755D"]


def draw_ellipse(ax, mean, cov, color, n_std=(1, 2), alpha=0.25, lw=2):
    """Ve duong dong muc cua mot Gaussian 2D: truc chinh = vector rieng cua Sigma."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for s in n_std:
        ax.add_patch(Ellipse(mean, 2 * s * np.sqrt(vals[0]), 2 * s * np.sqrt(vals[1]),
                             angle=angle, facecolor=color, edgecolor=color,
                             alpha=alpha / s, lw=lw))


def scatter_clusters(ax, X, labels=None, proba=None, s=18):
    if proba is not None and proba.shape[1] == 2:
        # mau pha tron theo xac suat hau nghiem -> thay ro "soft assignment"
        c = np.array([matplotlib.colors.to_rgb(PALETTE[0])]) * proba[:, [0]] + \
            np.array([matplotlib.colors.to_rgb(PALETTE[1])]) * proba[:, [1]]
        ax.scatter(X[:, 0], X[:, 1], c=np.clip(c, 0, 1), s=s, edgecolor="none")
    elif labels is not None:
        for k in np.unique(labels):
            m = labels == k
            ax.scatter(X[m, 0], X[m, 1], s=s, color=PALETTE[int(k) % len(PALETTE)],
                       label=f"cụm {k}", edgecolor="none")
    else:
        ax.scatter(X[:, 0], X[:, 1], s=s, color="0.35", edgecolor="none")


def plot_model_fit(pipeline, X, y=None, out=None):
    """Ve du lieu + ellipse cua tung thanh phan, trong don vi goc."""
    os.makedirs(FIGDIR, exist_ok=True)
    out = out or os.path.join(FIGDIR, "model_fit.png")
    gmm, scaler = pipeline["gmm"], pipeline["scaler"]
    proba = pipeline.predict_proba(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter_clusters(ax, X, labels=pipeline.predict(X), proba=proba if proba.shape[1] == 2 else None)

    means = scaler.inverse_transform(gmm.means_)
    S = np.diag(scaler.scale_)
    for k in range(gmm.n_components):
        cov_k = _full_cov(gmm, k)
        draw_ellipse(ax, means[k], S @ cov_k @ S.T, PALETTE[k % len(PALETTE)])
        ax.plot(*means[k], "k+", ms=12, mew=2)
    ax.set_xlabel("duration – thời gian phun (phút)")
    ax.set_ylabel("waiting – thời gian chờ (phút)")
    ax.set_title(f"GMM K={gmm.n_components}: dữ liệu tô màu theo xác suất hậu nghiệm")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _full_cov(gmm, k):
    """Quy moi covariance_type ve ma tran day du (d x d)."""
    d = gmm.means_.shape[1]
    ct, C = gmm.covariance_type, gmm.covariances_
    if ct == "full":
        return C[k]
    if ct == "tied":
        return C
    if ct == "diag":
        return np.diag(C[k])
    return np.eye(d) * C[k]          # spherical
