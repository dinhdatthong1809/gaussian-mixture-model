"""Vì sao mỗi vòng lặp EM không bao giờ làm giảm log-likelihood?

Với mọi phân phối q(z) trên biến ẩn, log-likelihood tách được thành:

    L(θ) = F(q, θ) + KL( q(z) ‖ p(z | x, θ) )

trong đó  F(q, θ) = E_q[ log p(x, z | θ) ] − E_q[ log q(z) ]  là ELBO.
Vì KL ≥ 0 nên ELBO luôn là cận dưới: L(θ) ≥ F(q, θ).

    Bước E: chọn q = p(z | x, θ) -> KL = 0 -> F(q, θ) = L(θ)   (cận dưới chạm đúng)
    Bước M: cực đại F theo θ     -> F(q, θ_mới) ≥ F(q, θ_cũ)

Ghép lại:  L(θ_mới) ≥ F(q, θ_mới) ≥ F(q, θ_cũ) = L(θ_cũ).

Script tự cài đặt EM (không dùng sklearn) để ghi lại ELBO và log-likelihood
sau TỪNG NỬA BƯỚC, rồi vẽ đường bậc thang minh hoạ bất đẳng thức trên.

Chạy: python3 src/elbo_em.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
K = 2
N_ITER = 12
REG = 1e-6
rng = np.random.default_rng(0)


def log_normal(x, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (x[:, None] - mu) ** 2 / var)


def log_joint(x, w, mu, var):
    """log[ pi_k * N(x_i | mu_k, var_k) ] cho mọi (i, k)."""
    return np.log(w) + log_normal(x, mu, var)


def log_likelihood(x, w, mu, var):
    """L(θ) trung bình trên mỗi điểm."""
    lj = log_joint(x, w, mu, var)
    m = lj.max(axis=1, keepdims=True)
    return float(np.mean(m.ravel() + np.log(np.exp(lj - m).sum(axis=1))))


def elbo(x, q, w, mu, var):
    """F(q, θ) trung bình trên mỗi điểm; 0·log0 được coi là 0."""
    lj = log_joint(x, w, mu, var)
    mask = q > 1e-12
    return float(np.sum(q[mask] * (lj[mask] - np.log(q[mask]))) / len(x))


def e_step(x, w, mu, var):
    """q_ik = p(z_i = k | x_i, θ) — xác suất hậu nghiệm."""
    lj = log_joint(x, w, mu, var)
    m = lj.max(axis=1, keepdims=True)
    p = np.exp(lj - m)
    return p / p.sum(axis=1, keepdims=True)


def m_step(x, q):
    """Cực đại ELBO theo θ: nghiệm đóng cho pi, mu, sigma^2."""
    nk = q.sum(axis=0)
    w = nk / len(x)
    mu = (q * x[:, None]).sum(axis=0) / nk
    var = (q * (x[:, None] - mu) ** 2).sum(axis=0) / nk + REG
    return w, mu, var


df = pd.read_csv(ROOT / "data" / "BaiHat.csv")
x = df["diemSoiDong"].to_numpy(dtype=float)

# Khởi tạo: 2 tâm lấy ngẫu nhiên từ dữ liệu, phương sai chung
w = np.full(K, 1 / K)
mu = rng.choice(x, size=K, replace=False)
var = np.full(K, x.var(ddof=1))

history = []   # (nhãn, ELBO, log-likelihood)
q = e_step(x, w, mu, var)
history.append(("E0", elbo(x, q, w, mu, var), log_likelihood(x, w, mu, var)))

for it in range(1, N_ITER + 1):
    w, mu, var = m_step(x, q)                  # bước M: θ mới, q giữ nguyên
    history.append((f"M{it}", elbo(x, q, w, mu, var), log_likelihood(x, w, mu, var)))
    q = e_step(x, w, mu, var)                  # bước E: q mới, θ giữ nguyên
    history.append((f"E{it}", elbo(x, q, w, mu, var), log_likelihood(x, w, mu, var)))

labels = [h[0] for h in history]
elbos = np.array([h[1] for h in history])
lls = np.array([h[2] for h in history])
gaps = lls - elbos

print(f"{'bước':>5} {'ELBO F(q,θ)':>13} {'log-lik L(θ)':>14} {'KL = L − F':>12}")
for lab, f, l, g in zip(labels, elbos, lls, gaps):
    print(f"{lab:>5} {f:13.6f} {l:14.6f} {g:12.6f}")

assert np.all(np.diff(elbos) >= -1e-10), "ELBO phải không giảm"
assert np.all(np.diff(lls) >= -1e-10), "log-likelihood phải không giảm"
assert np.all(gaps >= -1e-10), "L(θ) luôn ≥ ELBO"
assert np.all(np.abs(gaps[::2]) < 1e-10), "sau mỗi bước E, KL phải bằng 0"
print("\nKiểm tra: ELBO không giảm, L(θ) không giảm, L ≥ F, và sau bước E thì L = F.")

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
steps = np.arange(len(history))
is_e = np.array([lab.startswith("E") for lab in labels])

ax.vlines(steps, elbos, lls, color="#c9b8d8", lw=1.5, zorder=1)
ax.plot(steps, lls, color="#0d7c86", lw=2, ls="--", label="log-likelihood L(θ)", zorder=2)
ax.plot(steps, elbos, color="#7c3aad", lw=2.5, label="ELBO F(q, θ)", zorder=3)
ax.scatter(steps[is_e], elbos[is_e], s=55, color="#7c3aad", zorder=4,
           label="sau bước E: KL = 0 nên F chạm đúng L")
ax.scatter(steps[~is_e], elbos[~is_e], s=55, marker="s", color="#d64550", zorder=4,
           label="sau bước M: F tăng, kéo L lên theo")
ax.set_ylabel("giá trị trung bình trên mỗi điểm")
ax.set_title("EM trên điểm sôi động: mỗi nửa bước đều đẩy cận dưới ELBO lên cao hơn")
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

ax2.bar(steps, gaps, color=np.where(is_e, "#7c3aad", "#d64550"), alpha=0.75)
ax2.set_ylabel("KL(q ‖ hậu nghiệm)")
ax2.set_xlabel("nửa bước của EM")
ax2.set_xticks(steps[::2])
ax2.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], fontsize=8)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out_path = ROOT / "elbo_em.png"
fig.savefig(out_path, dpi=150)
print(f"Đã lưu biểu đồ: {out_path}")
plt.show()
