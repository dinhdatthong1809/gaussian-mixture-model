"""
GMM cai dat tu dau bang NumPy (chi dung de minh hoa cong thuc trong bao cao).

Thuat toan EM cho Gaussian Mixture Model:
    E-step: gamma[n,k] = pi_k * N(x_n | mu_k, Sigma_k) / sum_j (...)
    M-step: N_k = sum_n gamma[n,k]
            pi_k    = N_k / N
            mu_k    = (1/N_k) * sum_n gamma[n,k] * x_n
            Sigma_k = (1/N_k) * sum_n gamma[n,k] * (x_n - mu_k)(x_n - mu_k)^T

Moi phep tinh mat do deu lam trong khong gian log + log-sum-exp de tranh tran so.
"""
from __future__ import annotations

import numpy as np


def _log_gaussian(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """log N(x | mu, Sigma) cho tung dong cua X. Dung Cholesky thay vi nghich dao."""
    d = X.shape[1]
    L = np.linalg.cholesky(Sigma)                  # Sigma = L L^T
    diff = X - mu                                  # (N, d)
    # giai L z = diff^T  =>  z^T z = (x-mu)^T Sigma^-1 (x-mu)  (khoang cach Mahalanobis)
    z = np.linalg.solve(L, diff.T)                 # (d, N)
    maha = np.sum(z ** 2, axis=0)                  # (N,)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))     # log|Sigma|
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class GaussianMixtureScratch:
    """GMM voi covariance day du (full), toi uu bang EM."""

    def __init__(self, n_components=2, max_iter=200, tol=1e-6, reg_covar=1e-6, random_state=0):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar            # cong vao duong cheo Sigma de tranh suy bien
        self.random_state = random_state
        self.history_: list[float] = []       # log-likelihood qua tung vong lap
        self.snapshots_: list[dict] = []      # tham so sau moi vong (de ve hinh minh hoa)

    # ---------------------------------------------------------------- khoi tao
    def _init_params(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        N, d = X.shape
        # khoi tao kieu k-means++ rut gon: chon tam cach xa nhau
        idx = [rng.integers(N)]
        for _ in range(self.K - 1):
            dist = np.min(
                np.stack([np.sum((X - X[i]) ** 2, axis=1) for i in idx]), axis=0
            )
            idx.append(int(np.argmax(dist)))
        self.means_ = X[idx].astype(float)
        cov = np.cov(X.T) + self.reg_covar * np.eye(d)
        self.covariances_ = np.stack([cov.copy() for _ in range(self.K)])
        self.weights_ = np.full(self.K, 1.0 / self.K)

    # ------------------------------------------------------------------ E-step
    def _e_step(self, X: np.ndarray):
        # log( pi_k * N(x_n | mu_k, Sigma_k) ) cho moi (n, k)
        log_w = np.stack(
            [np.log(self.weights_[k]) + _log_gaussian(X, self.means_[k], self.covariances_[k])
             for k in range(self.K)],
            axis=1,
        )                                          # (N, K)
        log_norm = _logsumexp(log_w, axis=1)       # (N,) = log p(x_n)
        log_gamma = log_w - log_norm[:, None]      # chuan hoa -> log responsibility
        return np.exp(log_gamma), float(np.sum(log_norm))

    # ------------------------------------------------------------------ M-step
    def _m_step(self, X: np.ndarray, gamma: np.ndarray) -> None:
        N, d = X.shape
        Nk = gamma.sum(axis=0) + 1e-12             # so diem "hieu dung" cua cum k
        self.weights_ = Nk / N
        self.means_ = (gamma.T @ X) / Nk[:, None]
        covs = np.empty((self.K, d, d))
        for k in range(self.K):
            diff = X - self.means_[k]              # (N, d)
            covs[k] = (gamma[:, k, None] * diff).T @ diff / Nk[k]
            covs[k].flat[:: d + 1] += self.reg_covar
        self.covariances_ = covs

    # -------------------------------------------------------------------- fit
    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        self._init_params(X)
        prev = -np.inf
        for it in range(self.max_iter):
            gamma, ll = self._e_step(X)            # E-step
            self._m_step(X, gamma)                 # M-step
            self.history_.append(ll)
            self.snapshots_.append(
                dict(iter=it, weights=self.weights_.copy(),
                     means=self.means_.copy(), covariances=self.covariances_.copy(), ll=ll)
            )
            if abs(ll - prev) < self.tol * abs(ll):
                break
            prev = ll
        self.n_iter_ = len(self.history_)
        self.lower_bound_ = self.history_[-1]
        return self

    # ------------------------------------------------------------ du doan / do
    def predict_proba(self, X):
        return self._e_step(np.asarray(X, dtype=float))[0]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)
        log_w = np.stack(
            [np.log(self.weights_[k]) + _log_gaussian(X, self.means_[k], self.covariances_[k])
             for k in range(self.K)], axis=1)
        return _logsumexp(log_w, axis=1)           # log p(x): dung lam diem "binh thuong/bat thuong"

    def n_parameters(self, d: int) -> int:
        # (K-1) trong so + K*d trung binh + K*d(d+1)/2 phan tu covariance
        return (self.K - 1) + self.K * d + self.K * d * (d + 1) // 2

    def bic(self, X) -> float:
        X = np.asarray(X, dtype=float)
        N, d = X.shape
        ll = float(np.sum(self.score_samples(X)))
        return -2.0 * ll + self.n_parameters(d) * np.log(N)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/old_faithful.csv")
    X = df[["duration", "waiting"]].to_numpy()
    Xs = (X - X.mean(0)) / X.std(0)
    g = GaussianMixtureScratch(n_components=2, random_state=0).fit(Xs)
    print(f"so vong lap EM     : {g.n_iter_}")
    print(f"log-likelihood     : {g.lower_bound_:.4f}")
    print(f"trong so pi        : {np.round(g.weights_, 4)}")
    print(f"trung binh (chuan hoa):\n{np.round(g.means_, 4)}")
