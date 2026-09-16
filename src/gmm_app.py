#!/usr/bin/env python3
"""
Ung dung GMM: train - danh gia - du doan - phat hien bat thuong.

Dataset mac dinh: Old Faithful geyser (272 lan phun, 2 dac trung):
    duration : thoi gian phun (phut)
    waiting  : thoi gian cho den lan phun tiep theo (phut)
    kind     : nhan that ("short"/"long") - CHI dung de danh gia, khong dua vao train.

Vi du:
    python3 src/gmm_app.py train   --k auto --plot
    python3 src/gmm_app.py predict --input "4.1,82"
    python3 src/gmm_app.py predict --csv data/new_samples.csv
    python3 src/gmm_app.py eval
    python3 src/gmm_app.py outliers --top 5
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "old_faithful.csv")
MODEL = os.path.join(ROOT, "models", "gmm.joblib")
FEATURES = ["duration", "waiting"]


# --------------------------------------------------------------------- du lieu
def load_data(path=DATA):
    df = pd.read_csv(path)
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["kind"].to_numpy() if "kind" in df.columns else None
    return df, X, y


def build_model(k: int, cov_type: str, seed: int) -> Pipeline:
    """Chuan hoa dac trung roi moi fit GMM: 2 truc cua Old Faithful lech thang do rat manh."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gmm", GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            n_init=10,              # chay lai 10 lan, giu nghiem co log-likelihood cao nhat
            init_params="k-means++",
            reg_covar=1e-6,         # chong suy bien covariance
            max_iter=500,
            random_state=seed,
        )),
    ])


def select_k(X, k_max: int, cov_type: str, seed: int):
    """Chon so cum bang BIC (cang nho cang tot)."""
    rows = []
    for k in range(1, k_max + 1):
        m = build_model(k, cov_type, seed).fit(X)
        Xs = m["scaler"].transform(X)
        rows.append(dict(k=k, bic=m["gmm"].bic(Xs), aic=m["gmm"].aic(Xs),
                         loglik=m["gmm"].score(Xs) * len(Xs)))
    tab = pd.DataFrame(rows)
    return int(tab.loc[tab.bic.idxmin(), "k"]), tab


# ----------------------------------------------------------------------- train
def cmd_train(args):
    df, X, y = load_data(args.data)
    if args.k == "auto":
        k, tab = select_k(X, args.k_max, args.cov, args.seed)
        print("Chon so cum bang BIC:")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:10.2f}"))
        print(f"-> K toi uu = {k}\n")
    else:
        k, tab = int(args.k), None

    model = build_model(k, args.cov, args.seed).fit(X)
    gmm: GaussianMixture = model["gmm"]
    Xs = model["scaler"].transform(X)

    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "k": k}, MODEL)

    print(f"Da train GMM: K={k}, covariance_type={args.cov}, hoi tu sau {gmm.n_iter_} vong EM")
    print(f"log-likelihood = {gmm.score(Xs) * len(Xs):.2f} | BIC = {gmm.bic(Xs):.2f}")
    # doi tham so ve don vi goc de doc cho de
    scaler = model["scaler"]
    means = scaler.inverse_transform(gmm.means_)
    for i in range(k):
        print(f"  cum {i}: pi={gmm.weights_[i]:.3f}  "
              f"duration~{means[i, 0]:.2f} phut, waiting~{means[i, 1]:.1f} phut")
    print(f"\nModel luu tai: {MODEL}")

    if y is not None:
        _report_vs_labels(model.predict(X), y)
    if args.plot:
        from plots import plot_model_fit
        out = plot_model_fit(model, X, y)
        print(f"Bieu do: {out}")


def _report_vs_labels(pred, y):
    from sklearn.metrics import adjusted_rand_score
    tab = pd.crosstab(pd.Series(pred, name="cum du doan"), pd.Series(y, name="nhan that"))
    print("\nDoi chieu voi nhan that (khong dung khi train):")
    print(tab.to_string())
    print(f"Adjusted Rand Index = {adjusted_rand_score(y, pred):.4f}")


# --------------------------------------------------------------------- predict
def _load_model():
    if not os.path.exists(MODEL):
        raise SystemExit("Chua co model. Chay: python3 src/gmm_app.py train --k auto")
    return joblib.load(MODEL)


def cmd_predict(args):
    bundle = _load_model()
    model = bundle["model"]
    if args.csv:
        Xnew = pd.read_csv(args.csv)[FEATURES].to_numpy(dtype=float)
    else:
        Xnew = np.array([[float(v) for v in row.split(",")] for row in args.input])

    labels = model.predict(Xnew)
    proba = model.predict_proba(Xnew)               # xac suat hau nghiem = soft assignment
    logp = model["gmm"].score_samples(model["scaler"].transform(Xnew))

    out = pd.DataFrame(Xnew, columns=FEATURES)
    out["cum"] = labels
    for k in range(proba.shape[1]):
        out[f"P(cum {k})"] = proba[:, k].round(4)
    out["log p(x)"] = logp.round(3)
    print(out.to_string(index=False))
    if args.json:
        print(json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2))


# ------------------------------------------------------------------- eval / do
def cmd_eval(args):
    bundle = _load_model()
    model = bundle["model"]
    df, X, y = load_data(args.data)
    pred = model.predict(X)
    Xs = model["scaler"].transform(X)
    gmm = model["gmm"]
    print(f"K = {bundle['k']} | log-likelihood = {gmm.score(Xs) * len(Xs):.2f} | "
          f"BIC = {gmm.bic(Xs):.2f} | AIC = {gmm.aic(Xs):.2f}")
    conf = model.predict_proba(X).max(axis=1)
    print(f"Do tin cay trung binh (max posterior) = {conf.mean():.4f}; "
          f"so diem mo ho (<0.9) = {(conf < 0.9).sum()}/{len(X)}")
    if y is not None:
        _report_vs_labels(pred, y)


def cmd_outliers(args):
    """log p(x) thap = diem it kha nang sinh ra tu mo hinh -> ung vien bat thuong."""
    bundle = _load_model()
    model = bundle["model"]
    df, X, _ = load_data(args.data)
    logp = model["gmm"].score_samples(model["scaler"].transform(X))
    df = df.assign(**{"log p(x)": logp.round(3)}).sort_values("log p(x)")
    print(f"{args.top} diem bat thuong nhat:")
    print(df.head(args.top).to_string(index=False))


# ------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="GMM tren du lieu Old Faithful geyser")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="huan luyen mo hinh")
    t.add_argument("--data", default=DATA)
    t.add_argument("--k", default="auto", help="'auto' (chon bang BIC) hoac mot so nguyen")
    t.add_argument("--k-max", type=int, default=8)
    t.add_argument("--cov", default="full",
                   choices=["full", "tied", "diag", "spherical"])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--plot", action="store_true", help="ve bieu do ket qua")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="du doan cum cho diem moi")
    pr.add_argument("--input", nargs="*", default=[], metavar="duration,waiting")
    pr.add_argument("--csv", help="file csv co cot duration,waiting")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_predict)

    e = sub.add_parser("eval", help="danh gia mo hinh da luu")
    e.add_argument("--data", default=DATA)
    e.set_defaults(func=cmd_eval)

    o = sub.add_parser("outliers", help="liet ke diem co log p(x) thap nhat")
    o.add_argument("--data", default=DATA)
    o.add_argument("--top", type=int, default=5)
    o.set_defaults(func=cmd_outliers)

    args = p.parse_args()
    if getattr(args, "cmd", None) == "predict" and not args.input and not args.csv:
        args.input = ["4.1,82", "1.9,54", "3.0,68"]     # vi du mac dinh
    args.func(args)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
