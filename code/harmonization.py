#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 5 — Harmonization trade-off: can the batch effect be subtracted?
======================================================================
THE CLOSING ARGUMENT of the paper.

Step 1c (check C4) showed site signal is DISTRIBUTED, not low-rank:
    1 PC -> 0.360 | 5 PCs -> 0.719 | 50 PCs -> 0.893 | full -> 0.998
That predicts a painful trade-off: removing enough site signal to matter should
also remove biological signal. This step measures that curve directly.

THE QUESTION
    For each harmonization strength, plot
        x = residual site leakage (how much hospital identity survives)
        y = LOSO risk-prediction AUROC (how much biology survives)
    If y falls as fast as x, you CANNOT simply subtract the batch effect
    -> the field needs site-aware EVALUATION, not post-hoc correction.

THREE METHODS
    (a) SITE-PC REMOVAL  [INDUCTIVE - honest]
        Fit PCA on training sites only; rank PCs by how strongly they separate
        training sites (ANOVA F); project out the top-k. Sweep k.
        No test-site information used at any point.

    (b) SITE-CENTERING  [TRANSDUCTIVE - flagged]
        Subtract each site's own feature mean. Requires UNLABELED data from the
        target site. Realistic in deployment, but a weaker claim than (a).

    (c) SITE-STANDARDIZATION / ComBat-lite  [TRANSDUCTIVE - flagged]
        Per-site mean AND variance alignment, then restore grand mean/scale.

    The inductive/transductive distinction MUST be stated in the paper.
    (b) and (c) assume access to target-hospital data before deployment.

OUTPUTS
    results/harmonization/harmonization_results.csv
    results/harmonization/step5_summary.json
    figures/fig6_harmonization_tradeoff.png     <- the money figure
    figures/fig7_harmonization_by_k.png

RULE: Never delete any files.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy import stats

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
BASE       = os.environ.get("PCA_ROOT", "./data")
MANIFEST   = f"{BASE}/manifest"
CACHE_DIR  = f"{BASE}/cache/embeddings"
HARM_DIR   = f"{BASE}/results/harmonization"
FIG_DIR    = f"{BASE}/results/figures"
LOSO_PATH  = f"{MANIFEST}/folds_loso_frozen.csv"
LOSO_LB    = f"{BASE}/results/loso_leaderboard.csv"

K_SWEEP    = [0, 1, 2, 5, 10, 20, 50]     # 0 = no harmonization (baseline)
SEED       = 42
C_FIXED    = 0.01          # fixed C keeps the sweep fast and comparable
MIN_PATIENTS_PER_SITE = 15

SLIDE_ENCODERS = ["titan", "chief", "madeleine", "feather"]

# Representative subset (fast). Set RUN_ALL=True for the full 30.
RUN_ALL = False
SUBSET = [
    "chief", "titan", "madeleine", "feather",          # all 4 slide encoders
    "phikon", "phikon_v2",                             # highest leakage (0.998)
    "h0-mini", "conch_v1", "ctranspath",               # strong patch performers
    "hibou_l", "virchow", "hoptimus0",                 # biggest LOSO collapses
]

ALL_ENCODERS = SLIDE_ENCODERS + [
    "kaiko-vits8", "kaiko-vits16", "lunit-vits8", "conch_v1", "kaiko-vitb8",
    "kaiko-vitb16", "h0-mini", "phikon", "ctranspath", "conch_v15", "hibou_l",
    "phikon_v2", "gpfm", "kaiko-vitl14", "hoptimus0", "hoptimus1", "openmidnight",
    "midnight12k", "uni_v1", "resnet50", "uni_v2", "musk", "virchow", "virchow2",
    "gigapath", "genbio-pathfm",
]


def extract_tss(pid):
    p = str(pid).split("-")
    return p[1].upper() if len(p) >= 2 else "UNKNOWN"


def load_cache(enc):
    d = np.load(os.path.join(CACHE_DIR, f"{enc}.npz"), allow_pickle=True)
    return (d["X"], np.asarray(d["y"]).astype(int), list(d["stems"]),
            np.asarray(d["patient_ids"]), str(d["encoder_type"]))


# ════════════════════════════════════════════════════════════════════════════
# HARMONIZATION METHODS
# ════════════════════════════════════════════════════════════════════════════

def site_pc_removal_fit(X_tr, sites_tr, k, seed=SEED):
    """
    INDUCTIVE. Fit PCA on training data, rank components by ANOVA F across
    training sites, return a transform that drops the top-k site-discriminative PCs.
    """
    n_comp = min(X_tr.shape[0] - 1, X_tr.shape[1], 200)
    sc = StandardScaler().fit(X_tr)
    p = PCA(n_components=n_comp, random_state=seed).fit(sc.transform(X_tr))
    Z = p.transform(sc.transform(X_tr))

    # rank PCs by site-discriminability (one-way ANOVA F per component)
    uniq = np.unique(sites_tr)
    F = np.zeros(Z.shape[1])
    for j in range(Z.shape[1]):
        groups = [Z[sites_tr == s, j] for s in uniq if (sites_tr == s).sum() > 1]
        if len(groups) >= 2:
            try:
                F[j] = float(stats.f_oneway(*groups)[0])
            except Exception:
                F[j] = 0.0
    F = np.nan_to_num(F, nan=0.0, posinf=0.0)
    drop = set(np.argsort(-F)[:k].tolist()) if k > 0 else set()
    keep = np.array([j for j in range(Z.shape[1]) if j not in drop])

    def transform(X):
        return p.transform(sc.transform(X))[:, keep]

    return transform


def site_centering(X, sites):
    """TRANSDUCTIVE. Subtract each site's own mean."""
    Xh = X.copy().astype(np.float64)
    for s in np.unique(sites):
        m = sites == s
        Xh[m] -= Xh[m].mean(axis=0, keepdims=True)
    return Xh


def site_standardization(X, sites):
    """TRANSDUCTIVE. ComBat-lite: align per-site mean AND variance, restore grand stats."""
    X = X.astype(np.float64)
    gm, gs = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xh = X.copy()
    for s in np.unique(sites):
        m = sites == s
        mu = Xh[m].mean(axis=0, keepdims=True)
        sd = Xh[m].std(axis=0, keepdims=True) + 1e-8
        Xh[m] = (Xh[m] - mu) / sd * gs + gm
    return Xh


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def loso_auroc(X, y, pids, folds_df, stems, transform_fn=None,
               pretransformed=False):
    """LOSO AUROC. transform_fn is fitted per-fold on TRAIN only (inductive)."""
    idx = {s: i for i, s in enumerate(stems)}
    f = folds_df[folds_df["slide_stem"].isin(idx)]
    probs, trues = [], []
    for k in sorted([v for v in f["fold"].unique() if v >= 0]):
        te_s = f[f["fold"] == k]["slide_stem"].tolist()
        tr_s = f[f["fold"] != k]["slide_stem"].tolist()
        te = [idx[s] for s in te_s if s in idx]
        tr = [idx[s] for s in tr_s if s in idx]
        if not te or not tr:
            continue
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        if transform_fn is not None and not pretransformed:
            sites_tr = np.array([extract_tss(p) for p in pids[tr]])
            tf = transform_fn(X[tr], sites_tr)
            Xtr, Xte = tf(X[tr]), tf(X[te])
        else:
            Xtr, Xte = X[tr], X[te]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        clf = LogisticRegression(C=C_FIXED, max_iter=1000, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, ytr)
        probs.append(clf.predict_proba(Xte)[:, 1]); trues.append(yte)
    if not probs:
        return np.nan
    yt = np.concatenate(trues); yp = np.concatenate(probs)
    return float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan


def residual_site_leakage(X, pids, usable):
    """Balanced accuracy for predicting site from (harmonized) embeddings."""
    sites = np.array([extract_tss(p) for p in pids])
    keep = np.isin(sites, usable)
    Xs, ps, ss = X[keep], pids[keep], sites[keep]
    uq = sorted(np.unique(ss)); s2i = {s: i for i, s in enumerate(uq)}
    ys = np.array([s2i[s] for s in ss])
    chance = 1.0 / len(uq)
    try:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        pred = np.full(len(ys), -1)
        for tr, te in skf.split(Xs, ys, groups=ps):
            sc = StandardScaler()
            clf = LogisticRegression(C=1.0, max_iter=1500, solver="lbfgs",
                                     class_weight="balanced", random_state=SEED)
            clf.fit(sc.fit_transform(Xs[tr]), ys[tr])
            pred[te] = clf.predict(sc.transform(Xs[te]))
        m = pred >= 0
        b = balanced_accuracy_score(ys[m], pred[m])
        return float(b), float((b - chance) / (1 - chance)), float(chance)
    except Exception:
        return np.nan, np.nan, chance


# ════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════

def fig_tradeoff(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, tp, ttl in [(axes[0], "slide", "Slide encoders"),
                        (axes[1], "patch", "Patch encoders")]:
        sub = df[(df.type == tp) & (df.method == "pc_removal")]
        for enc, g in sub.groupby("encoder"):
            g = g.sort_values("k")
            ax.plot(g["site_leakage"], g["loso_auroc"], "-o", ms=5, lw=1.6,
                    alpha=.85, label=enc)
            for _, r in g.iterrows():
                if r["k"] in (0, 50):
                    ax.annotate(f"k={int(r['k'])}",
                                (r["site_leakage"], r["loso_auroc"]),
                                textcoords="offset points", xytext=(5, 4), fontsize=7)
        ax.set_xlabel("Residual site leakage (normalised)")
        ax.set_ylabel("LOSO AUROC")
        ax.set_title(f"{ttl} — site-PC removal", fontsize=11)
        ax.grid(alpha=.3); ax.legend(fontsize=7, loc="best")
    fig.suptitle("Trade-off: removing hospital signal also removes biology",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_by_k(df, path):
    pc = df[df.method == "pc_removal"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for enc, g in pc.groupby("encoder"):
        g = g.sort_values("k")
        c = "#2563EB" if g["type"].iloc[0] == "slide" else "#9CA3AF"
        axes[0].plot(g["k"], g["loso_auroc"], "-o", ms=4, color=c, alpha=.8, label=enc)
        axes[1].plot(g["k"], g["site_leakage"], "-o", ms=4, color=c, alpha=.8)
    axes[0].set(xlabel="PCs removed (k)", ylabel="LOSO AUROC",
                title="Biological signal vs harmonization strength")
    axes[1].set(xlabel="PCs removed (k)", ylabel="Residual site leakage",
                title="Hospital signal vs harmonization strength")
    for a in axes:
        a.grid(alpha=.3); a.set_xscale("symlog")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Blue = slide encoders · Grey = patch encoders",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(HARM_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 74); print("STEP 5 — HARMONIZATION TRADE-OFF"); print("=" * 74)

    folds_df = pd.read_csv(LOSO_PATH)
    man = pd.read_csv(f"{MANIFEST}/manifest_slides.csv")
    man["tss"] = man["patient_id"].map(extract_tss)
    cnt = man.groupby("patient_id")["tss"].first().value_counts()
    usable = sorted(cnt[cnt >= MIN_PATIENTS_PER_SITE].index.tolist())
    print(f"\nUsable sites: {len(usable)} -> {usable}")

    encoders = ALL_ENCODERS if RUN_ALL else SUBSET
    encoders = [e for e in encoders if os.path.exists(f"{CACHE_DIR}/{e}.npz")]
    print(f"Encoders: {len(encoders)}  (RUN_ALL={RUN_ALL})")

    rows = []
    for ei, enc in enumerate(encoders, 1):
        X, y, stems, pids, etype = load_cache(enc)
        print(f"\n[{ei}/{len(encoders)}] {enc} [{etype}] X={X.shape}")

        # ── (a) site-PC removal, inductive, k sweep ──────────────────────
        for k in K_SWEEP:
            tfn = None if k == 0 else (lambda Xtr, st, kk=k:
                                       site_pc_removal_fit(Xtr, st, kk))
            auc = loso_auroc(X, y, pids, folds_df, stems, transform_fn=tfn)
            # residual leakage: apply the same removal globally (upper bound)
            if k == 0:
                Xh = X
            else:
                sites_all = np.array([extract_tss(p) for p in pids])
                tf = site_pc_removal_fit(X, sites_all, k)
                Xh = tf(X)
            b, norm, ch = residual_site_leakage(Xh, pids, usable)
            rows.append({"encoder": enc, "type": etype, "method": "pc_removal",
                         "k": k, "loso_auroc": auc, "site_bacc": b,
                         "site_leakage": norm, "chance": ch,
                         "inductive": True})
            print(f"   pc_removal k={k:<3} AUROC={auc:.4f}  leakage={norm:.4f}")

        # ── (b) site-centering, transductive ─────────────────────────────
        sites_all = np.array([extract_tss(p) for p in pids])
        Xc = site_centering(X, sites_all)
        auc_c = loso_auroc(Xc, y, pids, folds_df, stems, pretransformed=True)
        b, norm, ch = residual_site_leakage(Xc, pids, usable)
        rows.append({"encoder": enc, "type": etype, "method": "site_centering",
                     "k": np.nan, "loso_auroc": auc_c, "site_bacc": b,
                     "site_leakage": norm, "chance": ch, "inductive": False})
        print(f"   site_centering    AUROC={auc_c:.4f}  leakage={norm:.4f}  [transductive]")

        # ── (c) site standardization (ComBat-lite), transductive ─────────
        Xs = site_standardization(X, sites_all)
        auc_s = loso_auroc(Xs, y, pids, folds_df, stems, pretransformed=True)
        b, norm, ch = residual_site_leakage(Xs, pids, usable)
        rows.append({"encoder": enc, "type": etype, "method": "site_standardization",
                     "k": np.nan, "loso_auroc": auc_s, "site_bacc": b,
                     "site_leakage": norm, "chance": ch, "inductive": False})
        print(f"   site_standardize  AUROC={auc_s:.4f}  leakage={norm:.4f}  [transductive]")

    df = pd.DataFrame(rows)
    df.to_csv(f"{HARM_DIR}/harmonization_results.csv", index=False)

    # ── analysis ──────────────────────────────────────────────────────────
    print("\n" + "=" * 74); print("TRADE-OFF ANALYSIS"); print("=" * 74)
    pc = df[df.method == "pc_removal"].dropna(subset=["loso_auroc", "site_leakage"])
    base = pc[pc.k == 0].set_index("encoder")

    print(f"\n{'Encoder':<16}{'Type':<7}{'k=0 AUROC':>10}{'k=50 AUROC':>11}"
          f"{'ΔAUROC':>9}{'k=0 leak':>10}{'k=50 leak':>10}{'Δleak':>8}")
    print("-" * 74)
    summ_rows = []
    for enc, g in pc.groupby("encoder"):
        g = g.sort_values("k")
        if 0 not in g.k.values or 50 not in g.k.values:
            continue
        a0 = g[g.k == 0].loso_auroc.iloc[0]; a50 = g[g.k == 50].loso_auroc.iloc[0]
        l0 = g[g.k == 0].site_leakage.iloc[0]; l50 = g[g.k == 50].site_leakage.iloc[0]
        tp = g["type"].iloc[0]
        print(f"{enc:<16}{tp:<7}{a0:>10.4f}{a50:>11.4f}{a50-a0:>+9.4f}"
              f"{l0:>10.4f}{l50:>10.4f}{l50-l0:>+8.4f}")
        summ_rows.append({"encoder": enc, "type": tp, "d_auroc": a50-a0,
                          "d_leak": l50-l0})
    sm = pd.DataFrame(summ_rows)

    print("\n" + "-" * 74)
    if len(sm):
        print(f"Mean ΔAUROC (k=0→50)      : {sm.d_auroc.mean():+.4f}")
        print(f"Mean Δleakage (k=0→50)    : {sm.d_leak.mean():+.4f}")
        # efficiency: how much biology lost per unit of site removed
        valid = sm[sm.d_leak < -0.01]
        if len(valid):
            eff = (valid.d_auroc / valid.d_leak).mean()
            print(f"Mean AUROC lost per unit leakage removed : {eff:.3f}")
        print("\nBy encoder type:")
        print(sm.groupby("type")[["d_auroc", "d_leak"]].mean().round(4).to_string())

    # correlation across the whole sweep
    if len(pc) > 5:
        r, p = stats.spearmanr(pc.site_leakage, pc.loso_auroc)
        print(f"\nSpearman ρ(residual leakage, LOSO AUROC) across sweep = "
              f"{r:+.3f}  p={p:.4g}")
        if r > 0.3:
            print("  -> POSITIVE: as site signal is removed, biological signal falls too.")
            print("     THE BATCH EFFECT CANNOT BE CLEANLY SUBTRACTED. This is the")
            print("     closing argument: the field needs site-aware EVALUATION,")
            print("     not post-hoc correction.")
        elif r < -0.3:
            print("  -> NEGATIVE: removing site signal IMPROVES generalization.")
            print("     Harmonization works — a different (also publishable) story.")
        else:
            print("  -> No clear relationship; report the per-encoder curves instead.")

    # transductive methods
    print("\n" + "-" * 74); print("TRANSDUCTIVE METHODS (require target-site data)")
    print("-" * 74)
    for meth in ("site_centering", "site_standardization"):
        m = df[df.method == meth].dropna(subset=["loso_auroc"])
        if len(m) and len(base):
            deltas = [m[m.encoder == e].loso_auroc.iloc[0] - base.loc[e, "loso_auroc"]
                      for e in m.encoder if e in base.index]
            print(f"  {meth:<22} mean ΔAUROC vs baseline = {np.mean(deltas):+.4f}  "
                  f"mean leakage = {m.site_leakage.mean():.4f}")

    # figures
    print("\n" + "-" * 74); print("FIGURES"); print("-" * 74)
    for fn, name in [(fig_tradeoff, "fig6_harmonization_tradeoff.png"),
                     (fig_by_k, "fig7_harmonization_by_k.png")]:
        try:
            fn(df, f"{FIG_DIR}/{name}"); print(f"  saved {name}")
        except Exception as e:
            print(f"  !! {name}: {e}")

    out = {"n_encoders": len(encoders), "run_all": RUN_ALL,
           "mean_d_auroc": float(sm.d_auroc.mean()) if len(sm) else None,
           "mean_d_leak": float(sm.d_leak.mean()) if len(sm) else None,
           "tradeoff_rho": float(r) if len(pc) > 5 else None,
           "tradeoff_p": float(p) if len(pc) > 5 else None}
    with open(f"{HARM_DIR}/step5_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  Saved -> {HARM_DIR}/")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    print("=" * 74)
    print("\nNext: build_leaderboard.py")
    return df


if __name__ == "__main__":
    df = main()
