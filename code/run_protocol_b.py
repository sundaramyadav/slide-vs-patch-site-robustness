#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 3 — Leave-one-site-out (LOSO) re-evaluation of all 30 encoders
====================================================================
THE DECISIVE EXPERIMENT.

Phase 3 ranked 30 encoders using patient-grouped CV — correct for preventing
PATIENT leakage, but the same hospitals appeared in both train and test. Step 1
showed every encoder can identify the hospital (up to bacc 0.998, chance 0.125).

This step re-ranks all 30 under LOSO: each fold holds out an entire hospital.
That is the deployment scenario — a model meeting a site it has never seen.

KEY QUESTION
------------
    Does the leaderboard REORDER?
    Spearman rho between the Phase 3 ranking and the LOSO ranking.
    Low rho  -> published rankings are protocol-dependent  -> the paper.
    High rho -> rankings are robust  -> still a valid measurement-validity result.

BUILT-IN SENSITIVITY ANALYSIS (mandatory, not optional)
-------------------------------------------------------
Fold 7 = site YL is 52 high / 4 low (92.9% prevalence vs 37.7% globally).
When YL is held out the model trains on data where high-risk is rare and tests
where it is overwhelming. That fold will be brutal for every encoder BY DESIGN.
We therefore report EVERY result twice:
    (i)  all 8 folds
    (ii) YL excluded (7 folds)
If conclusions flip between (i) and (ii), YL is driving them and we must say so.

READS THE CACHE built in Step 1 -> fast (minutes, not hours). CPU is fine.
Checkpointed per encoder: safe to re-run after a disconnect.

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

EVAL_ENGINE_DIR = os.environ.get("PCA_ROOT", "./data")
if EVAL_ENGINE_DIR not in sys.path:
    sys.path.insert(0, EVAL_ENGINE_DIR)

from eval_engine import compute_metrics, bootstrap_ci, compute_ece

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
BASE          = os.environ.get("PCA_ROOT", "./data")
MANIFEST_DIR  = f"{BASE}/manifest"
CACHE_DIR     = f"{BASE}/cache/embeddings"
RESULTS_DIR   = f"{BASE}/results/loso"
SITE_DIR      = f"{BASE}/results/site_analysis"

LOSO_PATH      = f"{MANIFEST_DIR}/folds_loso_frozen.csv"
PHASE3_LB      = f"{BASE}/results/phase3_leaderboard.csv"
LEAK_PATH      = f"{SITE_DIR}/site_leakage_scores.csv"
LOSO_LB        = f"{BASE}/results/loso_leaderboard.csv"

EXCLUDE_SITE   = "YL"          # the extreme-prevalence site for sensitivity analysis
C_RANGE        = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
BOOTSTRAP_N    = 1000
SEED           = 42

PATCH_ENCODERS = [
    "kaiko-vits8", "kaiko-vits16", "lunit-vits8", "conch_v1",
    "kaiko-vitb8", "kaiko-vitb16", "h0-mini", "phikon", "ctranspath", "conch_v15",
    "hibou_l", "phikon_v2", "gpfm", "kaiko-vitl14",
    "hoptimus0", "hoptimus1", "openmidnight", "midnight12k", "uni_v1",
    "resnet50", "uni_v2", "musk", "virchow", "virchow2", "gigapath",
    "genbio-pathfm",
]
SLIDE_ENCODERS = ["titan", "chief", "madeleine", "feather"]
ALL_ENCODERS   = PATCH_ENCODERS + SLIDE_ENCODERS


def load_cache(encoder: str):
    d = np.load(os.path.join(CACHE_DIR, f"{encoder}.npz"), allow_pickle=True)
    return (d["X"], np.asarray(d["y"]).astype(int), list(d["stems"]),
            np.asarray(d["patient_ids"]), str(d["encoder_type"]))


def _select_C(Xtr, ytr, groups_tr, C_range):
    """Inner 3-fold patient-grouped CV to choose L2 strength (AUROC criterion)."""
    from sklearn.model_selection import StratifiedGroupKFold
    if len(np.unique(ytr)) < 2:
        return 1.0
    try:
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = {C: [] for C in C_range}
        for ti, vi in inner.split(Xtr, ytr, groups=groups_tr):
            if len(np.unique(ytr[vi])) < 2:
                continue
            for C in C_range:
                clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                         class_weight="balanced", random_state=SEED)
                clf.fit(Xtr[ti], ytr[ti])
                scores[C].append(roc_auc_score(ytr[vi], clf.predict_proba(Xtr[vi])[:, 1]))
        valid = {C: np.mean(v) for C, v in scores.items() if v}
        return float(max(valid, key=valid.get)) if valid else 1.0
    except Exception:
        return 1.0


def run_loso(X, y, stems, patient_ids, folds_df, exclude_site=None):
    """
    LOSO CV. Each fold >= 0 is one held-out site. fold == -1 rows are train-only.
    Returns OOF prediction DataFrame.
    """
    stem_idx = {s: i for i, s in enumerate(stems)}
    f = folds_df[folds_df["slide_stem"].isin(stem_idx)].copy()
    if exclude_site is not None:
        f = f[f["tss"] != exclude_site]

    test_folds = sorted([k for k in f["fold"].unique() if k >= 0])
    recs = []
    for k in test_folds:
        te_stems = f[f["fold"] == k]["slide_stem"].tolist()
        tr_stems = f[f["fold"] != k]["slide_stem"].tolist()   # includes fold == -1
        te = [stem_idx[s] for s in te_stems if s in stem_idx]
        tr = [stem_idx[s] for s in tr_stems if s in stem_idx]
        if not te or not tr:
            continue
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue   # AUROC undefined

        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        C = _select_C(Xtr, ytr, patient_ids[tr], C_RANGE)
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        site_k = f[f["fold"] == k]["tss"].iloc[0]
        fold_auc = roc_auc_score(yte, prob)
        for j, gi in enumerate(te):
            recs.append({"slide_stem": stems[gi], "patient_id": patient_ids[gi],
                         "y_true": int(yte[j]), "y_prob": float(prob[j]),
                         "repeat": 0, "fold": int(k), "tss": site_k,
                         "best_C": C, "fold_auroc": fold_auc})
    return pd.DataFrame(recs)


def evaluate(oof_df, encoder, etype, tag):
    """Metrics + bootstrap CI + ECE from an OOF frame."""
    if oof_df.empty:
        return None
    pool = (oof_df.groupby("slide_stem")
                  .agg(y_true=("y_true", "first"), y_prob=("y_prob", "mean"))
                  .reset_index())
    if pool["y_true"].nunique() < 2:
        return None
    pt = compute_metrics(pool["y_true"].values, pool["y_prob"].values)
    ci, thr = bootstrap_ci(oof_df=oof_df, n_boot=BOOTSTRAP_N, verbose=False)
    ece, _ = compute_ece(pool["y_true"].values, pool["y_prob"].values, n_bins=10)
    per_fold = (oof_df.groupby(["fold", "tss"])["fold_auroc"].first()
                      .reset_index().to_dict("records"))
    return {"encoder": encoder, "encoder_type": etype, "variant": tag,
            "n_slides": int(pool.shape[0]),
            "n_patients": int(oof_df["patient_id"].nunique()),
            "threshold": thr, "ece": ece, "metrics": ci,
            "point_metrics": pt, "per_fold": per_fold}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 74)
    print("STEP 3 — LOSO RE-EVALUATION (all 30 encoders)")
    print("=" * 74)

    folds_df = pd.read_csv(LOSO_PATH)
    print(f"\nLOSO folds: {sorted(folds_df[folds_df.fold>=0].fold.unique())}")
    print(f"Sites: {folds_df[folds_df.fold>=0].groupby('fold')['tss'].first().to_dict()}")

    results = []
    for enc in ALL_ENCODERS:
        cp = os.path.join(CACHE_DIR, f"{enc}.npz")
        if not os.path.exists(cp):
            print(f"  !! cache missing: {enc} — skipping")
            continue
        ck = os.path.join(RESULTS_DIR, f"{enc}_loso.json")
        if os.path.exists(ck):
            with open(ck) as fh:
                results.append(json.load(fh))
            print(f"  [cached] {enc}")
            continue

        X, y, stems, pids, etype = load_cache(enc)
        t1 = time.time()

        oof_all = run_loso(X, y, stems, pids, folds_df, exclude_site=None)
        r_all   = evaluate(oof_all, enc, etype, "all_folds")

        oof_ex  = run_loso(X, y, stems, pids, folds_df, exclude_site=EXCLUDE_SITE)
        r_ex    = evaluate(oof_ex, enc, etype, f"excl_{EXCLUDE_SITE}")

        if r_all is None:
            print(f"  !! {enc}: no valid folds")
            continue
        rec = {"encoder": enc, "encoder_type": etype,
               "all_folds": r_all, f"excl_{EXCLUDE_SITE}": r_ex}
        with open(ck, "w") as fh:
            json.dump(rec, fh, indent=2)
        oof_all.to_csv(os.path.join(RESULTS_DIR, f"{enc}_loso_oof.csv"), index=False)
        results.append(rec)

        a = r_all["metrics"]["auroc"]["mean"]
        e = r_ex["metrics"]["auroc"]["mean"] if r_ex else float("nan")
        print(f"  {enc:16} [{etype:5}] LOSO AUROC={a:.4f} | "
              f"excl-{EXCLUDE_SITE}={e:.4f} | {time.time()-t1:.0f}s")

    # ── build leaderboard ─────────────────────────────────────────────────
    rows = []
    for r in results:
        a = r["all_folds"]; ex = r.get(f"excl_{EXCLUDE_SITE}")
        rows.append({
            "encoder": r["encoder"], "type": r["encoder_type"],
            "loso_auroc": round(a["metrics"]["auroc"]["mean"], 4),
            "loso_lo": round(a["metrics"]["auroc"]["ci_lower"], 4),
            "loso_hi": round(a["metrics"]["auroc"]["ci_upper"], 4),
            "loso_aupr": round(a["metrics"]["aupr"]["mean"], 4),
            "loso_mcc": round(a["metrics"]["mcc"]["mean"], 4),
            "loso_ece": round(a["ece"], 4),
            "loso_auroc_exclYL": round(ex["metrics"]["auroc"]["mean"], 4) if ex else np.nan,
            "n_slides": a["n_slides"],
        })
    lb = pd.DataFrame(rows).sort_values("loso_auroc", ascending=False).reset_index(drop=True)
    lb.insert(0, "loso_rank", range(1, len(lb) + 1))

    # ── merge Phase 3 + leakage ───────────────────────────────────────────
    if os.path.exists(PHASE3_LB):
        p3 = pd.read_csv(PHASE3_LB)[["encoder", "auroc"]].rename(
            columns={"auroc": "phase3_auroc"})
        lb = lb.merge(p3, on="encoder", how="left")
        lb["phase3_rank"] = lb["phase3_auroc"].rank(ascending=False)
        lb["auroc_drop"]  = (lb["phase3_auroc"] - lb["loso_auroc"]).round(4)
        lb["rank_change"] = (lb["phase3_rank"] - lb["loso_rank"]).astype(float)
    if os.path.exists(LEAK_PATH):
        lk = pd.read_csv(LEAK_PATH)[["encoder", "site_leakage_score"]]
        lb = lb.merge(lk, on="encoder", how="left")

    lb.to_csv(LOSO_LB, index=False)

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("LOSO LEADERBOARD")
    print("=" * 74)
    print(f"{'Rk':<4}{'Encoder':<17}{'Type':<7}{'LOSO':>8}{'Phase3':>8}"
          f"{'Drop':>8}{'ΔRank':>7}{'exclYL':>8}")
    print("-" * 74)
    for _, r in lb.iterrows():
        d  = r.get("auroc_drop", np.nan)
        rc = r.get("rank_change", np.nan)
        p3 = r.get("phase3_auroc", np.nan)
        ey = r.get("loso_auroc_exclYL", np.nan)
        print(f"{int(r['loso_rank']):<4}{r['encoder']:<17}{r['type']:<7}"
              f"{r['loso_auroc']:>8.4f}{p3:>8.4f}{d:>8.4f}"
              f"{rc:>+7.0f}{ey:>8.4f}")

    # ── THE key statistic ─────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("KEY STATISTICS")
    print("=" * 74)
    if "phase3_auroc" in lb.columns and lb["phase3_auroc"].notna().sum() > 3:
        sub = lb.dropna(subset=["phase3_auroc"])
        rho, p = stats.spearmanr(sub["phase3_auroc"], sub["loso_auroc"])
        print(f"\n  Spearman rho (Phase3 vs LOSO ranking) = {rho:+.3f}  p={p:.4g}")
        if rho > 0.85:
            print("  -> Rankings are LARGELY STABLE. Reframe as measurement-validity.")
        elif rho > 0.5:
            print("  -> MODERATE reorder. Report which encoders move and why.")
        else:
            print("  -> SUBSTANTIAL REORDER. This is the headline result.")
        print(f"\n  Mean AUROC drop      : {sub['auroc_drop'].mean():+.4f}")
        print(f"  Largest drop         : {sub.loc[sub.auroc_drop.idxmax(),'encoder']} "
              f"({sub['auroc_drop'].max():+.4f})")
        print(f"  Smallest drop        : {sub.loc[sub.auroc_drop.idxmin(),'encoder']} "
              f"({sub['auroc_drop'].min():+.4f})")

        # 4a: slide vs patch drop  (the redesigned headline)
        print("\n  --- 4a: SLIDE vs PATCH LOSO drop ---")
        g = sub.groupby("type")["auroc_drop"].agg(["mean", "std", "count"])
        print(g.round(4).to_string())
        s = sub[sub.type == "slide"]["auroc_drop"]
        pch = sub[sub.type == "patch"]["auroc_drop"]
        if len(s) >= 2 and len(pch) >= 2:
            u, pu = stats.mannwhitneyu(s, pch, alternative="two-sided")
            print(f"  Mann-Whitney U p = {pu:.4g}  "
                  f"({'slide encoders drop LESS' if s.mean() < pch.mean() else 'patch drop less'})")

        # 4c: leakage vs drop (with ceiling caveat)
        if "site_leakage_score" in sub.columns and sub["site_leakage_score"].notna().sum() > 5:
            print("\n  --- 4c: leakage vs LOSO drop (ceiling caveat applies) ---")
            r1, p1 = stats.spearmanr(sub["site_leakage_score"], sub["auroc_drop"])
            print(f"  ALL 30    rho={r1:+.3f} p={p1:.4g}")
            sp = sub[sub.type == "patch"]
            if len(sp) > 5:
                r2, p2 = stats.spearmanr(sp["site_leakage_score"], sp["auroc_drop"])
                print(f"  PATCH-26  rho={r2:+.3f} p={p2:.4g}   "
                      f"(leakage SD={sp['site_leakage_score'].std():.4f} — underpowered)")

        # YL sensitivity
        if lb["loso_auroc_exclYL"].notna().sum() > 3:
            s2 = lb.dropna(subset=["loso_auroc_exclYL"])
            rho2, _ = stats.spearmanr(s2["loso_auroc"], s2["loso_auroc_exclYL"])
            print(f"\n  --- YL sensitivity ---")
            print(f"  rho(all folds vs excl-YL) = {rho2:+.3f}")
            print(f"  mean AUROC all folds  : {s2['loso_auroc'].mean():.4f}")
            print(f"  mean AUROC excl-YL    : {s2['loso_auroc_exclYL'].mean():.4f}")
            if rho2 < 0.85:
                print("  !! Conclusions are SENSITIVE to YL — must be stated prominently.")
            else:
                print("  Conclusions are robust to YL exclusion.")

    print(f"\n  Saved -> {LOSO_LB}")
    print(f"  Per-encoder JSON + OOF -> {RESULTS_DIR}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("=" * 74)
    return lb


if __name__ == "__main__":
    lb = main()
