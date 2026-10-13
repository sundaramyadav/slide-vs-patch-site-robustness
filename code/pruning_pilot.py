#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PATCH PRUNING PILOT — go/no-go for instance-level site mitigation
==================================================================
Tests the one genuinely novel idea remaining from the TSS proposal (H5):

    Can we improve cross-site generalization by DELETING the patches that
    carry the strongest institutional signature, rather than by correcting
    features (ComBat) or images (stain normalization)?

All published mitigation is image-space or feature-space. Instance-level
selection for site-debiasing appears in no paper we could find. Existing patch
selection literature selects for DIAGNOSTIC INFORMATIVENESS or compute
efficiency, never for removing site bias.

⚠ STRONG NEGATIVE PRIOR — READ BEFORE RUNNING
---------------------------------------------
Our spatial analysis found that 89–98% of ALL patches carry above-chance site
signal (Gini 0.034–0.226; top-10% of patches hold only 10–13% of total signal).
If nearly every patch is contaminated, deleting the worst 5% should not help.
This pilot exists to settle that quickly, not to confirm a hope.

THE THREE CONDITIONS (controls are the point)
---------------------------------------------
For each pruning fraction k we drop k% of patches and mean-pool the rest:

    top     drop the k% with the HIGHEST site score   <- the hypothesis
    random  drop a random k%                          <- controls for "fewer patches"
    bottom  drop the k% with the LOWEST site score    <- controls for "any pruning"

Without `random` and `bottom`, an apparent improvement is uninterpretable: it
could just be that changing the patch count changes the representation.
**top must beat BOTH controls, or the hypothesis fails.**

OPTIMISTIC BY DESIGN
--------------------
The patch site-scorer is fitted on ALL sites, including the held-out one. That
is transductive and leaks — deliberately. It gives pruning the BEST possible
chance. If pruning fails under these favourable conditions, it fails under
honest ones too, and we can stop. If it succeeds, a fold-wise inductive rerun
is required before any claim is made.

RUNTIME: ~1-2 h per encoder (one Drive pass over all patch bags). Pilot = 2.

RULE: never deletes any files.
"""

import os
import sys
import json
import time
import warnings
import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

# ════════════════════════════════════════════════════════════════════════════
BASE       = os.environ.get("PCA_ROOT", "/content/drive/MyDrive/ProstateCancer")
MANIFEST   = f"{BASE}/manifest"
SPATIAL    = f"{BASE}/results/spatial"
OUT_DIR    = f"{BASE}/results/pruning"

INVENTORY  = f"{MANIFEST}/feature_inventory.csv"
MAN_CSV    = f"{MANIFEST}/manifest_slides.csv"
LOSO_PATH  = f"{MANIFEST}/folds_loso_frozen.csv"

# Pilot on two encoders: the decisive one and the most fragile one.
ENCODERS = [
    "conch_v15",   # TITAN's backbone; most heterogeneous site signal (Gini 0.226)
    "hibou_l",     # largest LOSO collapse (0.829 -> 0.635) — most room to improve
]

PRUNE_FRACTIONS = [0.0, 0.05, 0.10, 0.25, 0.50]
CONDITIONS      = ["top", "random", "bottom"]

MIN_PATIENTS_PER_SITE = 15
MAX_PATCHES_PER_SLIDE = 600     # cap per slide for memory
SCORER_PATCHES_PER_SLIDE = 150  # subsample used to FIT the site scorer
SEED  = 42
C_REG = 0.01

import re
TCGA_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def extract_tss(pid):
    p = str(pid).split("-")
    return p[1].upper() if len(p) >= 2 else "UNKNOWN"


def load_bag(path, max_patches=MAX_PATCHES_PER_SLIDE, seed=SEED):
    try:
        with h5py.File(path, "r") as f:
            k = "features" if "features" in f else list(f.keys())[0]
            X = f[k][:]
    except Exception:
        return None
    if X.ndim != 2:
        return None
    if len(X) > max_patches:
        X = X[np.random.default_rng(seed).choice(len(X), max_patches, replace=False)]
    return X.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
def build_site_scorer(slides, usable, verbose=True):
    """
    Fit a patch-level site classifier on a subsample across all sites.
    TRANSDUCTIVE by design (see module docstring) — this is the optimistic setting.
    Returns (scaler, clf, class_index_map) or None.
    """
    Xs, sites, pats = [], [], []
    for i, r in slides.iterrows():
        X = load_bag(r["feat_path"], SCORER_PATCHES_PER_SLIDE)
        if X is None:
            continue
        Xs.append(X)
        sites += [r["tss"]] * len(X)
        pats  += [r["patient_id"]] * len(X)
        if verbose and (i + 1) % 50 == 0:
            print(f"      scorer: {i+1}/{len(slides)} slides")
    if not Xs:
        return None
    X = np.vstack(Xs)
    sites = np.array(sites); pats = np.array(pats)
    uq = sorted(np.unique(sites)); s2i = {s: j for j, s in enumerate(uq)}
    y = np.array([s2i[s] for s in sites])

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                             class_weight="balanced", random_state=SEED, n_jobs=-1)
    clf.fit(sc.transform(X), y)

    # sanity: how well does it separate sites?
    skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    tr, te = next(iter(skf.split(X, y, groups=pats)))
    s2 = StandardScaler().fit(X[tr])
    c2 = LogisticRegression(C=1.0, max_iter=800, solver="lbfgs",
                            class_weight="balanced", random_state=SEED, n_jobs=-1)
    c2.fit(s2.transform(X[tr]), y[tr])
    bacc = balanced_accuracy_score(y[te], c2.predict(s2.transform(X[te])))
    if verbose:
        print(f"      scorer patch-level site bacc = {bacc:.4f} "
              f"(chance {1/len(uq):.4f}) on {X.shape[0]} patches")
    return sc, clf, s2i, float(bacc)


def pooled_variants(X, scores, fractions, conditions, rng):
    """
    For one slide: return {(cond,k): pooled_vector}.
    Computed in a single pass so each slide is read from Drive only once.
    """
    out = {}
    n = len(X)
    order_desc = np.argsort(-scores)     # highest site score first
    order_asc  = np.argsort(scores)
    for k in fractions:
        m = int(round(k * n))
        for cond in conditions:
            if m == 0:
                keep = np.arange(n)
            elif cond == "top":
                keep = order_desc[m:]
            elif cond == "bottom":
                keep = order_asc[m:]
            else:
                keep = rng.permutation(n)[m:]
            if len(keep) < 5:
                keep = np.arange(n)
            out[(cond, k)] = X[keep].mean(axis=0)
    return out


def loso_auroc(Z, y, stems, pids, folds_df):
    idx = {s: i for i, s in enumerate(stems)}
    f = folds_df[folds_df["slide_stem"].isin(idx)]
    probs, trues = [], []
    for k in sorted([v for v in f["fold"].unique() if v >= 0]):
        te = [idx[s] for s in f[f.fold == k]["slide_stem"] if s in idx]
        tr = [idx[s] for s in f[f.fold != k]["slide_stem"] if s in idx]
        if not te or not tr:
            continue
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        sc = StandardScaler()
        clf = LogisticRegression(C=C_REG, max_iter=1000, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED)
        clf.fit(sc.fit_transform(Z[tr]), ytr)
        probs.append(clf.predict_proba(sc.transform(Z[te]))[:, 1]); trues.append(yte)
    if not probs:
        return np.nan
    yt, yp = np.concatenate(trues), np.concatenate(probs)
    return float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan


def slide_site_leakage(Z, pids, usable):
    sites = np.array([extract_tss(p) for p in pids])
    keep = np.isin(sites, usable)
    Zs, ps, ss = Z[keep], pids[keep], sites[keep]
    uq = sorted(np.unique(ss)); s2i = {s: i for i, s in enumerate(uq)}
    y = np.array([s2i[s] for s in ss]); chance = 1/len(uq)
    try:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        pred = np.full(len(y), -1)
        for tr, te in skf.split(Zs, y, groups=ps):
            sc = StandardScaler()
            clf = LogisticRegression(C=1.0, max_iter=1200, solver="lbfgs",
                                     class_weight="balanced", random_state=SEED)
            clf.fit(sc.fit_transform(Zs[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(Zs[te]))
        m = pred >= 0
        b = balanced_accuracy_score(y[m], pred[m])
        return float((b - chance) / (1 - chance))
    except Exception:
        return np.nan


# ════════════════════════════════════════════════════════════════════════════
def run_encoder(enc, inv, man, folds_df, usable):
    """
    NOTE ON SCHEMA (verified against the actual files):
        feature_inventory.csv columns:
            slide_stem, patient_id, encoder, encoder_type, coords_name,
            risk_group, risk_binary, slide_suffix, feat_path, feat_dim, n_patches
    The inventory ALREADY contains patient_id, risk_binary and feat_path, so we do
    NOT merge against the manifest. Merging duplicated those columns into
    <col>_x / <col>_y and was the source of two KeyErrors. Use the inventory alone.
    """
    print(f"\n{'='*76}\n  {enc}\n{'='*76}")

    need = ["slide_stem", "patient_id", "encoder", "risk_binary", "feat_path"]
    missing = [c for c in need if c not in inv.columns]
    if missing:
        raise KeyError(
            f"feature_inventory.csv is missing {missing}. Found: {list(inv.columns)}")

    sub = inv[inv["encoder"] == enc].copy()
    sub["tss"] = sub["patient_id"].map(extract_tss)
    sub = (sub[sub["tss"].isin(usable)]
             .drop_duplicates("slide_stem")
             .reset_index(drop=True))
    if sub.empty:
        print(f"  !! no slides for {enc} in usable sites"); return None
    sub["risk_binary"] = sub["risk_binary"].astype(int)
    print(f"  {len(sub)} slides, {sub.patient_id.nunique()} patients, "
          f"labels {sub.risk_binary.value_counts().to_dict()}")

    print("  [1/3] fitting patch-level site scorer (transductive, optimistic) ...")
    built = build_site_scorer(sub, usable)
    if built is None:
        print("  !! scorer failed"); return None
    scaler, clf, s2i, scorer_bacc = built

    print("  [2/3] scoring + pruning every slide (single Drive pass) ...")
    rng = np.random.default_rng(SEED)
    pooled = {(c, k): [] for c in CONDITIONS for k in PRUNE_FRACTIONS}
    stems, ys, pids = [], [], []
    t0 = time.time()
    for i, r in sub.iterrows():
        X = load_bag(r["feat_path"])
        if X is None:
            continue
        P = clf.predict_proba(scaler.transform(X))
        j = s2i.get(r["tss"])
        sc = P[:, j] if j is not None else P.max(axis=1)
        for key, vec in pooled_variants(X, sc, PRUNE_FRACTIONS, CONDITIONS, rng).items():
            pooled[key].append(vec)
        stems.append(r["slide_stem"]); ys.append(int(r["risk_binary"]))
        pids.append(r["patient_id"])
        if (i + 1) % 50 == 0:
            print(f"      {i+1}/{len(sub)} slides  ({time.time()-t0:.0f}s)")

    y = np.array(ys); pids = np.array(pids)
    print(f"  [3/3] LOSO evaluation of {len(pooled)} variants ...")
    rows = []
    for (cond, k), vecs in pooled.items():
        Z = np.vstack(vecs)
        auc  = loso_auroc(Z, y, stems, pids, folds_df)
        leak = slide_site_leakage(Z, pids, usable)
        rows.append({"encoder": enc, "condition": cond, "prune_frac": k,
                     "loso_auroc": auc, "site_leakage": leak,
                     "n_slides": len(stems), "scorer_bacc": scorer_bacc})
        print(f"      {cond:<7} k={k:<5} AUROC={auc:.4f}  leakage={leak:+.4f}")
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 76)
    print("PATCH PRUNING PILOT  (go/no-go for Paper 2)")
    print("=" * 76)
    print("\n  PRIOR: spatial analysis found 89-98% of patches carry site signal.")
    print("  Expectation is that pruning FAILS. This pilot settles it quickly.")

    inv = pd.read_csv(INVENTORY)
    man = pd.read_csv(MAN_CSV)
    man["tss"] = man["patient_id"].map(extract_tss)
    folds_df = pd.read_csv(LOSO_PATH)
    cnt = man.groupby("patient_id")["tss"].first().value_counts()
    usable = sorted(cnt[cnt >= MIN_PATIENTS_PER_SITE].index.tolist())
    print(f"\n  Usable sites: {len(usable)}")

    all_rows = []
    for enc in ENCODERS:
        ck = f"{OUT_DIR}/{enc}_pruning.csv"
        if os.path.exists(ck):
            print(f"\n  [cached] {enc}")
            all_rows.append(pd.read_csv(ck)); continue
        df = run_encoder(enc, inv, man, folds_df, usable)
        if df is not None:
            df.to_csv(ck, index=False); all_rows.append(df)

    if not all_rows:
        print("\nNothing ran."); return None
    D = pd.concat(all_rows, ignore_index=True)
    D.to_csv(f"{OUT_DIR}/pruning_pilot_summary.csv", index=False)

    # ── decision ──────────────────────────────────────────────────────────
    print("\n" + "=" * 76); print("RESULTS"); print("=" * 76)
    for enc, g in D.groupby("encoder"):
        base = g[(g.prune_frac == 0) & (g.condition == "top")]["loso_auroc"].iloc[0]
        print(f"\n  {enc}   baseline (no pruning) AUROC = {base:.4f}")
        print(f"  {'k':<8}{'top':>10}{'random':>10}{'bottom':>10}"
              f"{'top-rand':>10}{'top-base':>10}")
        print("  " + "-" * 58)
        for k in PRUNE_FRACTIONS:
            if k == 0:
                continue
            t = g[(g.prune_frac == k) & (g.condition == "top")]["loso_auroc"].iloc[0]
            r = g[(g.prune_frac == k) & (g.condition == "random")]["loso_auroc"].iloc[0]
            b = g[(g.prune_frac == k) & (g.condition == "bottom")]["loso_auroc"].iloc[0]
            print(f"  {k:<8.2f}{t:>10.4f}{r:>10.4f}{b:>10.4f}"
                  f"{t-r:>+10.4f}{t-base:>+10.4f}")

    print("\n" + "=" * 76); print("DECISION"); print("=" * 76)
    MARGIN = 0.01
    verdicts = []
    for enc, g in D.groupby("encoder"):
        base = g[(g.prune_frac == 0) & (g.condition == "top")]["loso_auroc"].iloc[0]
        gains = []
        for k in [f for f in PRUNE_FRACTIONS if f > 0]:
            t = g[(g.prune_frac == k) & (g.condition == "top")]["loso_auroc"].iloc[0]
            r = g[(g.prune_frac == k) & (g.condition == "random")]["loso_auroc"].iloc[0]
            b = g[(g.prune_frac == k) & (g.condition == "bottom")]["loso_auroc"].iloc[0]
            gains.append({"k": k, "vs_base": t - base,
                          "vs_random": t - r, "vs_bottom": t - b})
        gd = pd.DataFrame(gains)
        # The hypothesis requires top-pruning to beat ALL THREE comparisons at the
        # SAME k. Selecting k by vs_base alone would let through a case where
        # top-pruning gains but random-pruning gains just as much — precisely what
        # the controls exist to detect. So score each k by its WEAKEST gain.
        gd["min_gain"] = gd[["vs_base", "vs_random", "vs_bottom"]].min(axis=1)
        best = gd.loc[gd.min_gain.idxmax()]
        ok = bool(best.min_gain > MARGIN)
        verdicts.append(ok)
        print(f"\n  {enc}: best k={best.k:.2f}  "
              f"vs baseline {best.vs_base:+.4f} | vs random {best.vs_random:+.4f} "
              f"| vs bottom {best.vs_bottom:+.4f}")
        print(f"      weakest gain = {best.min_gain:+.4f} "
              f"(must exceed {MARGIN:+.2f})  -> {'PASS' if ok else 'FAIL'}")

    print("\n" + "-" * 76)
    if all(verdicts) and verdicts:
        print("  >>> GO <<<")
        print("  Top-site-score pruning beats BOTH controls on every encoder tested.")
        print("  NEXT: rerun INDUCTIVELY (fit the scorer on training sites only,")
        print("        per fold) before making any claim. The current setting leaks.")
    elif any(verdicts):
        print("  >>> MIXED <<<")
        print("  Works on some encoders, not others. Not a headline result.")
        print("  Report as a secondary observation, or investigate why it differs.")
    else:
        print("  >>> NO-GO <<<")
        print("  Pruning does not beat its controls, even in the optimistic")
        print("  transductive setting. This is consistent with the spatial finding")
        print("  that 89-98% of patches carry site signal — there is no small")
        print("  contaminated subset to remove.")
        print("  -> Do NOT build Paper 2 on patch pruning. Finalize Paper 1")
        print("     (slide-vs-patch gap + spatial structure) and preprint it.")
    print("-" * 76)
    print(f"\n  Saved -> {OUT_DIR}/pruning_pilot_summary.csv")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    return D


if __name__ == "__main__":
    D = main()
