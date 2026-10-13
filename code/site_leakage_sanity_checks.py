#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1c — SANITY CHECKS for the site-leakage result
====================================================
Step 1b reported site balanced accuracy up to 0.998 (chance = 0.125) with
macro-AUROC = 1.000. That is extreme. Before building a paper on it we must
rule out pipeline bugs and trivial explanations.

FOUR CHECKS
-----------
  C1. PERMUTATION TEST  (the critical one)
      Shuffle site labels WITHIN the patient-grouped structure and re-run.
      Expect leakage -> ~0 (bacc -> ~chance = 0.125).
      If permuted leakage stays high, the CV is leaking and Step 1b is void.

  C2. TRIVIAL BASELINE: n_patches only
      Can the site be predicted from tissue area (patch count) alone?
      If yes, part of the "representational" signal is trivial metadata.

  C3. TRIVIAL BASELINE: embedding NORM only
      Single scalar = L2 norm of the slide embedding. Captures global
      intensity/contrast scale. If this alone predicts site well, the signal
      is dominated by global stain statistics rather than structure.

  C4. LOW-DIMENSIONAL PROBE
      How many principal components are needed to reach high site accuracy?
      If 1-5 PCs suffice, site is a dominant, low-rank direction in the
      embedding space -> directly motivates the Step 5 harmonization
      (projecting out those PCs).

INTERPRETATION GUIDE (printed at the end)
-----------------------------------------
  C1 must PASS (permuted ~ chance) or nothing else is valid.
  C2/C3 high  -> report as a caveat; the effect is partly trivial.
  C2/C3 low   -> the site signal is genuinely representational. Stronger paper.
  C4 small k  -> excellent news: harmonization is likely to work.

RUNTIME: ~2-4 min for a subset of encoders (reads the cache, not Drive H5s).

RULE: Never delete any files.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
BASE          = os.environ.get("PCA_ROOT", "./data")
MANIFEST_DIR  = f"{BASE}/manifest"
CACHE_DIR     = f"{BASE}/cache/embeddings"
SITE_DIR      = f"{BASE}/results/site_analysis"
MANIFEST_PATH = f"{MANIFEST_DIR}/manifest_slides.csv"

MIN_PATIENTS_PER_SITE = 15
N_FOLDS   = 5
SEED      = 42
N_PERM    = 5          # permutation repeats (5 is enough to see collapse)

# Test a representative subset: the two extremes + a mid + the slide encoders.
# (Running all 30 is unnecessary for a sanity check and slow.)
CHECK_ENCODERS = [
    "phikon_v2",      # highest leakage (0.9977)
    "phikon",         # highest leakage (0.9977)
    "h0-mini",        # best patch performer, high leakage (0.9876)
    "conch_v15",      # low-leakage patch (0.8944)
    "chief",          # best overall performer (leakage 0.9041)
    "feather",        # LOWEST leakage of all 30 (0.7038)
]

PCA_KS = [1, 2, 5, 10, 20, 50]


def extract_tss(pid: str) -> str:
    p = str(pid).split("-")
    return p[1].upper() if len(p) >= 2 else "UNKNOWN"


def load_cache(encoder: str):
    d = np.load(os.path.join(CACHE_DIR, f"{encoder}.npz"), allow_pickle=True)
    return (d["X"], d["y"], list(d["stems"]),
            np.asarray(d["patient_ids"]), str(d["encoder_type"]))


def site_bacc(X, patient_ids, y_site, n_folds=N_FOLDS, seed=SEED):
    """Patient-grouped CV balanced accuracy for predicting site from X."""
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred = np.full(len(y_site), -1, dtype=int)
    for tr, te in skf.split(X, y_site, groups=patient_ids):
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                 class_weight="balanced", random_state=seed)
        clf.fit(Xtr, y_site[tr])
        pred[te] = clf.predict(Xte)
    m = pred >= 0
    return float(balanced_accuracy_score(y_site[m], pred[m]))


def main():
    os.makedirs(SITE_DIR, exist_ok=True)
    print("=" * 74)
    print("STEP 1c — SANITY CHECKS on the site-leakage result")
    print("=" * 74)

    man = pd.read_csv(MANIFEST_PATH)
    man["tss"] = man["patient_id"].map(extract_tss)
    pat_site = man.groupby("patient_id")["tss"].first()
    counts   = pat_site.value_counts()
    usable   = sorted(counts[counts >= MIN_PATIENTS_PER_SITE].index.tolist())
    chance   = 1.0 / len(usable)
    print(f"\nUsable sites: {len(usable)} -> {usable}")
    print(f"Chance balanced accuracy: {chance:.4f}")

    # n_patches lookup for C2
    npatch_map = dict(zip(man["slide_stem"], man["n_patches"]))

    rows = []
    for enc in CHECK_ENCODERS:
        cp = os.path.join(CACHE_DIR, f"{enc}.npz")
        if not os.path.exists(cp):
            print(f"\n!! cache missing for {enc} — skipping")
            continue

        X, y, stems, pids, etype = load_cache(enc)
        sites = np.array([extract_tss(p) for p in pids])
        keep  = np.isin(sites, usable)
        X, pids, sites, stems = X[keep], pids[keep], sites[keep], list(np.array(stems)[keep])

        uniq = sorted(np.unique(sites))
        s2i  = {s: i for i, s in enumerate(uniq)}
        y_site = np.array([s2i[s] for s in sites])

        print(f"\n{'─'*74}")
        print(f"  {enc}  [{etype}]  X={X.shape}")
        print(f"{'─'*74}")

        # ── real ──
        real = site_bacc(X, pids, y_site)
        print(f"  REAL site bacc                      : {real:.4f}")

        # ── C1 permutation ──
        perms = []
        rng = np.random.default_rng(SEED)
        for i in range(N_PERM):
            # permute site labels at the PATIENT level (preserve group structure)
            up = np.unique(pids)
            pat_lbl = {p: y_site[pids == p][0] for p in up}
            shuffled = rng.permutation(list(pat_lbl.values()))
            newmap = dict(zip(pat_lbl.keys(), shuffled))
            y_perm = np.array([newmap[p] for p in pids])
            perms.append(site_bacc(X, pids, y_perm, seed=SEED + i))
        perm_mean, perm_max = float(np.mean(perms)), float(np.max(perms))
        c1_pass = perm_mean < chance + 0.10
        print(f"  C1 PERMUTED bacc (mean of {N_PERM})      : {perm_mean:.4f} "
              f"(max {perm_max:.4f})   -> {'PASS' if c1_pass else 'FAIL'}")

        # ── C2 n_patches only ──
        npv = np.array([npatch_map.get(s, np.nan) for s in stems], dtype=float)
        if np.isnan(npv).all():
            c2 = float("nan"); print("  C2 n_patches baseline               : n/a")
        else:
            npv = np.nan_to_num(npv, nan=float(np.nanmedian(npv)))
            c2 = site_bacc(npv.reshape(-1, 1), pids, y_site)
            print(f"  C2 n_patches-only bacc              : {c2:.4f}")

        # ── C3 embedding norm only ──
        nrm = np.linalg.norm(X, axis=1).reshape(-1, 1)
        c3 = site_bacc(nrm, pids, y_site)
        print(f"  C3 embedding-norm-only bacc         : {c3:.4f}")

        # ── C4 PCA probe ──
        print(f"  C4 low-rank probe (site bacc by #PCs):")
        pca_res = {}
        Xs = StandardScaler().fit_transform(X)
        maxk = max([k for k in PCA_KS if k < min(X.shape)])
        p = PCA(n_components=maxk, random_state=SEED).fit(Xs)
        Z = p.transform(Xs)
        for k in [k for k in PCA_KS if k <= maxk]:
            b = site_bacc(Z[:, :k], pids, y_site)
            pca_res[k] = round(b, 4)
            bar = "█" * int(b * 30)
            print(f"       {k:>3} PCs : {b:.4f}  {bar}")

        rows.append({
            "encoder": enc, "type": etype, "real_bacc": round(real, 4),
            "perm_bacc_mean": round(perm_mean, 4),
            "perm_bacc_max": round(perm_max, 4),
            "c1_pass": bool(c1_pass),
            "npatches_bacc": None if np.isnan(c2) else round(c2, 4),
            "norm_bacc": round(c3, 4),
            "chance": round(chance, 4),
            **{f"pca_{k}": v for k, v in pca_res.items()},
        })

    df = pd.DataFrame(rows)
    out = os.path.join(SITE_DIR, "site_leakage_sanity_checks.csv")
    df.to_csv(out, index=False)

    # ── verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)

    all_pass = bool(df["c1_pass"].all()) if len(df) else False
    print(f"\nC1 PERMUTATION: {'ALL PASS' if all_pass else 'FAILURE DETECTED'}")
    if all_pass:
        print("  Permuted site labels collapse to ~chance. The CV is sound and the")
        print("  Step 1b leakage scores are VALID. Proceed.")
    else:
        print("  !! Permuted labels still predict well -> the CV is leaking.")
        print("  !! STOP. Do not build on Step 1b until this is fixed.")

    if len(df):
        mx_np = df["npatches_bacc"].dropna().max() if df["npatches_bacc"].notna().any() else 0
        mx_nm = df["norm_bacc"].max()
        print(f"\nC2 max n_patches-only bacc : {mx_np:.4f}")
        print(f"C3 max norm-only bacc      : {mx_nm:.4f}")
        if max(mx_np or 0, mx_nm) > 0.5:
            print("  -> A trivial scalar predicts site substantially.")
            print("     Report as a caveat; part of the effect is metadata, not structure.")
        else:
            print("  -> Trivial baselines are weak. The site signal is genuinely")
            print("     representational (high-dimensional), which STRENGTHENS the paper.")

        pca_cols = [c for c in df.columns if c.startswith("pca_")]
        if pca_cols:
            print("\nC4 low-rank structure (mean across tested encoders):")
            for c in sorted(pca_cols, key=lambda x: int(x.split("_")[1])):
                print(f"     {c.replace('pca_',''):>3} PCs : {df[c].mean():.4f}")
            print("  -> If few PCs already reach high accuracy, site is a dominant")
            print("     low-rank direction and Step 5 harmonization should work well.")

    print(f"\n  Saved -> {out}")
    print("=" * 74)
    return df


if __name__ == "__main__":
    df = main()
