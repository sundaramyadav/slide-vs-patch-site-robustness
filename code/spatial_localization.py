#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPATIAL LOCALIZATION OF SITE SIGNAL WITHIN SLIDES
==================================================
The mechanism experiment for the slide-vs-patch robustness gap.

THE ARGUMENT THIS TESTS
-----------------------
We showed slide-level encoders degrade 5.2x less under leave-one-site-out than
patch-level encoders. WHY? Our hypothesis:

    If site signal is concentrated in a SUBSET of patches, then
      * mean pooling averages contaminated patches in with everything else
        and cannot suppress them, whereas
      * a slide-level attention objective can learn to down-weight them.

    If instead site signal is uniformly diffuse across all patches, that
    explanation fails and we must look elsewhere.

THE DECISIVE CASE: conch_v15
----------------------------
conch_v15 is TITAN's patch backbone. In our benchmark:
    conch_v15 + mean pooling  -> AUROC 0.791, rank 29/30   (fragile)
    TITAN (attends over the SAME patches) -> 0.844, rank 5, LOSO-robust
Same patches. Different aggregation. Opposite robustness.
If site signal is spatially concentrated in conch_v15 patches, the mechanism
is demonstrated rather than merely asserted.

WHAT IS MEASURED
----------------
  1. PER-PATCH SITE SCORE
     Patient-grouped CV linear probe predicting tissue source site from each
     PATCH embedding. Score = predicted probability of the patch's TRUE site.
     High score = this patch carries a strong institutional signature.

  2. CONCENTRATION (tests H1)
     Gini coefficient and top-decile share of the score distribution.
     Kömen et al. (arXiv:2411.05489) report that ALMOST ALL patches carry
     recoverable TSS signatures. We test that claim with a continuous score.
       Gini -> 0  = uniformly diffuse  (supports Kömen, refutes our mechanism)
       Gini high  = concentrated       (supports our mechanism)

  3. SPATIAL CLUSTERING (Moran's I)
     Using patch coordinates. Do high-site-signal patches cluster together
     spatially, or are they scattered at random across the tissue?
       I > 0 = clustered   I ~ 0 = random   I < 0 = dispersed

  4. EDGE EFFECT
     Is site signal stronger at the tissue boundary (where folds, tears and
     section-edge artifacts concentrate) than in the tissue interior?

  5. HEATMAPS
     Per-slide spatial maps of the site score. No published work shows these.

INPUTS: patch-level H5 features + coordinates (NOT the mean-pooled cache).
        Reads a SUBSAMPLE, so runtime stays manageable on Drive.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats
from scipy.spatial import cKDTree

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
BASE        = os.environ.get("PCA_ROOT", "./data")
MANIFEST    = f"{BASE}/manifest"
OUT_DIR     = f"{BASE}/results/spatial"
FIG_DIR     = f"{BASE}/results/figures"

INVENTORY   = f"{MANIFEST}/feature_inventory.csv"
MANIFEST_CSV= f"{MANIFEST}/manifest_slides.csv"

# Encoders to analyse. conch_v15 FIRST — it is the decisive case (TITAN's backbone).
ENCODERS = [
    "conch_v15",    # TITAN's backbone: mean-pool rank 29 vs TITAN rank 5  <-- KEY
    "phikon",       # highest site leakage (0.998)
    "hibou_l",      # largest LOSO collapse (0.829 -> 0.635)
    "h0-mini",      # best patch performer under Protocol A
]

MIN_PATIENTS_PER_SITE = 15
SLIDES_PER_SITE       = 8      # subsample: 8 sites x 8 = ~64 slides per encoder
MAX_PATCHES_PER_SLIDE = 400    # subsample patches to bound memory / Drive I/O
N_FOLDS               = 5
SEED                  = 42
MORAN_K               = 8      # neighbours for spatial weights
N_HEATMAP_SLIDES      = 6


def extract_tss(pid):
    p = str(pid).split("-")
    return p[1].upper() if len(p) >= 2 else "UNKNOWN"


import re
TCGA_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


# ════════════════════════════════════════════════════════════════════════════
# LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_patch_bag(feat_path, max_patches=MAX_PATCHES_PER_SLIDE, seed=SEED):
    """
    Return (features NxD, coords Nx2) for one slide.
    TRIDENT usually writes 'coords' alongside 'features' in the same H5; if it
    is absent we fall back to the sibling patches_*.h5 file.
    """
    with h5py.File(feat_path, "r") as f:
        keys = list(f.keys())
        fkey = "features" if "features" in keys else keys[0]
        X = f[fkey][:]
        C = f["coords"][:] if "coords" in keys else None

    if C is None:
        # fall back: patches_<suffix>.h5 in the same coords directory
        d = os.path.dirname(feat_path)
        stem = os.path.basename(feat_path)[:-3]
        for pref in ("features_", "slide_features_"):
            if stem.startswith(pref):
                stem = stem[len(pref):]
                break
        parts = stem.split("_", 1)
        suffix = parts[1] if len(parts) > 1 else ""
        cand = os.path.join(d, f"patches_{suffix}.h5" if suffix else "patches.h5")
        if os.path.exists(cand):
            with h5py.File(cand, "r") as f:
                ck = "coords" if "coords" in f else list(f.keys())[0]
                C = f[ck][:]
    if X.ndim != 2 or C is None or len(C) != len(X):
        return None, None

    if len(X) > max_patches:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_patches, replace=False)
        X, C = X[idx], C[idx]
    return X.astype(np.float32), np.asarray(C, dtype=float)


def build_patch_dataset(encoder, inv, man, usable, verbose=True):
    """
    Assemble a subsampled patch dataset across sites for one encoder.

    SCHEMA NOTE (verified against the real files): feature_inventory.csv carries
        slide_stem, patient_id, encoder, encoder_type, coords_name, risk_group,
        risk_binary, slide_suffix, feat_path, feat_dim, n_patches
    It already has patient_id and feat_path, so we do NOT merge with the manifest —
    doing so duplicates columns into <col>_x / <col>_y and breaks lookups.
    """
    need = ["slide_stem", "patient_id", "encoder", "feat_path"]
    missing = [c for c in need if c not in inv.columns]
    if missing:
        raise KeyError(
            f"feature_inventory.csv is missing {missing}. Found: {list(inv.columns)}")

    sub = inv[inv["encoder"] == encoder].copy()
    sub["tss"] = sub["patient_id"].map(extract_tss)
    sub = sub[sub["tss"].isin(usable)].drop_duplicates("slide_stem")
    if sub.empty:
        return None, None, None

    rng = np.random.default_rng(SEED)
    chosen = []
    for site, g in sub.groupby("tss"):
        g = g.drop_duplicates("slide_stem")
        take = min(SLIDES_PER_SITE, len(g))
        chosen.append(g.iloc[rng.choice(len(g), take, replace=False)])
    chosen = pd.concat(chosen, ignore_index=True)

    Xs, Cs, meta = [], [], []
    for i, r in chosen.iterrows():
        X, C = load_patch_bag(r["feat_path"])
        if X is None:
            continue
        Xs.append(X); Cs.append(C)
        meta.append(pd.DataFrame({
            "slide_stem": r["slide_stem"], "patient_id": r["patient_id"],
            "tss": r["tss"], "x": C[:, 0], "y": C[:, 1],
            "patch_idx": np.arange(len(X)),
        }))
        if verbose and (i + 1) % 10 == 0:
            print(f"    loaded {i+1}/{len(chosen)} slides")

    if not Xs:
        return None, None, None
    return np.vstack(Xs), pd.concat(meta, ignore_index=True), chosen


# ════════════════════════════════════════════════════════════════════════════
# PER-PATCH SITE SCORE
# ════════════════════════════════════════════════════════════════════════════

def per_patch_site_score(X, meta):
    """
    Patient-grouped CV. For each patch, the predicted probability of its TRUE site.
    Patient grouping guarantees the patch's own patient is never in training.
    """
    sites = meta["tss"].values
    uq = sorted(np.unique(sites)); s2i = {s: i for i, s in enumerate(uq)}
    y = np.array([s2i[s] for s in sites])
    groups = meta["patient_id"].values
    score = np.full(len(y), np.nan)

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y, groups=groups):
        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED,
                                 n_jobs=-1)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        P = clf.predict_proba(sc.transform(X[te]))
        cls = {c: j for j, c in enumerate(clf.classes_)}
        for row, gi in enumerate(te):
            j = cls.get(y[gi])
            score[gi] = P[row, j] if j is not None else np.nan
    return score, len(uq)


# ════════════════════════════════════════════════════════════════════════════
# CONCENTRATION  (H1)
# ════════════════════════════════════════════════════════════════════════════

def gini(v):
    v = np.sort(np.asarray(v, dtype=float))
    v = v[~np.isnan(v)]
    if len(v) == 0 or v.sum() == 0:
        return np.nan
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def concentration_stats(scores, chance):
    s = np.asarray(scores)[~np.isnan(scores)]
    if len(s) == 0:
        return {}
    order = np.sort(s)[::-1]
    return {
        "n_patches": int(len(s)),
        "mean": float(s.mean()), "median": float(np.median(s)),
        "chance": float(chance),
        "frac_above_chance": float((s > chance).mean()),
        "frac_above_2x_chance": float((s > 2 * chance).mean()),
        "gini": gini(s),
        "top10pct_share": float(order[:max(1, len(s)//10)].sum() / s.sum()),
        "top25pct_share": float(order[:max(1, len(s)//4)].sum() / s.sum()),
        "p90_over_p50": float(np.percentile(s, 90) / max(np.percentile(s, 50), 1e-9)),
    }


# ════════════════════════════════════════════════════════════════════════════
# SPATIAL CLUSTERING  (Moran's I)
# ════════════════════════════════════════════════════════════════════════════

def morans_i(coords, values, k=MORAN_K):
    """Moran's I with k-nearest-neighbour spatial weights. >0 = clustered."""
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    coords, v = np.asarray(coords)[ok], v[ok]
    n = len(v)
    if n < k + 2:
        return np.nan, np.nan
    z = v - v.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return np.nan, np.nan
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)      # first neighbour is self
    idx = idx[:, 1:]
    num = float(sum(z[i] * z[idx[i]].sum() for i in range(n)))
    W = n * k
    I = (n / W) * (num / denom)
    EI = -1.0 / (n - 1)
    # permutation-free normal approximation for the z-score
    b2 = (z ** 4).sum() / n / ((z ** 2).sum() / n) ** 2
    S0, S1, S2 = W, 2.0 * W, 4.0 * n * k * k
    A = n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2)
    B = b2 * ((n**2 - n) * S1 - 2 * n * S2 + 6 * S0**2)
    VI = (A - B) / ((n - 1) * (n - 2) * (n - 3) * S0**2) - EI**2
    zsc = (I - EI) / np.sqrt(VI) if VI and VI > 0 else np.nan
    return float(I), float(zsc)


def edge_effect(coords, values):
    """
    Is site signal stronger near the tissue boundary?
    'Edge-ness' = normalised distance to the convex hull of patch coordinates,
    approximated by distance to the nearest coordinate-extreme.
    Returns Spearman rho(edgeness, score); positive = stronger at the edge.
    """
    c = np.asarray(coords, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    c, v = c[ok], v[ok]
    if len(v) < 30:
        return np.nan, np.nan
    centre = c.mean(axis=0)
    d = np.linalg.norm(c - centre, axis=1)
    d = d / (d.max() + 1e-9)          # 0 = centre, 1 = periphery
    r, p = stats.spearmanr(d, v)
    return float(r), float(p)


# ════════════════════════════════════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════════════════════════════════════

def fig_heatmaps(meta, encoder, path, n=N_HEATMAP_SLIDES):
    """Spatial maps of the per-patch site score for representative slides."""
    slides = (meta.groupby("slide_stem")["site_score"].mean()
                  .sort_values(ascending=False))
    pick = list(slides.index[:n // 2]) + list(slides.index[-(n - n // 2):])
    cols = 3; rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, pick):
        d = meta[meta.slide_stem == s]
        sc = ax.scatter(d["x"], d["y"], c=d["site_score"], s=9,
                        cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"{d['tss'].iloc[0]} · mean {d['site_score'].mean():.2f}",
                     fontsize=9)
        ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.04)
    for ax in axes[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"Per-patch site score across the slide — {encoder}\n"
                 f"bright = patch strongly identifies its hospital",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_concentration(all_meta, stats_rows, path):
    encs = [r["encoder"] for r in stats_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for enc in encs:
        d = all_meta[all_meta.encoder == enc]["site_score"].dropna()
        axes[0].hist(d, bins=50, alpha=0.45, label=enc, density=True)
    ch = stats_rows[0].get("chance", 0.125)
    axes[0].axvline(ch, color="k", ls="--", lw=1, label=f"chance = {ch:.3f}")
    axes[0].set(xlabel="per-patch site score", ylabel="density",
                title="Is site signal concentrated or diffuse?")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

    # Lorenz curves — visual read of concentration
    for enc in encs:
        d = np.sort(all_meta[all_meta.encoder == enc]["site_score"].dropna().values)
        cum = np.cumsum(d) / d.sum()
        axes[1].plot(np.linspace(0, 1, len(cum)), cum, lw=2, label=enc)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="perfectly uniform")
    axes[1].set(xlabel="cumulative fraction of patches",
                ylabel="cumulative fraction of site signal",
                title="Lorenz curve (bow = concentration)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    print("=" * 76)
    print("SPATIAL LOCALIZATION OF SITE SIGNAL")
    print("=" * 76)

    inv = pd.read_csv(INVENTORY)
    man = pd.read_csv(MANIFEST_CSV)
    man["tss"] = man["patient_id"].map(extract_tss)
    cnt = man.groupby("patient_id")["tss"].first().value_counts()
    usable = sorted(cnt[cnt >= MIN_PATIENTS_PER_SITE].index.tolist())
    chance = 1.0 / len(usable)
    print(f"\nUsable sites: {len(usable)} -> {usable}   (chance = {chance:.4f})")

    rows, all_meta = [], []
    for enc in ENCODERS:
        ck = f"{OUT_DIR}/{enc}_patch_scores.csv"
        print(f"\n{'-'*76}\n  {enc}\n{'-'*76}")

        if os.path.exists(ck):
            meta = pd.read_csv(ck); print("  [cached]")
        else:
            print("  loading patch bags ...")
            X, meta, chosen = build_patch_dataset(enc, inv, man, usable)
            if X is None:
                print(f"  !! no data for {enc}"); continue
            print(f"  X = {X.shape}  ({meta.slide_stem.nunique()} slides, "
                  f"{meta.patient_id.nunique()} patients)")
            print("  computing per-patch site scores ...")
            sc, k = per_patch_site_score(X, meta)
            meta["site_score"] = sc
            meta.to_csv(ck, index=False)
            del X

        meta["encoder"] = enc
        all_meta.append(meta)

        # 1. concentration
        cs = concentration_stats(meta["site_score"].values, chance)
        print(f"\n  CONCENTRATION (H1)")
        print(f"    mean score            : {cs['mean']:.4f}  (chance {chance:.4f})")
        print(f"    fraction above chance : {cs['frac_above_chance']:.4f}")
        print(f"    Gini                  : {cs['gini']:.4f}")
        print(f"    top-10% share         : {cs['top10pct_share']:.4f}")
        print(f"    p90/p50 ratio         : {cs['p90_over_p50']:.3f}")

        # 2. spatial clustering, per slide
        mi, mz, ee, ep = [], [], [], []
        for s, d in meta.groupby("slide_stem"):
            I, z = morans_i(d[["x", "y"]].values, d["site_score"].values)
            r, p = edge_effect(d[["x", "y"]].values, d["site_score"].values)
            if not np.isnan(I): mi.append(I); mz.append(z)
            if not np.isnan(r): ee.append(r); ep.append(p)
        print(f"\n  SPATIAL CLUSTERING (Moran's I, per slide)")
        print(f"    mean I                : {np.mean(mi):+.4f}  "
              f"(0 = random, >0 = clustered)")
        print(f"    median z              : {np.median(mz):+.2f}")
        print(f"    slides with I > 0     : {np.mean(np.array(mi) > 0):.3f}")
        print(f"\n  EDGE EFFECT (rho of periphery vs score)")
        print(f"    mean rho              : {np.mean(ee):+.4f}")
        print(f"    slides p < 0.05       : {np.mean(np.array(ep) < 0.05):.3f}")

        rows.append({"encoder": enc, **cs,
                     "morans_I_mean": float(np.mean(mi)) if mi else np.nan,
                     "morans_z_median": float(np.median(mz)) if mz else np.nan,
                     "frac_slides_clustered": float(np.mean(np.array(mi) > 0)) if mi else np.nan,
                     "edge_rho_mean": float(np.mean(ee)) if ee else np.nan,
                     "edge_frac_sig": float(np.mean(np.array(ep) < 0.05)) if ep else np.nan})

        try:
            fig_heatmaps(meta, enc, f"{FIG_DIR}/spatial_heatmap_{enc}.png")
            print(f"\n  saved spatial_heatmap_{enc}.png")
        except Exception as e:
            print(f"  !! heatmap failed: {e}")

    if not rows:
        print("\nNo encoders processed."); return None

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/spatial_summary.csv", index=False)
    AM = pd.concat(all_meta, ignore_index=True)
    try:
        fig_concentration(AM, rows, f"{FIG_DIR}/spatial_concentration.png")
        print(f"\n  saved spatial_concentration.png")
    except Exception as e:
        print(f"  !! concentration figure failed: {e}")

    # ── verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * 76); print("VERDICT"); print("=" * 76)
    g = df["gini"].mean(); I = df["morans_I_mean"].mean()
    fa = df["frac_above_chance"].mean()
    print(f"\n  mean Gini              : {g:.4f}")
    print(f"  mean Moran's I         : {I:+.4f}")
    print(f"  mean frac above chance : {fa:.4f}")

    print("\n  H1 (concentration):")
    if g > 0.30:
        print("    CONCENTRATED. A minority of patches carries most site signal.")
        print("    -> SUPPORTS the mechanism: mean pooling cannot suppress what")
        print("       attention can down-weight. Strengthens the paper.")
    elif g > 0.15:
        print("    MODERATELY concentrated. Report the distribution honestly;")
        print("    the mechanism is plausible but not clean.")
    else:
        print("    DIFFUSE. Consistent with Komen et al. ('almost all patches').")
        print("    -> Our mechanism hypothesis FAILS. Do not claim it. The")
        print("       slide/patch gap must be explained some other way, e.g. by")
        print("       the training objective rather than patch selection.")

    print("\n  Spatial structure:")
    if I > 0.15:
        print("    CLUSTERED. High-site-signal patches sit together spatially.")
        print("    -> Site signal has spatial structure; heatmaps are publishable")
        print("       and patch pruning becomes a natural intervention.")
    elif I > 0.05:
        print("    Weakly clustered.")
    else:
        print("    Spatially RANDOM. Site signal is not regionally organised.")

    print(f"\n  NOTE: conch_v15 is the decisive encoder — it is TITAN's backbone.")
    if "conch_v15" in df.encoder.values:
        r = df[df.encoder == "conch_v15"].iloc[0]
        print(f"    conch_v15: Gini {r['gini']:.4f}, Moran's I {r['morans_I_mean']:+.4f}")
        print(f"    Mean-pooled conch_v15 ranked 29/30; TITAN, attending over the")
        print(f"    SAME patches, ranked 5 and was LOSO-robust. If these numbers show")
        print(f"    concentration, that contrast becomes a demonstrated mechanism.")

    print(f"\n  Saved -> {OUT_DIR}/spatial_summary.csv")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
    print("=" * 76)
    return df


if __name__ == "__main__":
    df = main()
