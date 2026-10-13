#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1 — Embedding cache + site-leakage scores  (Experiment A)
===============================================================
TWO JOBS IN ONE PASS:

  1A. CACHE: read each encoder's features from Drive ONCE, mean-pool patch
      encoders to one vector per slide, save as a compact .npz.
      -> Steps 1B, 3 and 5 then load in seconds instead of hours.
      This is the ONLY expensive step in the whole study.

  1B. SITE-LEAKAGE SCORE: for each encoder, train a multiclass classifier to
      predict the TISSUE SOURCE SITE (hospital) from the slide embedding.
      High accuracy = the encoder strongly encodes hospital identity.

WHY THIS MATTERS
----------------
If an encoder can identify the submitting hospital from morphology-free
batch signal (staining, scanner, cutting protocol), and hospitals differ in
case-mix (ours range from 13% to 93% high-risk), then the encoder can score
well on risk prediction WITHOUT understanding morphology at all.

The site-leakage score computed here is the x-axis of the smoking-gun figure
in Step 4. The y-axis (LOSO performance drop) comes from Step 3.

DESIGN NOTES
------------
* Site prediction uses only USABLE sites (>= MIN_PATIENTS_PER_SITE patients),
  so we are not asking the model to predict classes with 1-2 examples.
* Patient-grouped CV — same leakage discipline as everywhere else.
* Scores are reported RAW (balanced accuracy) and NORMALISED against chance:
      normalised = (bacc - chance) / (1 - chance),  chance = 1/n_sites
  Normalised score is comparable across different site counts.
* Checkpointed: re-run safely after a Colab disconnect.

RUNTIME: cache build ~4-7 hrs on Drive FUSE (run overnight, CPU runtime).
         Site-leakage scoring afterwards: ~2 min for all 30.

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

from eval_engine import load_encoder_data

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
BASE           = os.environ.get("PCA_ROOT", "./data")
MANIFEST_DIR   = f"{BASE}/manifest"
CACHE_DIR      = f"{BASE}/cache/embeddings"          # created if missing
SITE_DIR       = f"{BASE}/results/site_analysis"

MANIFEST_PATH  = f"{MANIFEST_DIR}/manifest_slides.csv"
INVENTORY_PATH = f"{MANIFEST_DIR}/feature_inventory.csv"
FOLDS_PATH     = f"{MANIFEST_DIR}/folds_frozen.csv"

MIN_PATIENTS_PER_SITE = 15    # must match Step 0
N_SITE_FOLDS          = 5
SEED                  = 42
C_RANGE_SITE          = [0.01, 0.1, 1.0, 10.0]

# All 30 encoders (26 patch + 4 slide)
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


def extract_tss(patient_id: str) -> str:
    parts = str(patient_id).split("-")
    return parts[1].upper() if len(parts) >= 2 else "UNKNOWN"


# ════════════════════════════════════════════════════════════════════════════
# 1A — CACHE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def cache_path(encoder: str) -> str:
    return os.path.join(CACHE_DIR, f"{encoder}.npz")


def build_cache_for(encoder: str, etype: str) -> bool:
    """Load one encoder from Drive, mean-pool if patch-level, save .npz. Idempotent."""
    cp = cache_path(encoder)
    if os.path.exists(cp):
        print(f"  [cached] {encoder}")
        return True
    t0 = time.time()
    try:
        X, y, stems, patient_ids, slides_df, folds_df = load_encoder_data(
            manifest_path=MANIFEST_PATH,
            inventory_path=INVENTORY_PATH,
            folds_path=FOLDS_PATH,
            encoder_name=encoder,
            aggregation="mean",
            verbose=False,
        )
    except Exception as e:
        print(f"  !! {encoder}: load failed — {e}")
        return False

    np.savez_compressed(
        cp,
        X=X.astype(np.float32),
        y=np.asarray(y, dtype=np.int8),
        stems=np.asarray(stems, dtype=object),
        patient_ids=np.asarray(patient_ids, dtype=object),
        encoder_type=etype,
    )
    print(f"  [built ] {encoder:16} X={X.shape}  {time.time()-t0:.0f}s")
    return True


def load_cache(encoder: str):
    d = np.load(cache_path(encoder), allow_pickle=True)
    return (d["X"], d["y"], list(d["stems"]), np.asarray(d["patient_ids"]),
            str(d["encoder_type"]))


# ════════════════════════════════════════════════════════════════════════════
# 1B — SITE-LEAKAGE SCORING
# ════════════════════════════════════════════════════════════════════════════

def site_leakage_score(X, patient_ids, site_labels, encoder: str):
    """
    Multiclass CV: predict tissue source site from the slide embedding.
    Patient-grouped folds. Returns a dict of scores.
    """
    sites = np.asarray(site_labels)
    n_sites = len(np.unique(sites))
    chance = 1.0 / n_sites

    # map site strings -> ints
    uniq = sorted(np.unique(sites))
    s2i = {s: i for i, s in enumerate(uniq)}
    y_site = np.array([s2i[s] for s in sites])

    skf = StratifiedGroupKFold(n_splits=N_SITE_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = np.full(len(y_site), -1, dtype=int)
    oof_prob = np.zeros((len(y_site), n_sites), dtype=float)

    for tr, te in skf.split(X, y_site, groups=patient_ids):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                 class_weight="balanced", random_state=SEED)
        clf.fit(Xtr, y_site[tr])
        oof_pred[te] = clf.predict(Xte)
        p = clf.predict_proba(Xte)
        # align columns to full site set (a fold may miss a class)
        for j, cls in enumerate(clf.classes_):
            oof_prob[te, cls] = p[:, j]

    mask = oof_pred >= 0
    bacc = balanced_accuracy_score(y_site[mask], oof_pred[mask])
    acc  = accuracy_score(y_site[mask], oof_pred[mask])
    try:
        macro_auc = roc_auc_score(y_site[mask], oof_prob[mask],
                                  multi_class="ovr", average="macro")
    except Exception:
        macro_auc = float("nan")

    normalised = (bacc - chance) / (1.0 - chance)

    return {
        "encoder": encoder,
        "n_sites": int(n_sites),
        "chance": round(float(chance), 4),
        "site_bacc": round(float(bacc), 4),
        "site_acc": round(float(acc), 4),
        "site_macro_auroc": round(float(macro_auc), 4),
        "site_leakage_score": round(float(normalised), 4),   # <- the headline number
        "n_slides": int(mask.sum()),
        "feat_dim": int(X.shape[1]),
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(build_cache=True, score_sites=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SITE_DIR, exist_ok=True)
    t_start = time.time()

    print("=" * 72)
    print("STEP 1 — Embedding cache + site-leakage scores")
    print("=" * 72)

    # ── 1A: cache ─────────────────────────────────────────────────────────
    if build_cache:
        print(f"\n[1A] Building embedding cache -> {CACHE_DIR}")
        todo = [e for e in ALL_ENCODERS if not os.path.exists(cache_path(e))]
        print(f"  {len(ALL_ENCODERS)-len(todo)} cached, {len(todo)} to build\n")
        for enc in ALL_ENCODERS:
            etype = "slide" if enc in SLIDE_ENCODERS else "patch"
            build_cache_for(enc, etype)

    available = [e for e in ALL_ENCODERS if os.path.exists(cache_path(e))]
    print(f"\n  Cache ready for {len(available)}/{len(ALL_ENCODERS)} encoders")
    if not score_sites:
        print("\nscore_sites=False -> stopping after cache build.")
        return None

    # ── 1B: site labels ───────────────────────────────────────────────────
    print(f"\n[1B] Computing site-leakage scores")
    man = pd.read_csv(MANIFEST_PATH)
    man["tss"] = man["patient_id"].map(extract_tss)

    pat_site = (man.groupby("patient_id")["tss"].first())
    counts   = pat_site.value_counts()
    usable   = sorted(counts[counts >= MIN_PATIENTS_PER_SITE].index.tolist())
    print(f"  Usable sites (>= {MIN_PATIENTS_PER_SITE} patients): "
          f"{len(usable)} -> {usable}")

    results = []
    for enc in available:
        X, y, stems, patient_ids, etype = load_cache(enc)
        sites = np.array([extract_tss(p) for p in patient_ids])
        keep  = np.isin(sites, usable)
        if keep.sum() < 50:
            print(f"  !! {enc}: too few slides in usable sites — skipping")
            continue
        r = site_leakage_score(X[keep], patient_ids[keep], sites[keep], enc)
        r["encoder_type"] = etype
        results.append(r)
        print(f"  {enc:16} [{etype:5}] leakage={r['site_leakage_score']:.4f}  "
              f"bacc={r['site_bacc']:.4f}  (chance={r['chance']:.3f})")

    df = pd.DataFrame(results).sort_values("site_leakage_score", ascending=False)
    out = os.path.join(SITE_DIR, "site_leakage_scores.csv")
    df.to_csv(out, index=False)

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("SITE-LEAKAGE RANKING (higher = encodes hospital identity more strongly)")
    print("-" * 72)
    print(f"{'Rank':<5} {'Encoder':<17} {'Type':<6} {'Leakage':>8} "
          f"{'Bal-Acc':>8} {'MacroAUC':>9}")
    print("-" * 72)
    for i, (_, r) in enumerate(df.iterrows(), 1):
        print(f"{i:<5} {r['encoder']:<17} {r['encoder_type']:<6} "
              f"{r['site_leakage_score']:>8.4f} {r['site_bacc']:>8.4f} "
              f"{r['site_macro_auroc']:>9.4f}")

    print("\n" + "-" * 72)
    print("INTERPRETATION")
    print("-" * 72)
    print(f"  chance-level balanced accuracy : {df['chance'].iloc[0]:.4f}")
    print(f"  leakage score range            : "
          f"{df['site_leakage_score'].min():.4f} – {df['site_leakage_score'].max():.4f}")
    print(f"  mean leakage                   : {df['site_leakage_score'].mean():.4f}")
    hi = df.iloc[0]; lo = df.iloc[-1]
    print(f"  MOST site-confounded           : {hi['encoder']} "
          f"({hi['site_leakage_score']:.4f})")
    print(f"  LEAST site-confounded          : {lo['encoder']} "
          f"({lo['site_leakage_score']:.4f})")
    print("\n  A leakage score near 0 means the encoder cannot identify the hospital.")
    print("  Near 1 means hospital identity is almost perfectly recoverable — which")
    print("  is batch signal, not morphology.")
    print(f"\n  Saved -> {out}")
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    print("=" * 72)
    print("\nNEXT: Step 2 — freeze LOSO folds, then Step 3 — LOSO re-evaluation.")
    return df


if __name__ == "__main__":
    df = main(build_cache=True, score_sites=True)
