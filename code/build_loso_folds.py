#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 — Build & freeze LEAVE-ONE-SITE-OUT (LOSO) folds
========================================================
Reads the manifest only. Runs in seconds.

WHAT LOSO SIMULATES
-------------------
Deployment at a hospital the model has never seen. Each fold holds out ALL
patients from one tissue source site; the model trains on every other site.

This is the honest test. The Phase 0/3 folds were patient-grouped, which
correctly prevents PATIENT leakage — but the same hospitals still appeared in
both train and test. If an encoder encodes hospital identity, patient-grouped
CV rewards it. LOSO removes that reward.

FOLD CONSTRUCTION RULES
-----------------------
* One fold per USABLE site (>= MIN_PATIENTS_PER_SITE patients).
* Sites too small, or containing only one risk class when held out, become
  TRAIN-ONLY — they always appear in training, never as a test fold.
  (Held-out AUROC is undefined with a single class.)
* Output schema deliberately MIRRORS folds_frozen.csv so eval_engine.py
  works unchanged — only the file path differs.

OUTPUT SCHEMA (folds_loso_frozen.csv)
-------------------------------------
    slide_stem, patient_id, risk_binary, tss, repeat, fold
    'repeat' is always 0 (LOSO is deterministic — no randomness to repeat).
    'fold' k  ->  test set = all slides from usable site k.
    Train-only sites get fold = -1 (never tested, always trained on).

RULE: Never delete any files.
"""

import os
import json
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
BASE          = os.environ.get("PCA_ROOT", "/content/drive/MyDrive/ProstateCancer")
MANIFEST_DIR  = f"{BASE}/manifest"
SITE_DIR      = f"{BASE}/results/site_analysis"

MANIFEST_PATH = f"{MANIFEST_DIR}/manifest_slides.csv"
LOSO_PATH     = f"{MANIFEST_DIR}/folds_loso_frozen.csv"

MIN_PATIENTS_PER_SITE = 15    # must match Step 0 and Step 1


def extract_tss(patient_id: str) -> str:
    parts = str(patient_id).split("-")
    return parts[1].upper() if len(parts) >= 2 else "UNKNOWN"


def main():
    os.makedirs(SITE_DIR, exist_ok=True)

    print("=" * 72)
    print("STEP 2 — Freeze leave-one-site-out (LOSO) folds")
    print("=" * 72)

    man = pd.read_csv(MANIFEST_PATH)
    man["tss"] = man["patient_id"].map(extract_tss)
    print(f"\nLoaded: {len(man)} slides, {man['patient_id'].nunique()} patients, "
          f"{man['tss'].nunique()} sites")

    # ── identify usable sites ─────────────────────────────────────────────
    pat_site = man.groupby("patient_id")["tss"].first()
    counts   = pat_site.value_counts()
    big      = sorted(counts[counts >= MIN_PATIENTS_PER_SITE].index.tolist())

    # a site is testable only if it contains BOTH classes (else AUROC undefined)
    site_classes = man.groupby("tss")["risk_binary"].nunique()
    testable = [s for s in big if site_classes.get(s, 0) >= 2]
    dropped  = [s for s in big if s not in testable]

    print(f"\nSites with >= {MIN_PATIENTS_PER_SITE} patients : {len(big)} -> {big}")
    if dropped:
        print(f"  !! single-class (train-only)     : {dropped}")
    print(f"Testable LOSO folds                : {len(testable)} -> {testable}")

    if len(testable) < 3:
        print("\n!! Fewer than 3 testable sites — LOSO is not meaningful. Stop.")
        return None

    # ── assign folds ──────────────────────────────────────────────────────
    site_to_fold = {s: i for i, s in enumerate(testable)}
    folds = man[["slide_stem", "patient_id", "risk_binary", "tss"]].copy()
    folds["repeat"] = 0
    folds["fold"]   = folds["tss"].map(site_to_fold).fillna(-1).astype(int)

    n_trainonly = int((folds["fold"] == -1).sum())
    print(f"\nSlides assigned to a test fold     : {(folds['fold'] >= 0).sum()}")
    print(f"Slides train-only (fold = -1)      : {n_trainonly}")

    # ── per-fold report ───────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("LOSO FOLD COMPOSITION")
    print("-" * 72)
    print(f"{'Fold':<5} {'Site':<6} {'Patients':>9} {'Slides':>7} "
          f"{'High':>6} {'Low':>6} {'Prev':>7} {'TrainSlides':>12}")
    print("-" * 72)
    total_slides = len(folds)
    rows = []
    for site, k in site_to_fold.items():
        sub = folds[folds["fold"] == k]
        nh  = int(sub["risk_binary"].sum())
        nl  = len(sub) - nh
        npat = sub["patient_id"].nunique()
        prev = nh / len(sub) if len(sub) else 0
        ntrain = total_slides - len(sub)
        print(f"{k:<5} {site:<6} {npat:>9} {len(sub):>7} {nh:>6} {nl:>6} "
              f"{prev:>7.3f} {ntrain:>12}")
        rows.append({"fold": k, "tss": site, "n_patients": npat,
                     "n_slides": len(sub), "n_high": nh, "n_low": nl,
                     "prevalence": round(prev, 4), "n_train_slides": ntrain})

    # ── leakage assertions ────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("INTEGRITY CHECKS")
    print("-" * 72)
    pat_folds = folds.groupby("patient_id")["fold"].nunique()
    print(f"Patients spanning >1 fold (must be 0)  : "
          f"{int((pat_folds > 1).sum())}")
    site_folds = folds[folds["fold"] >= 0].groupby("tss")["fold"].nunique()
    print(f"Sites spanning >1 fold (must be 0)     : "
          f"{int((site_folds > 1).sum())}")
    covered = folds[folds['fold'] >= 0]['patient_id'].nunique()
    print(f"Patients testable across all folds     : "
          f"{covered} / {folds['patient_id'].nunique()}")

    # ── save ──────────────────────────────────────────────────────────────
    folds.to_csv(LOSO_PATH, index=False)
    comp = pd.DataFrame(rows)
    comp_path = os.path.join(SITE_DIR, "loso_fold_composition.csv")
    comp.to_csv(comp_path, index=False)

    meta = {
        "n_folds": len(testable),
        "testable_sites": testable,
        "train_only_sites": [s for s in man["tss"].unique() if s not in testable],
        "min_patients_per_site": MIN_PATIENTS_PER_SITE,
        "n_slides_testable": int((folds["fold"] >= 0).sum()),
        "n_slides_train_only": n_trainonly,
    }
    meta_path = os.path.join(SITE_DIR, "loso_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved -> {LOSO_PATH}")
    print(f"  Saved -> {comp_path}")
    print(f"  Saved -> {meta_path}")
    print("\n" + "=" * 72)
    print("NEXT: Step 3 — re-evaluate all 30 encoders using folds_loso_frozen.csv")
    print("      (eval_engine.py is reused UNCHANGED — only the folds path differs)")
    print("=" * 72)
    return folds, comp


if __name__ == "__main__":
    out = main()
