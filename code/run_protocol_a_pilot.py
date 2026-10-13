#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase1_titan_pilot.py — End-to-end pilot on TITAN slide encoder
================================================================
PURPOSE: Prove the entire downstream skeleton (load → CV → metrics → bootstrap
         CIs → calibration → save) on ONE encoder before scaling to all 30.

Why TITAN first?
  * Strongest / most-cited slide encoder; pairs with conch_v15 features
  * One vector per slide (768-dim) → fastest possible full loop
  * 434 slides (25 TCGA-KK-A8I* slides missing, all high-risk, known gap)

Once this runs cleanly, every other encoder in Phases 2-4 calls eval_engine
with a different encoder name. The harness is identical.

Runtime estimate (Colab A100, Drive FUSE):
  * H5 loading 434 files: ~3-5 min (Drive FUSE bottleneck)
  * 15 CV folds (5-fold × 3 repeats) × inner C selection: ~2-3 min
  * Bootstrap 1000×: ~30 sec
  * Total: ~6-10 min
"""

import os
import sys
import time
import numpy as np

# ── If running as a Colab cell, point to where you saved eval_engine.py ──
EVAL_ENGINE_DIR = os.environ.get("PCA_ROOT", "./data")
if EVAL_ENGINE_DIR not in sys.path:
    sys.path.insert(0, EVAL_ENGINE_DIR)

from eval_engine import (
    load_encoder_data,
    run_linear_probe_cv,
    compute_metrics,
    bootstrap_ci,
    compute_ece,
    plot_results,
    save_results,
)

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — only thing you change between encoder runs
# ════════════════════════════════════════════════════════════════════════════
MANIFEST_DIR   = os.environ.get("PCA_ROOT", "./data") + "/manifest"
RESULTS_DIR    = os.environ.get("PCA_ROOT", "./data") + "/results/linear_probe"
FIGURES_DIR    = os.environ.get("PCA_ROOT", "./data") + "/results/figures"

MANIFEST_PATH  = os.path.join(MANIFEST_DIR, "manifest_slides.csv")
INVENTORY_PATH = os.path.join(MANIFEST_DIR, "feature_inventory.csv")
FOLDS_PATH     = os.path.join(MANIFEST_DIR, "folds_frozen.csv")

ENCODER_NAME   = "titan"
METHOD         = "linear_probe"

# C search space for inner loop (feel free to widen after pilot confirms it's fast)
C_RANGE        = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

BOOTSTRAP_N    = 1000    # set to 200 for a quick smoke-test, 1000 for the real run
ECE_BINS       = 10

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 65)
    print(f"PHASE 1 — TITAN PILOT  ({ENCODER_NAME} / {METHOD})")
    print("=" * 65)

    # ── 1. Load features ──────────────────────────────────────────────────
    print("\n[1/5] Loading TITAN features from Drive...")
    X, y, stems, patient_ids, slides_df, folds_df = load_encoder_data(
        manifest_path  = MANIFEST_PATH,
        inventory_path = INVENTORY_PATH,
        folds_path     = FOLDS_PATH,
        encoder_name   = ENCODER_NAME,
        aggregation    = "mean",   # slide encoder → already 1×D, squeeze applied inside
        verbose        = True,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Quick sanity check before committing to the full CV
    assert X.ndim == 2,             f"X should be 2D, got {X.shape}"
    assert X.shape[0] == len(y),    "X / y length mismatch"
    assert not np.isnan(X).any(),   "NaNs in feature matrix — check H5 files"
    assert not np.isinf(X).any(),   "Infs in feature matrix — check H5 files"
    print(f"  Sanity checks passed  ✓")

    # ── 2. CV loop ────────────────────────────────────────────────────────
    print(f"\n[2/5] Running {METHOD} CV (5-fold × 3 repeats)...")
    t1 = time.time()
    oof_df = run_linear_probe_cv(
        X           = X,
        y           = y,
        stems       = stems,
        patient_ids = patient_ids,
        folds_df    = folds_df,
        C_range     = C_RANGE,
        verbose     = True,
    )
    print(f"  CV done in {time.time()-t1:.1f}s | OOF rows: {len(oof_df)}")

    # Print the C values chosen (good sanity check that inner CV is working)
    C_chosen = oof_df.groupby(["repeat", "fold"])["best_C"].first()
    print(f"  C values chosen across folds:\n{C_chosen.to_string()}")

    # ── 3. OOF metrics (point estimates from pooled predictions) ─────────
    print("\n[3/5] Computing point-estimate metrics on pooled OOF predictions...")
    # Pool: average probability across repeats per slide
    slide_pool = (oof_df.groupby("slide_stem")
                        .agg(y_true=("y_true",  "first"),
                             y_prob=("y_prob",  "mean"))
                        .reset_index())
    point_metrics = compute_metrics(
        slide_pool["y_true"].values,
        slide_pool["y_prob"].values,
    )
    print(f"  Point estimates: " +
          " | ".join(f"{k}={v:.4f}" for k, v in point_metrics.items()
                     if k not in ("threshold", "n", "n_pos")))

    # ── 4. Bootstrap CIs ─────────────────────────────────────────────────
    print(f"\n[4/5] Bootstrap CIs ({BOOTSTRAP_N} draws, patient-level resampling)...")
    t2 = time.time()
    ci, global_thr = bootstrap_ci(
        oof_df    = oof_df,
        n_boot    = BOOTSTRAP_N,
        verbose   = True,
    )
    print(f"  Bootstrap done in {time.time()-t2:.1f}s")

    # ── 5. Calibration (ECE) ─────────────────────────────────────────────
    print(f"\n[5/5] Calibration (ECE, {ECE_BINS} bins)...")
    ece, bin_data = compute_ece(
        y_true = slide_pool["y_true"].values,
        y_prob = slide_pool["y_prob"].values,
        n_bins = ECE_BINS,
    )
    print(f"  ECE = {ece:.4f}  (0 = perfect calibration, 0.1+ = poor)")

    # ── 6. Save & plot ───────────────────────────────────────────────────
    print("\n[6/6] Saving results and figures...")
    result = save_results(
        ci            = ci,
        global_thr    = global_thr,
        ece           = ece,
        oof_df        = oof_df,
        encoder_name  = ENCODER_NAME,
        encoder_type  = "slide",
        method        = METHOD,
        output_dir    = RESULTS_DIR,
        extra         = {"point_metrics": point_metrics},
    )
    plot_results(
        oof_df       = oof_df,
        ci           = ci,
        ece          = ece,
        bin_data     = bin_data,
        global_thr   = global_thr,
        encoder_name = ENCODER_NAME,
        output_dir   = FIGURES_DIR,
    )

    print(f"\n{'='*65}")
    print(f"PHASE 1 COMPLETE — Total time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*65}")
    print("\nNEXT: Phase 2 — run CHIEF, MADELEINE, FEATHER through the same harness.")
    print("      Change only: ENCODER_NAME = 'chief'  (or madeleine / feather)")
    print("      Everything else stays identical.")
    return result


if __name__ == "__main__":
    result = main()
