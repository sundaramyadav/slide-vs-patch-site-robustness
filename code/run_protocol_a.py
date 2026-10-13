#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase3_patch_meanpool_all.py  —  Mean-pool linear probe: all 26 patch encoders
===============================================================================
PURPOSE:
    The critical "is MIL even worth it?" control.
    Mean-pool collapses each patch bag into one slide vector,
    then a linear probe classifies it — identical to the slide encoder
    branch but using patch encoder features as input.

    If mean-pool patch encoders match or beat MIL, that is itself
    a finding (and simplifies the clinical pipeline considerably).

DESIGN:
    * Calls eval_engine.py exactly as Phase 1/2 did — no new code
    * Loops all 26 patch encoders in one unattended run
    * CHECKPOINT: skips any encoder whose metrics JSON already exists
      → safe to re-run after a Colab disconnect; picks up where it left off
    * Ordered by feature dimension (small→large) to front-load fast encoders
    * Prints a running leaderboard after each encoder
    * Saves a combined CSV leaderboard at the end

RUNTIME ESTIMATE (Colab Pro+ A100, Drive FUSE):
    * ~8–15 min per encoder (FUSE read bottleneck, not compute)
    * ~3.5–6 hrs total for all 26
    * Run overnight or use checkpoint to split across sessions

RULE: Never delete any files.
"""

import os
import sys
import json
import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── eval_engine location ────────────────────────────────────────────────────
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
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
MANIFEST_DIR  = os.environ.get("PCA_ROOT", "./data") + "/manifest"
RESULTS_DIR   = os.environ.get("PCA_ROOT", "./data") + "/results/meanpool"
FIGURES_DIR   = os.environ.get("PCA_ROOT", "./data") + "/results/figures"
LEADERBOARD   = os.environ.get("PCA_ROOT", "./data") + "/results/phase3_leaderboard.csv"

MANIFEST_PATH  = os.path.join(MANIFEST_DIR, "manifest_slides.csv")
INVENTORY_PATH = os.path.join(MANIFEST_DIR, "feature_inventory.csv")
FOLDS_PATH     = os.path.join(MANIFEST_DIR, "folds_frozen.csv")

METHOD         = "meanpool"
BOOTSTRAP_N    = 1000   # set to 200 for a quick smoke-test
ECE_BINS       = 10
AGGREGATION    = "mean"

# Extended C range — slide encoder runs showed 0.001 was pinned at bottom,
# so we go one notch lower to be rigorous.
C_RANGE = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

# ── Encoder run order: small-dim first (fastest FUSE reads) ─────────────────
# Ordered by feature dimension ascending within each config group.
# musk last within small-dim group because it has only 409 slides.
PATCH_ENCODERS = [
    # 384-dim — fastest (small files)
    "kaiko-vits8",
    "kaiko-vits16",
    "lunit-vits8",
    # 512-dim
    "conch_v1",
    # 768-dim
    "kaiko-vitb8",
    "kaiko-vitb16",
    "h0-mini",
    "phikon",
    "ctranspath",
    "conch_v15",
    # 1024-dim
    "hibou_l",
    "phikon_v2",
    "gpfm",
    "kaiko-vitl14",
    # 1536-dim
    "hoptimus0",
    "hoptimus1",
    "openmidnight",
    "midnight12k",
    "uni_v1",
    # 2048-dim
    "resnet50",
    # 1536-dim (UNI models — separate config)
    "uni_v2",
    # special: only 409 slides (20x/384px config)
    "musk",
    # 2560-dim — large files
    "virchow",
    "virchow2",
    # variable-dim
    "gigapath",
    # 4608-dim — largest, run last
    "genbio-pathfm",
]

assert len(PATCH_ENCODERS) == 26, f"Expected 26 patch encoders, got {len(PATCH_ENCODERS)}"

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def checkpoint_path(encoder: str) -> str:
    return os.path.join(RESULTS_DIR, f"{encoder}_{METHOD}_metrics.json")


def is_done(encoder: str) -> bool:
    """Return True if this encoder already has a saved metrics JSON."""
    return os.path.exists(checkpoint_path(encoder))


def run_one_encoder(encoder: str) -> dict:
    """Full pipeline for one patch encoder with mean-pool aggregation."""
    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"  {encoder.upper()}  [{METHOD}]")
    print(f"{'='*65}")

    # 1. Load
    print("  [1/5] Loading features (mean-pooling patches)...")
    try:
        X, y, stems, patient_ids, slides_df, folds_df = load_encoder_data(
            manifest_path  = MANIFEST_PATH,
            inventory_path = INVENTORY_PATH,
            folds_path     = FOLDS_PATH,
            encoder_name   = encoder,
            aggregation    = AGGREGATION,
            verbose        = True,
        )
    except Exception as e:
        print(f"  !! LOAD FAILED for {encoder}: {e}")
        return {"encoder": encoder, "status": "LOAD_FAILED", "error": str(e)}

    load_time = time.time() - t0

    # Sanity checks
    if X.shape[0] == 0:
        print(f"  !! No slides loaded for {encoder} — skipping.")
        return {"encoder": encoder, "status": "NO_DATA"}
    assert not np.isnan(X).any(), "NaNs in feature matrix"
    assert not np.isinf(X).any(), "Infs in feature matrix"
    print(f"  X: {X.shape}  load_time: {load_time:.1f}s")

    # 2. CV
    print("  [2/5] Running CV (5-fold × 3 repeats)...")
    t1 = time.time()
    oof_df = run_linear_probe_cv(
        X=X, y=y, stems=stems, patient_ids=patient_ids,
        folds_df=folds_df, C_range=C_RANGE, verbose=True,
    )
    cv_time = time.time() - t1
    print(f"  CV done: {cv_time:.1f}s | OOF rows: {len(oof_df)}")
    C_chosen = oof_df.groupby(["repeat", "fold"])["best_C"].first()
    print(f"  C chosen: {C_chosen.values.tolist()}")

    # 3. Point metrics
    print("  [3/5] Point-estimate metrics...")
    slide_pool = (oof_df.groupby("slide_stem")
                        .agg(y_true=("y_true", "first"),
                             y_prob=("y_prob", "mean"))
                        .reset_index())
    pt = compute_metrics(slide_pool["y_true"].values, slide_pool["y_prob"].values)
    print("  " + " | ".join(f"{k}={v:.4f}" for k, v in pt.items()
                             if k not in ("threshold", "n", "n_pos")))

    # 4. Bootstrap
    print(f"  [4/5] Bootstrap ({BOOTSTRAP_N} draws)...")
    t2 = time.time()
    ci, global_thr = bootstrap_ci(oof_df=oof_df, n_boot=BOOTSTRAP_N, verbose=True)
    print(f"  Bootstrap: {time.time()-t2:.1f}s")

    # 5. Calibration
    print(f"  [5/5] ECE...")
    ece, bin_data = compute_ece(
        slide_pool["y_true"].values, slide_pool["y_prob"].values, n_bins=ECE_BINS
    )
    print(f"  ECE = {ece:.4f}")

    # Save
    result = save_results(
        ci=ci, global_thr=global_thr, ece=ece,
        oof_df=oof_df, encoder_name=encoder, encoder_type="patch",
        method=METHOD, output_dir=RESULTS_DIR,
        extra={"point_metrics": pt,
               "feat_dim": int(X.shape[1]),
               "load_time_s": round(load_time, 1),
               "cv_time_s": round(cv_time, 1),
               "total_time_s": round(time.time()-t0, 1)},
    )
    plot_results(
        oof_df=oof_df, ci=ci, ece=ece, bin_data=bin_data,
        global_thr=global_thr, encoder_name=encoder, output_dir=FIGURES_DIR,
    )
    return result


def build_leaderboard(results: list) -> pd.DataFrame:
    """Build a sorted leaderboard DataFrame from a list of result dicts."""
    rows = []
    for r in results:
        if "metrics" not in r:
            continue
        m = r["metrics"]
        rows.append({
            "encoder":    r["encoder"],
            "type":       r.get("encoder_type", "patch"),
            "method":     r.get("method", METHOD),
            "n_slides":   r.get("n_slides", ""),
            "n_patients": r.get("n_patients", ""),
            "feat_dim":   r.get("feat_dim", ""),
            "auroc":      round(m["auroc"]["mean"], 4),
            "auroc_lo":   round(m["auroc"]["ci_lower"], 4),
            "auroc_hi":   round(m["auroc"]["ci_upper"], 4),
            "aupr":       round(m["aupr"]["mean"], 4),
            "mcc":        round(m["mcc"]["mean"], 4),
            "bacc":       round(m["bacc"]["mean"], 4),
            "f1":         round(m["f1"]["mean"], 4),
            "sens":       round(m["sens"]["mean"], 4),
            "spec":       round(m["spec"]["mean"], 4),
            "ece":        round(r.get("ece", float("nan")), 4),
            "threshold":  round(r.get("threshold", float("nan")), 4),
            "load_s":     r.get("load_time_s", ""),
            "total_s":    r.get("total_time_s", ""),
        })
    df = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def print_leaderboard(df: pd.DataFrame, title: str = "RUNNING LEADERBOARD"):
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}")
    print(f"  {'Rank':<5} {'Encoder':<18} {'AUROC':>7} {'[lo–hi]':>17} "
          f"{'AUPR':>7} {'MCC':>7} {'ECE':>7} {'N':>4}")
    print(f"  {'─'*75}")
    for _, r in df.iterrows():
        ci_str = f"[{r['auroc_lo']:.3f}–{r['auroc_hi']:.3f}]"
        print(f"  {r['rank']:<5} {str(r['encoder']):<18} {r['auroc']:>7.4f} "
              f"{ci_str:>17} {r['aupr']:>7.4f} {r['mcc']:>7.4f} "
              f"{r['ece']:>7.4f} {str(r['n_slides']):>4}")
    print(f"{'─'*80}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t_start = time.time()
    print("=" * 65)
    print("PHASE 3 — PATCH ENCODER MEAN-POOL PROBES (26 encoders)")
    print("=" * 65)

    # Check which encoders are already done
    done   = [e for e in PATCH_ENCODERS if is_done(e)]
    todo   = [e for e in PATCH_ENCODERS if not is_done(e)]
    print(f"\nCheckpoint: {len(done)} done, {len(todo)} remaining.")
    if done:
        print(f"  Already complete: {done}")
    if not todo:
        print("  All encoders complete. Building final leaderboard only.")

    results = []

    # Load already-completed results into memory for the leaderboard
    for enc in done:
        with open(checkpoint_path(enc)) as f:
            results.append(json.load(f))

    # Run remaining encoders
    for i, encoder in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] Starting {encoder} ...")
        try:
            result = run_one_encoder(encoder)
            if "metrics" in result:
                results.append(result)
                # Print running leaderboard after each success
                lb = build_leaderboard(results)
                print_leaderboard(lb, f"LEADERBOARD after {encoder}")
                lb.to_csv(LEADERBOARD, index=False)
                print(f"  Running leaderboard saved → {LEADERBOARD}")
        except Exception as e:
            print(f"\n!! UNEXPECTED ERROR for {encoder}: {e}")
            print("   Continuing with next encoder...")
            import traceback; traceback.print_exc()
            continue

    # ── Final leaderboard ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 3 COMPLETE")
    print("=" * 65)

    # Also load slide encoder results for the combined table
    slide_results_dir = os.environ.get("PCA_ROOT", "./data") + "/results/linear_probe"
    slide_encoders = ["chief", "titan", "madeleine", "feather"]
    slide_results = []
    for enc in slide_encoders:
        p = os.path.join(slide_results_dir, f"{enc}_linear_probe_metrics.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
                d["method"] = "linear_probe (slide)"
                slide_results.append(d)

    all_results = results + slide_results
    final_lb = build_leaderboard(all_results)
    final_lb.to_csv(LEADERBOARD, index=False)
    print_leaderboard(final_lb, "FINAL COMBINED LEADERBOARD (Phases 2+3)")

    # Separate patch-only leaderboard
    patch_lb = final_lb[final_lb["type"] == "patch"].reset_index(drop=True)
    patch_lb["rank"] = range(1, len(patch_lb) + 1)
    patch_path = LEADERBOARD.replace(".csv", "_patch_only.csv")
    patch_lb.to_csv(patch_path, index=False)

    total_min = (time.time() - t_start) / 60
    print(f"\nTotal wall-clock time: {total_min:.1f} min")
    print(f"Leaderboard → {LEADERBOARD}")
    print(f"Patch-only  → {patch_path}")

    # Key comparison: best patch meanpool vs best slide encoder
    best_patch = patch_lb.iloc[0] if len(patch_lb) else None
    best_slide = final_lb[final_lb["type"] == "slide"].iloc[0] if len(final_lb) else None
    if best_patch is not None and best_slide is not None:
        print(f"\n{'─'*50}")
        print(f"HEADLINE COMPARISON (mean-pool vs slide encoder):")
        print(f"  Best patch+meanpool: {best_patch['encoder']:18} "
              f"AUROC={best_patch['auroc']:.4f}")
        print(f"  Best slide encoder:  {best_slide['encoder']:18} "
              f"AUROC={best_slide['auroc']:.4f}")
        delta = best_patch['auroc'] - best_slide['auroc']
        direction = "patch>slide" if delta > 0 else "slide>patch"
        print(f"  Δ = {abs(delta):.4f} ({direction})")
        print(f"  → This answers: 'Is MIL worth it?'")
        print(f"    If patch+meanpool ≈ slide encoder → MIL adds little")
        print(f"    If patch+meanpool > slide encoder → MIL likely adds more")
        print(f"{'─'*50}")

    print("\nNEXT: Phase 4 — ABMIL on top patch encoders.")
    return final_lb


if __name__ == "__main__":
    final_lb = main()
