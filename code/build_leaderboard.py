#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 6 — The honest leaderboard + robustness report model cards
==============================================================
Assembles every result from Steps 1-5 into the paper's final table, and runs
the released tool over the encoder zoo to produce per-encoder model cards.

TWO DELIVERABLES
    1. honest_leaderboard.csv / .md
       All 30 encoders ranked by LOSO AUROC (the honest metric), alongside the
       random-CV number, the generalization gap, rank movement, site leakage,
       harmonization recovery, and calibration.

    2. model_cards/{encoder}.md
       One robustness report card per encoder — the artifact other groups reuse.

DEPLOYMENT SCORE
    A composite for practitioners, deliberately simple and transparent:
        0.60 * LOSO AUROC                      (does it work at a new hospital?)
      + 0.25 * (1 - normalised generalization gap)   (is the benchmark honest?)
      + 0.15 * (1 - normalised LOSO ECE)             (are probabilities usable?)
    Site leakage is reported but NOT scored: Step 4 showed leakage magnitude
    does not predict fragility (rho = +0.272, p = 0.15), so scoring it would
    imply a relationship the data does not support.

RULE: Never delete any files.
"""

import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.environ.get("PCA_ROOT", "/content/drive/MyDrive/ProstateCancer")
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from robustness_report import RobustnessReport, __version__ as RR_VERSION

# ════════════════════════════════════════════════════════════════════════════
CACHE_DIR   = f"{BASE}/cache/embeddings"
MANIFEST    = f"{BASE}/manifest"
SITE_DIR    = f"{BASE}/results/site_analysis"
HARM_DIR    = f"{BASE}/results/harmonization"
STEP6_DIR   = f"{BASE}/results/step6"
CARD_DIR    = f"{STEP6_DIR}/model_cards"

PHASE3_LB   = f"{BASE}/results/phase3_leaderboard.csv"
LOSO_LB     = f"{BASE}/results/loso_leaderboard.csv"
LEAK_CSV    = f"{SITE_DIR}/site_leakage_scores.csv"
HARM_CSV    = f"{HARM_DIR}/harmonization_results.csv"

GENERATE_CARDS = True     # set False to build the leaderboard only
MIN_PATIENTS_PER_SITE = 15

SLIDE_ENCODERS = ["titan", "chief", "madeleine", "feather"]


def extract_tss(pid):
    p = str(pid).split("-")
    return p[1].upper() if len(p) >= 2 else "UNKNOWN"


def _norm(s):
    """Min-max to [0,1]; all-equal -> 0.5."""
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi - lo < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


# ════════════════════════════════════════════════════════════════════════════
def build_leaderboard():
    print("=" * 78)
    print("STEP 6 — HONEST LEADERBOARD")
    print("=" * 78)

    loso = pd.read_csv(LOSO_LB)
    print(f"\nLoaded LOSO results: {len(loso)} encoders")

    lb = loso.copy()

    # site leakage
    if os.path.exists(LEAK_CSV):
        lk = pd.read_csv(LEAK_CSV)
        cols = [c for c in ["encoder", "site_leakage_score", "site_bacc", "feat_dim"]
                if c in lk.columns]
        lb = lb.merge(lk[cols], on="encoder", how="left",
                      suffixes=("", "_leak"))

    # harmonization (subset only)
    if os.path.exists(HARM_CSV):
        h = pd.read_csv(HARM_CSV)
        hs = h[h.method == "site_standardization"][["encoder", "loso_auroc", "site_leakage"]]
        hs = hs.rename(columns={"loso_auroc": "harmonized_auroc",
                                "site_leakage": "harmonized_leakage"})
        lb = lb.merge(hs, on="encoder", how="left")
        base = h[(h.method == "pc_removal") & (h.k == 0)][["encoder", "loso_auroc"]]
        base = base.rename(columns={"loso_auroc": "_harm_base"})
        lb = lb.merge(base, on="encoder", how="left")
        lb["harmonization_recovery"] = (lb["harmonized_auroc"] - lb["_harm_base"]).round(4)
        lb = lb.drop(columns=["_harm_base"])

    # generalization gap
    if "phase3_auroc" in lb.columns:
        lb["generalization_gap"] = (lb["phase3_auroc"] - lb["loso_auroc"]).round(4)

    # ── deployment score ─────────────────────────────────────────────────
    parts, weights = [], []
    parts.append(_norm(lb["loso_auroc"]));                weights.append(0.60)
    if "generalization_gap" in lb.columns:
        parts.append(1 - _norm(lb["generalization_gap"])); weights.append(0.25)
    if "loso_ece" in lb.columns:
        parts.append(1 - _norm(lb["loso_ece"]));           weights.append(0.15)
    w = np.array(weights) / np.sum(weights)
    lb["deployment_score"] = np.round(
        np.sum([p.values * wi for p, wi in zip(parts, w)], axis=0), 4)

    lb = lb.sort_values("loso_auroc", ascending=False).reset_index(drop=True)
    lb["honest_rank"] = range(1, len(lb) + 1)

    front = ["honest_rank", "encoder", "type", "loso_auroc", "loso_lo", "loso_hi",
             "phase3_auroc", "generalization_gap", "rank_change",
             "site_leakage_score", "harmonized_auroc", "harmonization_recovery",
             "loso_ece", "deployment_score"]
    lb = lb[[c for c in front if c in lb.columns] +
            [c for c in lb.columns if c not in front]]

    os.makedirs(STEP6_DIR, exist_ok=True)
    lb.to_csv(f"{STEP6_DIR}/honest_leaderboard.csv", index=False)
    return lb


def print_leaderboard(lb):
    print("\n" + "=" * 78)
    print("HONEST LEADERBOARD — ranked by leave-one-site-out AUROC")
    print("=" * 78)
    print(f"{'#':<4}{'Encoder':<16}{'Type':<7}{'LOSO':>8}{'RandCV':>8}"
          f"{'Gap':>8}{'ΔRank':>7}{'Leak':>7}{'Deploy':>8}")
    print("-" * 78)
    for _, r in lb.iterrows():
        g = r.get("generalization_gap", np.nan)
        rc = r.get("rank_change", np.nan)
        lk = r.get("site_leakage_score", np.nan)
        p3 = r.get("phase3_auroc", np.nan)
        print(f"{int(r['honest_rank']):<4}{r['encoder']:<16}{r['type']:<7}"
              f"{r['loso_auroc']:>8.4f}{p3:>8.4f}{g:>+8.4f}"
              f"{rc:>+7.0f}{lk:>7.3f}{r['deployment_score']:>8.3f}")


def write_markdown(lb):
    L = ["# Honest Leaderboard — Pathology Foundation Models", "",
         "Ranked by **leave-one-site-out (LOSO) AUROC**: performance at a hospital the",
         "model has never seen. The `RandCV` column is the number a standard",
         "patient-grouped benchmark would report; `Gap` is how much that overstates.", "",
         "| # | Encoder | Type | LOSO AUROC | Rand-CV | Gap | ΔRank | Site leakage | Deploy |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in lb.iterrows():
        def f(k, n=4, sign=False):
            v = r.get(k, np.nan)
            if pd.isna(v):
                return "—"
            return f"{v:+.{n}f}" if sign else f"{v:.{n}f}"
        rc = r.get("rank_change", np.nan)
        L.append(f"| {int(r['honest_rank'])} | `{r['encoder']}` | {r['type']} | "
                 f"**{f('loso_auroc')}** | {f('phase3_auroc')} | "
                 f"{f('generalization_gap', 4, True)} | "
                 f"{'—' if pd.isna(rc) else f'{rc:+.0f}'} | "
                 f"{f('site_leakage_score', 3)} | {f('deployment_score', 3)} |")

    slide = lb[lb.type == "slide"]; patch = lb[lb.type == "patch"]
    L += ["", "## Summary", "",
          f"- **Slide encoders** ({len(slide)}): mean LOSO AUROC "
          f"{slide.loso_auroc.mean():.4f}, mean gap "
          f"{slide.generalization_gap.mean():+.4f}",
          f"- **Patch encoders** ({len(patch)}): mean LOSO AUROC "
          f"{patch.loso_auroc.mean():.4f}, mean gap "
          f"{patch.generalization_gap.mean():+.4f}", "",
          "## How to read this", "",
          "- **Gap** — how much a standard benchmark overstates cross-hospital performance.",
          "- **ΔRank** — positions moved between patient-grouped CV and LOSO. "
          "Large movement means the standard ranking is protocol-dependent.",
          "- **Site leakage** — recoverability of hospital identity from the embedding "
          "(0 = none, 1 = perfect). Reported but **not** scored: leakage magnitude does "
          "not predict fragility (ρ = +0.272, p = 0.15).",
          "- **Deploy** — composite: 0.60·LOSO + 0.25·(1−gap) + 0.15·(1−ECE).", "",
          f"*Generated by robustness report v{RR_VERSION}.*"]
    with open(f"{STEP6_DIR}/honest_leaderboard.md", "w") as f:
        f.write("\n".join(L) + "\n")


def generate_cards():
    print("\n" + "=" * 78)
    print("MODEL CARDS")
    print("=" * 78)
    os.makedirs(CARD_DIR, exist_ok=True)

    caches = sorted(glob.glob(f"{CACHE_DIR}/*.npz"))
    print(f"  {len(caches)} cached encoders found")
    made = []
    for cp in caches:
        enc = os.path.basename(cp).replace(".npz", "")
        card = f"{CARD_DIR}/{enc}.md"
        if os.path.exists(card):
            print(f"  [cached] {enc}"); made.append(enc); continue
        try:
            d = np.load(cp, allow_pickle=True)
            X = d["X"]; y = np.asarray(d["y"]).astype(int)
            pids = np.asarray(d["patient_ids"])
            sites = np.array([extract_tss(p) for p in pids])
            a = RobustnessReport(X, y, pids, sites, encoder_name=enc,
                               min_patients_per_site=MIN_PATIENTS_PER_SITE,
                               verbose=False)
            r = a.run()
            a.to_markdown(card)
            a.to_json(f"{CARD_DIR}/{enc}.json")
            print(f"  {enc:16} LOSO={r['loso_auroc']:.4f} "
                  f"gap={r['generalization_gap']:+.4f} "
                  f"leak={r['site_leakage']:.3f}")
            made.append(enc)
        except Exception as e:
            print(f"  !! {enc}: {e}")
    return made


def main():
    os.makedirs(STEP6_DIR, exist_ok=True)
    lb = build_leaderboard()
    print_leaderboard(lb)
    write_markdown(lb)

    print("\n" + "-" * 78)
    print("KEY CONTRASTS FOR THE PAPER")
    print("-" * 78)
    slide = lb[lb.type == "slide"]; patch = lb[lb.type == "patch"]
    print(f"  slide mean LOSO {slide.loso_auroc.mean():.4f} | "
          f"gap {slide.generalization_gap.mean():+.4f}")
    print(f"  patch mean LOSO {patch.loso_auroc.mean():.4f} | "
          f"gap {patch.generalization_gap.mean():+.4f}")
    top = lb.iloc[0]
    print(f"\n  Honest #1: {top['encoder']} (LOSO {top['loso_auroc']:.4f}, "
          f"gap {top['generalization_gap']:+.4f})")
    if "rank_change" in lb.columns:
        mv = lb.reindex(lb.rank_change.abs().sort_values(ascending=False).index).head(5)
        print("\n  Largest rank movements:")
        for _, r in mv.iterrows():
            print(f"    {r['encoder']:16} {r['rank_change']:+.0f} positions")

    if GENERATE_CARDS:
        generate_cards()

    print(f"\n  Saved -> {STEP6_DIR}/honest_leaderboard.csv")
    print(f"  Saved -> {STEP6_DIR}/honest_leaderboard.md")
    if GENERATE_CARDS:
        print(f"  Saved -> {CARD_DIR}/")
    print("\n" + "=" * 78)
    print("NEXT: Step 7 — arXiv preprint + GitHub release.")
    print("=" * 78)
    return lb


if __name__ == "__main__":
    lb = main()
