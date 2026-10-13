#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 0 — GO/NO-GO CHECK: Tissue Source Site (TSS) distribution & label confounding
==================================================================================
Runs in ~5 seconds. Loads NO feature files — only the manifest CSV.

PURPOSE
-------
The entire "are FM benchmarks measuring biology or batch effects?" study hinges on
two facts we can establish right now:

  Q1. Do we have enough tissue source sites, with enough patients each,
      to run leave-one-site-out (LOSO) cross-validation?

  Q2. Does high-risk prevalence actually VARY across sites?
      If every hospital has the same Gleason case-mix, site cannot confound the
      label, and the confounding story collapses. If prevalence varies, the
      confounding pathway is real.

TCGA BARCODE ANATOMY
--------------------
    TCGA - KK - A8IL
     |      |     |
     |      |     +-- participant code
     |      +-------- TISSUE SOURCE SITE (the submitting institution)  <-- what we want
     +--------------- project

DECISION RULE (printed at the end)
----------------------------------
  GO        : >= 8 sites with >= 15 patients each, AND significant prevalence variation
  WEAK GO   : enough sites, but prevalence variation not significant
              -> study becomes a robustness/generalization paper (still publishable)
  NO-GO     : too few usable sites -> fall back to committee-disagreement design

RULE: Never delete any files.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — paths already match your Drive
# ════════════════════════════════════════════════════════════════════════════
MANIFEST_DIR  = os.environ.get("PCA_ROOT", "/content/drive/MyDrive/ProstateCancer") + "/manifest"
MANIFEST_PATH = os.path.join(MANIFEST_DIR, "manifest_slides.csv")
OUTPUT_DIR    = os.environ.get("PCA_ROOT", "/content/drive/MyDrive/ProstateCancer") + "/results/site_analysis"

# Thresholds for the decision rule
MIN_PATIENTS_PER_SITE = 15   # a site needs this many patients to be a usable LOSO fold
MIN_USABLE_SITES      = 8    # need this many usable sites for LOSO to be meaningful
ALPHA                 = 0.05


def extract_tss(patient_id: str) -> str:
    """TCGA-KK-A8IL -> 'KK'  (the tissue source site / submitting institution)."""
    parts = str(patient_id).split("-")
    return parts[1].upper() if len(parts) >= 2 else "UNKNOWN"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print("STEP 0 — GO/NO-GO: Tissue Source Site distribution & label confounding")
    print("=" * 72)

    # ── Load manifest ─────────────────────────────────────────────────────
    if not os.path.exists(MANIFEST_PATH):
        print(f"!! manifest not found at {MANIFEST_PATH}")
        print("   Is Drive mounted?  from google.colab import drive; drive.mount('/content/drive')")
        sys.exit(1)

    man = pd.read_csv(MANIFEST_PATH)
    print(f"\nLoaded manifest: {len(man)} slides, "
          f"{man['patient_id'].nunique()} patients")

    # ── Derive site ───────────────────────────────────────────────────────
    man["tss"] = man["patient_id"].map(extract_tss)

    # ── Per-site table (patient-level is the honest unit) ─────────────────
    pat = (man.groupby(["patient_id", "tss"])["risk_binary"]
              .max()                       # patient is high-risk if ANY slide is
              .reset_index(name="patient_high_risk"))

    site_tbl = (pat.groupby("tss")
                   .agg(n_patients=("patient_id", "nunique"),
                        n_high=("patient_high_risk", "sum"))
                   .reset_index())
    site_tbl["n_low"]      = site_tbl["n_patients"] - site_tbl["n_high"]
    site_tbl["prevalence"] = (site_tbl["n_high"] / site_tbl["n_patients"]).round(3)

    # slides per site too
    slides_per_site = man.groupby("tss")["slide_stem"].nunique()
    site_tbl["n_slides"] = site_tbl["tss"].map(slides_per_site)

    site_tbl = site_tbl.sort_values("n_patients", ascending=False).reset_index(drop=True)

    # ── Q1: enough sites? ─────────────────────────────────────────────────
    usable = site_tbl[site_tbl["n_patients"] >= MIN_PATIENTS_PER_SITE]

    print("\n" + "-" * 72)
    print("Q1 — SITE DISTRIBUTION")
    print("-" * 72)
    print(f"Total distinct sites            : {len(site_tbl)}")
    print(f"Sites with >= {MIN_PATIENTS_PER_SITE} patients        : {len(usable)}")
    print(f"Patients in usable sites        : {usable['n_patients'].sum()} "
          f"/ {site_tbl['n_patients'].sum()}")

    print(f"\n{'Site':<6} {'Patients':>9} {'Slides':>7} {'High':>6} {'Low':>6} "
          f"{'Prevalence':>11}  {'Usable':>7}")
    print("-" * 72)
    for _, r in site_tbl.iterrows():
        flag = "  YES" if r["n_patients"] >= MIN_PATIENTS_PER_SITE else "   no"
        print(f"{r['tss']:<6} {r['n_patients']:>9} {r['n_slides']:>7} "
              f"{r['n_high']:>6} {r['n_low']:>6} {r['prevalence']:>11.3f}  {flag:>7}")

    # ── Q2: is prevalence confounded with site? ───────────────────────────
    print("\n" + "-" * 72)
    print("Q2 — IS HIGH-RISK PREVALENCE CONFOUNDED WITH SITE?")
    print("-" * 72)

    overall_prev = pat["patient_high_risk"].mean()
    print(f"Overall high-risk prevalence    : {overall_prev:.3f}")

    chi2 = p_val = cramers_v = np.nan
    if len(usable) >= 2:
        # chi-square on usable sites only (small sites make the test unreliable)
        sub = pat[pat["tss"].isin(usable["tss"])]
        ct  = pd.crosstab(sub["tss"], sub["patient_high_risk"])
        if ct.shape[0] >= 2 and ct.shape[1] == 2:
            chi2, p_val, dof, expected = stats.chi2_contingency(ct)
            n = ct.values.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
            print(f"Chi-square (usable sites)       : chi2={chi2:.2f}, "
                  f"dof={dof}, p={p_val:.2e}")
            print(f"Cramer's V (effect size)        : {cramers_v:.3f}  "
                  f"(0.1 small / 0.3 medium / 0.5 large)")

            low_e = (expected < 5).sum()
            if low_e:
                print(f"  note: {low_e} cells with expected count < 5 "
                      f"— treat p-value as approximate")

        prev_range = usable["prevalence"].max() - usable["prevalence"].min()
        print(f"Prevalence range across sites   : "
              f"{usable['prevalence'].min():.3f} – {usable['prevalence'].max():.3f} "
              f"(spread = {prev_range:.3f})")
        print(f"Prevalence std across sites     : {usable['prevalence'].std():.3f}")

        # most extreme sites — these drive the confounding
        ext = usable.sort_values("prevalence")
        print(f"\nLowest-prevalence site  : {ext.iloc[0]['tss']} "
              f"({ext.iloc[0]['prevalence']:.3f}, n={ext.iloc[0]['n_patients']})")
        print(f"Highest-prevalence site : {ext.iloc[-1]['tss']} "
              f"({ext.iloc[-1]['prevalence']:.3f}, n={ext.iloc[-1]['n_patients']})")

    # ── LOSO feasibility preview ──────────────────────────────────────────
    print("\n" + "-" * 72)
    print("LOSO FEASIBILITY PREVIEW")
    print("-" * 72)
    if len(usable) >= 2:
        print(f"LOSO folds available            : {len(usable)} "
              f"(one per usable site)")
        print(f"Smallest test fold              : "
              f"{usable['n_patients'].min()} patients")
        print(f"Largest test fold               : "
              f"{usable['n_patients'].max()} patients")
        # check each held-out site has both classes
        no_both = usable[(usable["n_high"] == 0) | (usable["n_low"] == 0)]
        if len(no_both):
            print(f"!! {len(no_both)} usable site(s) have only ONE class "
                  f"— AUROC undefined when held out: "
                  f"{no_both['tss'].tolist()}")
            print(f"   -> these become train-only sites")
        else:
            print("All usable sites contain BOTH classes  (AUROC computable)  OK")

    # ── DECISION ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("DECISION")
    print("=" * 72)

    enough_sites   = len(usable) >= MIN_USABLE_SITES
    sig_confound   = (not np.isnan(p_val)) and (p_val < ALPHA)

    if enough_sites and sig_confound:
        verdict = "GO"
        msg = ("Site confounding is present AND LOSO is feasible.\n"
               "  -> Proceed with the full study: 'Are FM benchmarks measuring\n"
               "     biology or batch effects?' This is the strong version.")
    elif enough_sites:
        verdict = "WEAK GO"
        msg = ("Enough sites for LOSO, but prevalence variation is not significant.\n"
               "  -> Study still works, reframed as cross-institutional\n"
               "     GENERALIZATION rather than label confounding.\n"
               "     Experiments A, C, D, E, F all still run.")
    else:
        verdict = "NO-GO"
        msg = (f"Only {len(usable)} usable sites (need {MIN_USABLE_SITES}).\n"
               "  -> Fall back to the committee-disagreement design instead.")

    print(f"\n  >>> {verdict} <<<\n")
    print("  " + msg.replace("\n", "\n  "))

    # ── Save artefacts ────────────────────────────────────────────────────
    site_csv = os.path.join(OUTPUT_DIR, "site_distribution.csv")
    site_tbl.to_csv(site_csv, index=False)

    pat_csv = os.path.join(OUTPUT_DIR, "patient_site_labels.csv")
    pat.to_csv(pat_csv, index=False)   # feeds Step 2 (LOSO fold construction)

    summary = {
        "n_sites_total": len(site_tbl),
        "n_sites_usable": len(usable),
        "n_patients_total": int(site_tbl["n_patients"].sum()),
        "n_patients_usable": int(usable["n_patients"].sum()) if len(usable) else 0,
        "overall_prevalence": round(float(overall_prev), 4),
        "chi2": None if np.isnan(chi2) else round(float(chi2), 3),
        "p_value": None if np.isnan(p_val) else float(p_val),
        "cramers_v": None if np.isnan(cramers_v) else round(float(cramers_v), 4),
        "verdict": verdict,
    }
    import json
    with open(os.path.join(OUTPUT_DIR, "step0_gonogo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved -> {site_csv}")
    print(f"  Saved -> {pat_csv}")
    print(f"  Saved -> {os.path.join(OUTPUT_DIR, 'step0_gonogo_summary.json')}")
    print("=" * 72)

    return site_tbl, summary


if __name__ == "__main__":
    site_tbl, summary = main()
