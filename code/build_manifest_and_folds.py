#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 — Data contract & fold freezing
========================================
Builds the single source of truth for the downstream benchmark:

  1. feature_inventory.csv   one row per (slide, encoder): path, type, feat_dim, n_patches
  2. manifest_slides.csv     one row per slide: patient_id, labels, encoder coverage
  3. folds_frozen.csv        one row per slide: patient-grouped CV fold assignment
  4. phase0_integrity_report.md   human-readable summary of all checks

DESIGN DECISIONS (the senior-researcher reasons, so future-you remembers):
  * BINARY LABEL comes from the high_risk/ vs low_risk/ folder a slide's features
    live in. That is how the data was organized, so it is authoritative and needs
    no clinical file. A patient can legitimately have slides in BOTH folders
    (per-slide Gleason labels) — that is expected, not a bug.
  * ISUP MULTICLASS is OPTIONAL. If you supply a clinical CSV with primary/secondary
    Gleason patterns, the script computes Gleason sum -> ISUP grade group, and (if a
    folder label also exists) cross-checks the derived binary label against the folder.
  * FOLDS are frozen ONCE and reused by every model so all comparisons are on
    identical splits. Splitting is done at the PATIENT level (collapse to one row per
    patient, stratify, then map back to every slide) — this guarantees no patient's
    slides are split across folds (the #1 leakage trap) by construction.
  * Reads are bounded: feat_dim is read once per encoder; n_patches optionally read
    once per slide from a reference patch encoder (Drive/FUSE reads are slow).

Run in Colab after mounting Drive, or anywhere the feature tree is visible.
"""

import os
import re
import glob
import json
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

# ============================================================================
# CONFIG  —  EDIT THESE
# ============================================================================
# Root that directly contains the risk-group folders (high_risk/ and low_risk/).
DRIVE_OUTPUT = os.environ.get("PCA_ROOT", "/content/drive/MyDrive/ProstateCancer") + "/organized_output"
# Folder names under DRIVE_OUTPUT that hold the two classes (confirmed).
RISK_GROUP_DIRS = {"high_risk": 1, "low_risk": 0}   # name -> binary label

# FAST FIRST PASS. With this True the script only walks the tree and prints a
# discovery summary (encoders, slides, patients) WITHOUT reading h5 shapes or
# building folds. Run it True, eyeball the summary, then set False for the real build.
DISCOVER_ONLY = True

# Where the manifest/folds/report are written.
OUTPUT_DIR = os.environ.get("PCA_ROOT", "/content/drive/MyDrive/ProstateCancer") + "/manifest"
# Names that identify SLIDE encoders (everything else found is treated as patch).
SLIDE_ENCODER_NAMES = {"titan", "chief", "madeleine", "feather"}

# CV settings (plan: 5-fold, repeated x3).
N_SPLITS = 5
N_REPEATS = 3
SEED = 42

# Read n_patches per slide? Costs one extra Drive read per slide (slow on FUSE).
READ_NPATCHES = True
# Reference patch encoder to read n_patches from (first match found is used if None).
NPATCHES_REF_ENCODER = None

# ---- OPTIONAL: ISUP multiclass via a clinical Gleason file -----------------
# Set CLINICAL_CSV = None to skip ISUP entirely (binary-only run).
CLINICAL_CSV = None   # e.g. "/content/drive/MyDrive/ProstateCancer/gleason_per_slide.csv"
# Column names in that CSV (only used if CLINICAL_CSV is set):
COL_SLIDE_ID   = "slide_id"          # must match the slide stem (or a substring of it)
COL_GLEASON_1  = "primary_pattern"   # integer 1..5
COL_GLEASON_2  = "secondary_pattern" # integer 1..5
# If your file already has a grade-group or sum column, set these instead and the
# script will use them directly (leave as None to compute from patterns):
COL_ISUP_GG    = None   # e.g. "isup_grade"  (1..5)
COL_GLEASON_SUM = None  # e.g. "gleason_score" (6..10)

# Binary cutoff: a slide is HIGH risk if ISUP grade group >= this value.
# NOTE: this is only used to CROSS-CHECK against the folder label, and to define
# binary from clinical data if you ever run without the folder structure.
# Common choices: 3 (>= 4+3), or 2 (>= 3+4). The script prints the resulting class
# balance so you can match it to your known 187 high / 260 low split.
HIGH_RISK_MIN_GG = 3
# ============================================================================


TCGA_PATIENT_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
# coords-config folder, e.g. 20x_512px_0px_overlap (also matches 20.0x_512px...)
COORDS_RE = re.compile(r"(\d+\.?\d*x_\d+px[A-Za-z0-9_]*)", re.IGNORECASE)
# directories that hold coordinates / thumbnails / segmentation rather than features
SKIP_DIRS = {"patches", "thumbnails", "contours", "contours_geojson", "masks",
             "visualization", "tissue_seg", "segmentation"}

# Canonical encoder names (from the notebook's PATCH_ENCODERS / SLIDE_ENCODERS config).
# The reorganization baked the encoder into the FILENAME as
# features_{encoder}{_slidesuffix}.h5 ; since encoder names contain underscores we
# disambiguate by matching against this known list (longest match wins).
PATCH_ENCODERS_KNOWN = [
    "virchow", "virchow2", "phikon", "phikon_v2",
    "hoptimus0", "hoptimus1", "h0-mini", "openmidnight", "midnight12k",
    "gpfm", "hibou_l",
    "kaiko-vitb8", "kaiko-vitb16", "kaiko-vits8", "kaiko-vits16", "kaiko-vitl14",
    "lunit-vits8", "genbio-pathfm",
    "uni_v1", "uni_v2", "gigapath", "resnet50",
    "conch_v1", "conch_v15", "musk", "ctranspath",
]
SLIDE_ENCODERS_KNOWN = ["titan", "chief", "madeleine", "feather"]
KNOWN_ENCODERS = sorted(set(PATCH_ENCODERS_KNOWN) | set(SLIDE_ENCODERS_KNOWN),
                        key=len, reverse=True)


def parse_encoder_and_suffix(rest: str):
    """rest = filename minus the features_/slide_features_ prefix and the .h5 extension.
    Returns (encoder, slide_suffix) via longest known-name match, else (None, None)."""
    cands = [e for e in KNOWN_ENCODERS
             if rest == e or (rest.startswith(e) and rest[len(e):len(e) + 1] == "_")]
    if not cands:
        return None, None
    enc = max(cands, key=len)
    return enc, rest[len(enc):].lstrip("_")


def extract_patient_id(slide_stem: str) -> str:
    """TCGA patient = first 12 chars of the barcode, e.g. TCGA-EJ-A46D."""
    m = TCGA_PATIENT_RE.search(slide_stem)
    return m.group(1).upper() if m else slide_stem


def gleason_sum_to_isup(primary: int, secondary: int) -> int:
    """ISUP 2014 / Epstein grade groups from primary+secondary Gleason patterns."""
    p, s = int(primary), int(secondary)
    g = p + s
    if g <= 6:
        return 1
    if (p, s) == (3, 4):
        return 2
    if (p, s) == (4, 3):
        return 3
    if g == 8:
        return 4
    if g >= 9:
        return 5
    # Fallback for unusual combos (e.g. 3+5=8 handled above; 5+3 -> 8 -> GG4).
    return 4 if g == 8 else (5 if g >= 9 else 1)


def isup_to_binary(gg: int, high_min_gg: int) -> int:
    return 1 if int(gg) >= high_min_gg else 0


def _read_h5_shape(path: str):
    """Return (n_rows, feat_dim) for a TRIDENT feature h5, or (None, None) on failure."""
    try:
        with h5py.File(path, "r") as f:
            key = "features" if "features" in f else next(iter(f.keys()))
            shp = f[key].shape
        if len(shp) == 1:
            return 1, int(shp[0])
        return int(shp[0]), int(shp[-1])
    except Exception as e:
        print(f"    ! could not read shape from {os.path.basename(path)}: {e}")
        return None, None


def discover_features(drive_output: str, risk_group_dirs: dict):
    """
    Walk each risk-group tree. In the reorganized (flattened) layout the H5 files sit
    directly inside the coords folder, with the encoder baked into the FILENAME:
        {risk}/{patient}/{coords_name}/features_{encoder}{_suffix}.h5        (patch)
        {risk}/{patient}/{coords_name}/slide_features_{encoder}{_suffix}.h5  (slide)
    _suffix is present only for patients with >1 slide in that risk group.
    Returns (long DataFrame one row per (slide,encoder), list_of_unmatched_filenames).
    """
    rows, unmatched = [], []
    for rg_name, rg_label in risk_group_dirs.items():
        rg_dir = os.path.join(drive_output, rg_name)
        if not os.path.isdir(rg_dir):
            print(f"!! risk-group dir not found: {rg_dir}")
            continue
        for dirpath, dirs, files in os.walk(rg_dir):
            if os.path.basename(dirpath).lower() in SKIP_DIRS:
                dirs[:] = []                       # don't descend into seg/patches dirs
                continue
            h5s = [f for f in files if f.endswith(".h5")]
            if not h5s:
                continue
            coords_name = os.path.basename(dirpath)         # coords folder is the parent
            mp = TCGA_PATIENT_RE.search(dirpath)            # patient = folder above coords
            patient = mp.group(1).upper() if mp else os.path.basename(os.path.dirname(dirpath))
            for fn in h5s:
                stem = fn[:-3]
                if stem.startswith("slide_features_"):
                    rest, etype = stem[len("slide_features_"):], "slide"
                elif stem.startswith("features_"):
                    rest, etype = stem[len("features_"):], "patch"
                else:
                    continue                                # patches.h5 etc. — not features
                enc, suffix = parse_encoder_and_suffix(rest)
                if enc is None:
                    unmatched.append(fn)
                    continue
                # globally-unique physical-slide id (risk distinguishes both-group patients)
                slide_stem = f"{rg_name}/{patient}" + (f"/{suffix}" if suffix else "")
                rows.append({
                    "slide_stem": slide_stem,
                    "patient_id": patient,
                    "encoder": enc,
                    "encoder_type": etype,
                    "coords_name": coords_name,
                    "risk_group": rg_name,
                    "risk_binary": rg_label,
                    "slide_suffix": suffix,
                    "feat_path": os.path.join(dirpath, fn),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No feature .h5 matched under {drive_output}. "
            f"Check DRIVE_OUTPUT / RISK_GROUP_DIRS / the encoder list."
        )
    return df, unmatched


def print_discovery_summary(inv: pd.DataFrame, unmatched: list):
    """Summary for the DISCOVER_ONLY eyeball pass (encoder now parsed from filename)."""
    print("\n" + "-" * 60)
    print("DISCOVERY SUMMARY")
    print("-" * 60)
    print(f"feature files (rows) : {len(inv)}")
    print(f"unique slides        : {inv['slide_stem'].nunique()}")
    print(f"unique patients      : {inv['patient_id'].nunique()}")
    print(f"files per risk group : {inv['risk_group'].value_counts().to_dict()}")
    g = inv.groupby("encoder").agg(
        etype=("encoder_type", lambda s: s.iloc[0]),
        n_files=("feat_path", "size"),
        coords=("coords_name", lambda s: sorted(set(s))),
    )
    n_patch = int((g["etype"] == "patch").sum())
    n_slide = int((g["etype"] == "slide").sum())
    print(f"\nencoders found       : {len(g)}  ({n_patch} patch, {n_slide} slide)")
    for enc, r in g.sort_values(["etype", "n_files"], ascending=[True, False]).iterrows():
        print(f"  [{r['etype']:5}] {enc:16} files={r['n_files']:>5}  coords={r['coords']}")
    missing = sorted((set(PATCH_ENCODERS_KNOWN) | set(SLIDE_ENCODERS_KNOWN)) - set(g.index))
    if missing:
        print(f"\n!! expected encoders NOT found on disk: {missing}")
    if unmatched:
        print(f"\n!! {len(unmatched)} file(s) whose encoder didn't match the known list "
              f"(examples: {sorted(set(unmatched))[:8]})")


def add_feature_dims(inv: pd.DataFrame) -> pd.DataFrame:
    """feat_dim + encoder_type once per encoder (cheap); n_patches per slide."""
    # feat_dim and shape-truthed type: one sample per encoder
    dim_map, type_map = {}, {}
    for enc, grp in inv.groupby("encoder"):
        n, d = _read_h5_shape(grp["feat_path"].iloc[0])
        dim_map[enc] = d
        if enc.lower() in SLIDE_ENCODER_NAMES or (n is not None and n == 1):
            type_map[enc] = "slide"      # one vector per slide
        else:
            type_map[enc] = "patch"      # bag of patch vectors
    inv["feat_dim"] = inv["encoder"].map(dim_map)
    inv["encoder_type"] = inv["encoder"].map(type_map)

    inv["n_patches"] = np.nan
    if READ_NPATCHES:
        patch_inv = inv[inv["encoder_type"] == "patch"]
        ref_enc = NPATCHES_REF_ENCODER
        if ref_enc is None and not patch_inv.empty:
            ref_enc = patch_inv["encoder"].value_counts().idxmax()
        ref = patch_inv[patch_inv["encoder"] == ref_enc]
        npatch_map = {}
        for _, r in ref.iterrows():
            n, d = _read_h5_shape(r["feat_path"])
            npatch_map[r["slide_stem"]] = n
        inv["n_patches"] = inv["slide_stem"].map(npatch_map)
        print(f"  n_patches read from reference encoder: {ref_enc}")
    return inv


def load_clinical(inv: pd.DataFrame):
    """Optionally attach per-slide ISUP grade group. Returns slide->dict or None."""
    if not CLINICAL_CSV:
        return None
    clin = pd.read_csv(CLINICAL_CSV)
    # Map clinical slide IDs to our slide stems (substring match is robust to barcodes).
    stems = inv["slide_stem"].unique()
    out = {}
    for _, row in clin.iterrows():
        sid = str(row[COL_SLIDE_ID])
        match = next((s for s in stems if sid in s or s in sid), None)
        if match is None:
            continue
        if COL_ISUP_GG and not pd.isna(row.get(COL_ISUP_GG)):
            gg = int(row[COL_ISUP_GG])
            gsum = None
        elif COL_GLEASON_SUM and not pd.isna(row.get(COL_GLEASON_SUM)):
            gsum = int(row[COL_GLEASON_SUM])
            gg = 1 if gsum <= 6 else (4 if gsum == 8 else (5 if gsum >= 9 else 2))
        else:
            p, s = row[COL_GLEASON_1], row[COL_GLEASON_2]
            if pd.isna(p) or pd.isna(s):
                continue
            gsum = int(p) + int(s)
            gg = gleason_sum_to_isup(p, s)
        out[match] = {"gleason_sum": gsum, "isup_gg": gg,
                      "risk_from_clin": isup_to_binary(gg, HIGH_RISK_MIN_GG)}
    print(f"  clinical labels matched: {len(out)} / {len(stems)} slides")
    return out


def build_slide_manifest(inv: pd.DataFrame, clin: dict):
    """Collapse the (slide,encoder) inventory to one row per slide."""
    g = inv.groupby("slide_stem")
    recs = []
    for stem, grp in g:
        encs = sorted(grp["encoder"].unique())
        patch_encs = sorted(grp[grp.encoder_type == "patch"]["encoder"].unique())
        slide_encs = sorted(grp[grp.encoder_type == "slide"]["encoder"].unique())
        rgroups = sorted(grp["risk_group"].unique())
        rec = {
            "slide_stem": stem,
            "patient_id": grp["patient_id"].iloc[0],
            "risk_group": rgroups[0] if len(rgroups) == 1 else "|".join(rgroups),
            "risk_binary": int(grp["risk_binary"].iloc[0]),
            "n_patches": grp["n_patches"].dropna().iloc[0] if grp["n_patches"].notna().any() else np.nan,
            "n_encoders_present": len(encs),
            "n_patch_encoders": len(patch_encs),
            "n_slide_encoders": len(slide_encs),
            "patch_encoders_present": ",".join(patch_encs),
            "slide_encoders_present": ",".join(slide_encs),
        }
        if clin and stem in clin:
            rec.update(clin[stem])
        recs.append(rec)
    man = pd.DataFrame(recs).sort_values("slide_stem").reset_index(drop=True)
    return man


def freeze_folds(man: pd.DataFrame):
    """
    Patient-level StratifiedKFold, repeated. One row per patient (most-severe label
    used as the stratification target), mapped back to every slide. This guarantees
    all of a patient's slides land in the same fold (no leakage) by construction.
    """
    strat_col = "isup_gg" if "isup_gg" in man.columns and man["isup_gg"].notna().any() else "risk_binary"
    print(f"  stratifying folds on: {strat_col} (most-severe per patient)")

    # one row per patient, label = most severe (max) across that patient's slides
    pat = (man.groupby("patient_id")[strat_col].max()
              .reset_index().rename(columns={strat_col: "strat"}))

    fold_records = []
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
        pat_fold = np.empty(len(pat), dtype=int)
        for fold, (_, test_idx) in enumerate(skf.split(pat["patient_id"], pat["strat"])):
            pat_fold[test_idx] = fold
        pmap = dict(zip(pat["patient_id"], pat_fold))
        tmp = man[["slide_stem", "patient_id", "risk_binary"]].copy()
        if "isup_gg" in man.columns:
            tmp["isup_gg"] = man["isup_gg"].values
        tmp["repeat"] = rep
        tmp["fold"] = tmp["patient_id"].map(pmap)
        fold_records.append(tmp)
    folds = pd.concat(fold_records, ignore_index=True)
    return folds, strat_col


def integrity_report(inv, man, folds, strat_col):
    lines = ["# Phase 0 — Integrity Report\n"]
    encs = sorted(inv["encoder"].unique())
    patch_encs = sorted(inv[inv.encoder_type == "patch"]["encoder"].unique())
    slide_encs = sorted(inv[inv.encoder_type == "slide"]["encoder"].unique())

    lines.append(f"- Slides: **{man['slide_stem'].nunique()}**, "
                 f"Patients: **{man['patient_id'].nunique()}**")
    lines.append(f"- Encoders found: **{len(encs)}** "
                 f"({len(patch_encs)} patch, {len(slide_encs)} slide)")
    lines.append(f"  - patch: {patch_encs}")
    lines.append(f"  - slide: {slide_encs}")

    # class balance
    vc = man["risk_binary"].value_counts().to_dict()
    lines.append(f"\n## Binary class balance (from folder)\n"
                 f"- high (1): **{vc.get(1,0)}**, low (0): **{vc.get(0,0)}**")
    if "isup_gg" in man.columns and man["isup_gg"].notna().any():
        gg = man["isup_gg"].value_counts().sort_index().to_dict()
        lines.append(f"- ISUP GG distribution: {gg}")
        # cross-check folder vs clinical-derived binary
        if "risk_from_clin" in man.columns:
            chk = man.dropna(subset=["risk_from_clin"])
            mism = chk[chk["risk_binary"] != chk["risk_from_clin"]]
            lines.append(f"- Folder-vs-clinical binary mismatches "
                         f"(cutoff GG>={HIGH_RISK_MIN_GG}): **{len(mism)}** "
                         f"of {len(chk)} checked")
            if len(mism):
                lines.append(f"  - mismatched slides (first 10): "
                             f"{mism['slide_stem'].head(10).tolist()}")
                lines.append(f"  - ACTION: adjust HIGH_RISK_MIN_GG until this is ~0, "
                             f"OR trust the folder label and treat clinical as ISUP-only.")

    # encoder coverage gaps
    full = man[man["n_encoders_present"] == len(encs)]
    partial = man[man["n_encoders_present"] < len(encs)]
    lines.append(f"\n## Encoder coverage\n"
                 f"- Slides with ALL {len(encs)} encoders: **{len(full)}**")
    lines.append(f"- Slides MISSING some encoders: **{len(partial)}**")
    if len(partial):
        # which encoders are most often missing
        miss_counter = defaultdict(int)
        present_by_slide = inv.groupby("slide_stem")["encoder"].apply(set)
        allset = set(encs)
        for stem in partial["slide_stem"]:
            for e in (allset - present_by_slide.get(stem, set())):
                miss_counter[e] += 1
        worst = sorted(miss_counter.items(), key=lambda x: -x[1])[:15]
        lines.append(f"- Most-missing encoders (encoder: #slides missing): {worst}")
        lines.append(f"- partial slides (first 10): {partial['slide_stem'].head(10).tolist()}")

    # feat dims
    dims = inv.drop_duplicates("encoder").set_index("encoder")["feat_dim"].to_dict()
    lines.append(f"\n## Feature dimensions per encoder\n{json.dumps(dims, indent=2)}")

    # patients spanning both risk groups (expected with per-slide labels)
    spanning = (man.groupby("patient_id")["risk_binary"].nunique())
    both = spanning[spanning > 1].index.tolist()
    lines.append(f"\n## Patients with slides in BOTH risk groups (expected, not a bug)\n"
                 f"- count: **{len(both)}** patients")
    if both:
        lines.append(f"- examples (first 10): {both[:10]}")

    # n_patches sanity
    if man["n_patches"].notna().any():
        np_ = man["n_patches"].dropna()
        zeros = int((np_ == 0).sum())
        lines.append(f"\n## n_patches\n"
                     f"- min/median/max: {int(np_.min())} / {int(np_.median())} / {int(np_.max())}")
        lines.append(f"- slides with ZERO patches (drop these!): **{zeros}**")

    # fold balance
    lines.append(f"\n## Frozen folds (stratified on {strat_col})\n"
                 f"- {N_SPLITS}-fold x {N_REPEATS} repeats, grouped by patient")
    rep0 = folds[folds["repeat"] == 0]
    bal = rep0.groupby("fold")["risk_binary"].agg(["sum", "count"])
    bal["high_frac"] = (bal["sum"] / bal["count"]).round(3)
    lines.append("- repeat 0 per-fold balance (high / total / frac):")
    for f, r in bal.iterrows():
        lines.append(f"  - fold {f}: {int(r['sum'])} / {int(r['count'])} / {r['high_frac']}")

    # leakage assertion: a patient's slides must stay in ONE fold WITHIN each repeat
    leak = (folds.groupby(["repeat", "patient_id"])["fold"].nunique() > 1).sum()
    lines.append(f"\n## Leakage check\n"
                 f"- (repeat, patient) groups whose slides span >1 fold (must be 0): **{int(leak)}**")

    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[1/5] Discovering feature files (recursive walk)...")
    inv, unmatched = discover_features(DRIVE_OUTPUT, RISK_GROUP_DIRS)
    print(f"  found {len(inv)} (slide,encoder) rows, "
          f"{inv['slide_stem'].nunique()} slides, {inv['encoder'].nunique()} encoders")
    print_discovery_summary(inv, unmatched)

    if DISCOVER_ONLY:
        print("\nDISCOVER_ONLY=True → stopping here. If the summary looks right, "
              "set DISCOVER_ONLY=False and re-run to build the manifest + folds.")
        return

    print("[2/5] Reading feature dimensions...")
    inv = add_feature_dims(inv)

    print("[3/5] Loading clinical labels (optional)...")
    clin = load_clinical(inv)

    print("[4/5] Building slide manifest...")
    man = build_slide_manifest(inv, clin)

    print("[5/5] Freezing CV folds...")
    folds, strat_col = freeze_folds(man)

    # write artifacts
    inv_path   = os.path.join(OUTPUT_DIR, "feature_inventory.csv")
    man_path   = os.path.join(OUTPUT_DIR, "manifest_slides.csv")
    fold_path  = os.path.join(OUTPUT_DIR, "folds_frozen.csv")
    rep_path   = os.path.join(OUTPUT_DIR, "phase0_integrity_report.md")
    inv.to_csv(inv_path, index=False)
    man.to_csv(man_path, index=False)
    folds.to_csv(fold_path, index=False)
    report = integrity_report(inv, man, folds, strat_col)
    with open(rep_path, "w") as f:
        f.write(report)

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    print(f"\nWrote:\n  {inv_path}\n  {man_path}\n  {fold_path}\n  {rep_path}")


if __name__ == "__main__":
    main()
