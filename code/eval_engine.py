#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_engine.py  —  Reusable evaluation harness for the prostate benchmark
==========================================================================
Build ONCE. Every encoder in Phases 1-4 calls this.

Architecture
------------
load_encoder_data()     → (X, y, stems, patient_ids, slides_df)
run_linear_probe_cv()   → oof_df   (out-of-fold predictions, all repeats × folds)
compute_metrics()       → dict     (single eval: AUROC, AUPR, MCC, bACC, F1, Sens, Spec)
bootstrap_ci()          → dict     (metric → {mean, ci_lower, ci_upper})
compute_ece()           → (float, list)   (ECE scalar + per-bin data for reliability curve)
plot_results()          → saves PNG figures to output_dir

Design principles
-----------------
* Standardisation fit on TRAIN only inside each fold — never leaks test statistics
* C selection via 3-fold inner patient-grouped CV — proper nested CV
* Bootstrap resamples PATIENTS not slides — respects the grouping structure
* Threshold: Youden's J on pooled OOF (maximises sens+spec jointly) — no 0.5 assumption
* Everything saved: OOF predictions CSV + metrics JSON + figures
  → every downstream comparison uses the same artefacts
"""

import os
import json
import warnings
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
    f1_score, confusion_matrix, roc_curve
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── C search space for inner-loop hyperparameter selection ─────────────────
C_RANGE = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
INNER_FOLDS = 3
BOOTSTRAP_N = 1000
ECE_BINS    = 10
RANDOM_SEED = 42


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def _load_h5_vector(path: str) -> np.ndarray:
    """Load a feature vector/matrix from an H5 file.
    Handles both (1, D) slide-level and (N, D) patch-level arrays.
    For slide encoders → returns (D,) squeezed vector.
    For patch encoders → returns (N, D) matrix (mean-pool happens upstream).
    """
    with h5py.File(path, "r") as f:
        key = "features" if "features" in f else next(iter(f.keys()))
        arr = f[key][:]
    return arr.squeeze() if arr.ndim == 2 and arr.shape[0] == 1 else arr


def load_encoder_data(
    manifest_path: str,
    inventory_path: str,
    folds_path: str,
    encoder_name: str,
    aggregation: str = "mean",   # "mean" | "max" | "none" (none = return raw bags)
    verbose: bool = True,
):
    """
    Load features for a single encoder and join with labels + frozen folds.

    Returns
    -------
    X          : np.ndarray  shape (n_slides, feat_dim)
    y          : np.ndarray  shape (n_slides,)  binary label
    stems      : list[str]   slide_stem identifiers
    patient_ids: np.ndarray  shape (n_slides,)  for grouping
    slides_df  : pd.DataFrame joined manifest rows for this encoder
    folds_df   : pd.DataFrame fold assignments for these slides (all repeats)
    """
    manifest  = pd.read_csv(manifest_path)
    inventory = pd.read_csv(inventory_path)
    folds     = pd.read_csv(folds_path)

    enc_inv = inventory[inventory["encoder"] == encoder_name].copy()
    if enc_inv.empty:
        raise ValueError(f"Encoder '{encoder_name}' not found in inventory.")

    # join: manifest (labels) ← inventory (paths) on slide_stem
    slides_df = manifest.merge(enc_inv[["slide_stem", "feat_path", "encoder_type"]],
                               on="slide_stem", how="inner")
    # restrict folds to the slides we actually have features for
    folds_df  = folds[folds["slide_stem"].isin(slides_df["slide_stem"])].copy()

    n = len(slides_df)
    if verbose:
        print(f"  [{encoder_name}] {n} slides, "
              f"{slides_df['patient_id'].nunique()} patients, "
              f"label balance: {slides_df['risk_binary'].value_counts().to_dict()}")

    # load features
    X_list, load_fails = [], []
    for _, row in slides_df.iterrows():
        try:
            arr = _load_h5_vector(row["feat_path"])
            if arr.ndim == 2:          # patch encoder → aggregate now
                arr = {"mean": arr.mean(0), "max": arr.max(0)}[aggregation]
            X_list.append(arr.astype(np.float32))
        except Exception as e:
            load_fails.append((row["slide_stem"], str(e)))
            X_list.append(None)

    if load_fails:
        print(f"  !! {len(load_fails)} H5 load failures (dropping):", load_fails[:5])
        good = [i for i, x in enumerate(X_list) if x is not None]
        slides_df = slides_df.iloc[good].reset_index(drop=True)
        X_list    = [X_list[i] for i in good]
        folds_df  = folds_df[folds_df["slide_stem"].isin(slides_df["slide_stem"])]

    X           = np.stack(X_list, axis=0)
    y           = slides_df["risk_binary"].values.astype(int)
    stems       = slides_df["slide_stem"].tolist()
    patient_ids = slides_df["patient_id"].values

    if verbose:
        print(f"  X shape: {X.shape}  |  dtype: {X.dtype}")

    return X, y, stems, patient_ids, slides_df, folds_df


# ════════════════════════════════════════════════════════════════════════════
# 2. INNER C SELECTION
# ════════════════════════════════════════════════════════════════════════════

def _select_best_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    C_range: list = C_RANGE,
) -> float:
    """
    3-fold patient-grouped inner CV to pick the best L2 regularisation strength.
    Uses AUROC as the selection criterion (most appropriate for imbalanced data).
    """
    inner_cv  = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                     random_state=RANDOM_SEED)
    C_scores  = defaultdict(list)

    for _, (ti, vi) in enumerate(inner_cv.split(X_train, y_train, groups=groups_train)):
        if len(np.unique(y_train[vi])) < 2:
            continue                   # skip degenerate inner splits
        for C in C_range:
            clf = LogisticRegression(
                C=C, max_iter=500, solver="lbfgs",
                class_weight="balanced", random_state=RANDOM_SEED
            )
            clf.fit(X_train[ti], y_train[ti])
            prob = clf.predict_proba(X_train[vi])[:, 1]
            C_scores[C].append(roc_auc_score(y_train[vi], prob))

    best_C   = max(C_scores, key=lambda c: np.mean(C_scores[c]), default=1.0)
    return float(best_C)


# ════════════════════════════════════════════════════════════════════════════
# 3. CV LOOP — LINEAR PROBE
# ════════════════════════════════════════════════════════════════════════════

def run_linear_probe_cv(
    X: np.ndarray,
    y: np.ndarray,
    stems: list,
    patient_ids: np.ndarray,
    folds_df: pd.DataFrame,
    C_range: list = C_RANGE,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run L2-regularised logistic regression across the FROZEN 5-fold × 3-repeat CV.

    Critical design choices
    -----------------------
    * StandardScaler fit on TRAIN split only — no test leakage
    * C selected via inner 3-fold patient-grouped CV on TRAIN split
    * class_weight='balanced' corrects for the 41/59 slide imbalance
    * Returns one OOF prediction row per slide per (repeat, fold)
    """
    stem_to_idx = {s: i for i, s in enumerate(stems)}
    oof_records = []
    repeats     = sorted(folds_df["repeat"].unique())
    total       = len(repeats) * folds_df["fold"].nunique()
    done        = 0

    for repeat in repeats:
        rep_df = folds_df[folds_df["repeat"] == repeat]
        for fold in sorted(rep_df["fold"].unique()):
            test_stems  = set(rep_df[rep_df["fold"] == fold ]["slide_stem"])
            train_stems = set(rep_df[rep_df["fold"] != fold ]["slide_stem"])

            test_idx  = [stem_to_idx[s] for s in test_stems  if s in stem_to_idx]
            train_idx = [stem_to_idx[s] for s in train_stems if s in stem_to_idx]

            if len(test_idx) == 0 or len(train_idx) == 0:
                continue

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx],  y[test_idx]
            g_tr       = patient_ids[train_idx]

            # --- standardise inside fold ---
            scaler   = StandardScaler()
            X_tr_s   = scaler.fit_transform(X_tr)
            X_te_s   = scaler.transform(X_te)

            # --- inner C selection ---
            best_C   = _select_best_C(X_tr_s, y_tr, g_tr, C_range)

            # --- final model ---
            clf = LogisticRegression(
                C=best_C, max_iter=1000, solver="lbfgs",
                class_weight="balanced", random_state=RANDOM_SEED
            )
            clf.fit(X_tr_s, y_tr)
            probs = clf.predict_proba(X_te_s)[:, 1]

            for local_i, global_i in enumerate(test_idx):
                oof_records.append({
                    "slide_stem": stems[global_i],
                    "patient_id": patient_ids[global_i],
                    "y_true":     int(y_te[local_i]),
                    "y_prob":     float(probs[local_i]),
                    "repeat":     repeat,
                    "fold":       fold,
                    "best_C":     best_C,
                })

            done += 1
            if verbose:
                fold_auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else float("nan")
                print(f"  rep {repeat} fold {fold}/{total//len(repeats)-1} "
                      f"| C={best_C:.3f} | test AUROC={fold_auc:.4f} "
                      f"| train/test={len(train_idx)}/{len(test_idx)}")

    return pd.DataFrame(oof_records)


# ════════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ════════════════════════════════════════════════════════════════════════════

def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Optimal threshold by Youden's J = max(sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = None,
) -> dict:
    """
    Full metric panel on a single (y_true, y_prob) pair.
    threshold: if None, uses Youden's J (the principled choice for imbalanced data).
    """
    if len(np.unique(y_true)) < 2:
        return {}

    thr   = threshold if threshold is not None else _youden_threshold(y_true, y_prob)
    y_hat = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auroc":    float(roc_auc_score(y_true, y_prob)),
        "aupr":     float(average_precision_score(y_true, y_prob)),
        "mcc":      float(matthews_corrcoef(y_true, y_hat)),
        "bacc":     float(balanced_accuracy_score(y_true, y_hat)),
        "f1":       float(f1_score(y_true, y_hat, zero_division=0)),
        "sens":     float(sensitivity),
        "spec":     float(specificity),
        "threshold":float(thr),
        "n":        int(len(y_true)),
        "n_pos":    int(y_true.sum()),
    }


# ════════════════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP CI  (patient-level resampling)
# ════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    oof_df: pd.DataFrame,
    n_boot: int = BOOTSTRAP_N,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """
    Bootstrap 95% CIs by resampling PATIENTS with replacement.

    Why patient-level?
    ------------------
    A patient's slides are NOT independent observations — they share the same
    underlying biology. Slide-level bootstrap inflates effective sample size and
    produces artificially narrow CIs. Patient-level bootstrap is the correct
    procedure when your unit of clinical interest is the patient.

    OOF predictions are pooled across all repeats × folds before bootstrapping,
    giving each slide one probability per (repeat, fold). We average per-slide
    before resampling to avoid counting a slide multiple times per bootstrap draw.
    """
    # Average OOF predictions across repeats × folds → one prob per slide
    slide_probs = (oof_df.groupby("slide_stem")
                         .agg(y_true=("y_true", "first"),
                              y_prob=("y_prob", "mean"),
                              patient_id=("patient_id", "first"))
                         .reset_index())

    patients = slide_probs["patient_id"].unique()
    rng      = np.random.default_rng(seed)

    # Compute the global Youden threshold on the full OOF pool once
    global_thr = _youden_threshold(slide_probs["y_true"].values,
                                   slide_probs["y_prob"].values)

    boot_metrics = []
    for b in range(n_boot):
        samp_patients = rng.choice(patients, size=len(patients), replace=True)
        rows = pd.concat(
            [slide_probs[slide_probs["patient_id"] == p] for p in samp_patients],
            ignore_index=True
        )
        if rows["y_true"].nunique() < 2:
            continue
        m = compute_metrics(rows["y_true"].values, rows["y_prob"].values,
                            threshold=global_thr)
        if m:
            boot_metrics.append(m)

    if verbose:
        print(f"  Bootstrap: {len(boot_metrics)}/{n_boot} valid draws")

    boot_df = pd.DataFrame(boot_metrics)
    ci = {}
    for col in [c for c in boot_df.columns if c not in ("threshold", "n", "n_pos")]:
        vals = boot_df[col].dropna().values
        ci[col] = {
            "mean":     float(np.mean(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
            "std":      float(np.std(vals)),
        }
    return ci, float(global_thr)


# ════════════════════════════════════════════════════════════════════════════
# 6. CALIBRATION
# ════════════════════════════════════════════════════════════════════════════

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = ECE_BINS,
) -> tuple:
    """
    Expected Calibration Error (ECE).

    Bins predicted probabilities into n_bins equal-width buckets.
    ECE = Σ (|bin| / N) × |avg_confidence - avg_accuracy|

    Returns (ece_scalar, bin_data_list) where bin_data has the info
    needed for a reliability diagram.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data  = []
    ece       = 0.0
    n         = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if lo == bin_edges[-2]:    # include right edge for last bin
            mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            bin_data.append({"mid": (lo + hi) / 2, "conf": None,
                             "acc": None, "count": 0})
            continue
        conf = float(y_prob[mask].mean())
        acc  = float(y_true[mask].mean())
        cnt  = int(mask.sum())
        ece += (cnt / n) * abs(conf - acc)
        bin_data.append({"mid": (lo + hi) / 2, "conf": conf, "acc": acc, "count": cnt})

    return float(ece), bin_data


# ════════════════════════════════════════════════════════════════════════════
# 7. PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_results(
    oof_df: pd.DataFrame,
    ci: dict,
    ece: float,
    bin_data: list,
    global_thr: float,
    encoder_name: str,
    output_dir: str,
):
    """
    Three-panel figure:
      (A) ROC curve with 95% CI band (OOF pooled)
      (B) Precision-Recall curve
      (C) Reliability diagram (calibration)
    """
    slide_probs = (oof_df.groupby("slide_stem")
                         .agg(y_true=("y_true", "first"),
                              y_prob=("y_prob", "mean"))
                         .reset_index())
    y_t = slide_probs["y_true"].values
    y_p = slide_probs["y_prob"].values

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"{encoder_name.upper()} — Linear Probe | "
                 f"AUROC={ci['auroc']['mean']:.3f} "
                 f"[{ci['auroc']['ci_lower']:.3f}–{ci['auroc']['ci_upper']:.3f}]",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── A: ROC ───────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    fpr_all, tpr_all, _ = roc_curve(y_t, y_p)
    ax.plot(fpr_all, tpr_all, lw=2, color="#2563EB",
            label=f"OOF (AUC={ci['auroc']['mean']:.3f})")
    ax.fill_between(fpr_all, tpr_all, alpha=0.08, color="#2563EB")
    # mark Youden threshold
    fpr_j, tpr_j, thr_j = roc_curve(y_t, y_p)
    idx_j = np.argmax(tpr_j - fpr_j)
    ax.scatter([fpr_j[idx_j]], [tpr_j[idx_j]], s=80, color="#DC2626",
               zorder=5, label=f"Youden thr={global_thr:.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set(xlabel="FPR (1-Specificity)", ylabel="TPR (Sensitivity)",
           title="ROC Curve", xlim=[0, 1], ylim=[0, 1])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── B: Precision-Recall ───────────────────────────────────────────────
    from sklearn.metrics import precision_recall_curve
    ax2 = fig.add_subplot(gs[1])
    prec, rec, _ = precision_recall_curve(y_t, y_p)
    ax2.plot(rec, prec, lw=2, color="#16A34A",
             label=f"OOF (AUPR={ci['aupr']['mean']:.3f})")
    ax2.fill_between(rec, prec, alpha=0.08, color="#16A34A")
    baseline = y_t.mean()
    ax2.axhline(baseline, color="k", linestyle="--", lw=0.8, alpha=0.5,
                label=f"Baseline (prev={baseline:.2f})")
    ax2.set(xlabel="Recall (Sensitivity)", ylabel="Precision",
            title="Precision-Recall", xlim=[0, 1], ylim=[0, 1])
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── C: Reliability diagram ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    bins_with_data = [b for b in bin_data if b["conf"] is not None]
    if bins_with_data:
        mids  = [b["mid"]  for b in bins_with_data]
        confs = [b["conf"] for b in bins_with_data]
        accs  = [b["acc"]  for b in bins_with_data]
        cnts  = [b["count"]for b in bins_with_data]
        max_cnt = max(cnts)
        widths  = [0.08] * len(mids)
        bars = ax3.bar(mids, accs, width=widths, alpha=0.5,
                       color="#7C3AED", label="Actual frequency")
        ax3.bar(mids, confs, width=widths, alpha=0.25,
                color="#F59E0B", label="Mean confidence")
        for mid, cnt in zip(mids, cnts):
            ax3.text(mid, 0.02, str(cnt), ha="center", va="bottom",
                     fontsize=6, color="gray")
    ax3.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax3.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
            title=f"Reliability Diagram (ECE={ece:.4f})",
            xlim=[0, 1], ylim=[0, 1])
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{encoder_name}_linear_probe_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# 8. RESULTS SAVING
# ════════════════════════════════════════════════════════════════════════════

def save_results(
    ci: dict,
    global_thr: float,
    ece: float,
    oof_df: pd.DataFrame,
    encoder_name: str,
    encoder_type: str,
    method: str,
    output_dir: str,
    extra: dict = None,
):
    """
    Save three artefacts:
      {encoder}_{method}_metrics.json  — CI table (the definitive result row)
      {encoder}_{method}_oof.csv       — full OOF predictions (for DeLong tests later)
      {encoder}_{method}_summary.txt   — human-readable table for quick reading
    """
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "encoder":      encoder_name,
        "encoder_type": encoder_type,
        "method":       method,
        "n_slides":     int(oof_df["slide_stem"].nunique()),
        "n_patients":   int(oof_df["patient_id"].nunique()),
        "threshold":    global_thr,
        "ece":          ece,
        "metrics":      ci,
    }
    if extra:
        result.update(extra)

    json_path = os.path.join(output_dir, f"{encoder_name}_{method}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    oof_path = os.path.join(output_dir, f"{encoder_name}_{method}_oof.csv")
    oof_df.to_csv(oof_path, index=False)

    # human-readable summary
    lines = [
        f"{'='*60}",
        f"RESULTS: {encoder_name.upper()} — {method}",
        f"{'='*60}",
        f"{'Metric':<12}  {'Mean':>7}  {'95% CI':>20}",
        f"{'-'*44}",
    ]
    metric_labels = {
        "auroc": "AUROC", "aupr": "AUPR", "mcc": "MCC",
        "bacc": "Bal-ACC", "f1": "F1", "sens": "Sensitivity", "spec": "Specificity"
    }
    for key, label in metric_labels.items():
        if key in ci:
            v = ci[key]
            lines.append(f"{label:<12}  {v['mean']:>7.4f}  "
                         f"[{v['ci_lower']:.4f} – {v['ci_upper']:.4f}]")
    lines += [f"{'-'*44}",
              f"ECE (calibration): {ece:.4f}",
              f"Youden threshold:  {global_thr:.4f}",
              f"N slides:          {result['n_slides']}",
              f"N patients:        {result['n_patients']}",
              f"{'='*60}"]
    summary = "\n".join(lines)
    print("\n" + summary)

    txt_path = os.path.join(output_dir, f"{encoder_name}_{method}_summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary + "\n")

    print(f"\n  Saved → {json_path}")
    print(f"  Saved → {oof_path}")
    print(f"  Saved → {txt_path}")
    return result
