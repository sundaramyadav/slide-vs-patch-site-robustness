#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
robustness_report — per-encoder metric helper used by build_leaderboard.py
==========================================================================
Internal helper. Given slide embeddings, labels, patient IDs and site labels for
one encoder, it computes the quantities reported in the paper:

    site leakage         chance-normalised balanced accuracy of a site probe
    Protocol A AUROC     patient-grouped cross-validation
    Protocol B AUROC     leave-one-site-out
    generalization gap   Protocol A - Protocol B
    calibration (ECE)    10-bin expected calibration error

This is not a standalone benchmark or an evaluation framework, and it is not a
contribution of the paper; it exists so that build_leaderboard.py computes each
encoder's numbers through one code path. For a published, peer-reviewed
robustness benchmark for pathology foundation models, see PathoROB
(Kömen et al., Nature Communications 17:5218, 2026).
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

__version__ = "1.0.0"


class RobustnessReport:
    """Audit a slide-embedding representation for site confounding."""

    def __init__(self, X, y, patient_ids, site_labels, encoder_name="encoder",
                 min_patients_per_site=15, C=0.01, n_folds=5, seed=42,
                 verbose=True):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y).astype(int)
        self.pids = np.asarray(patient_ids)
        self.sites = np.asarray([str(s) for s in site_labels])
        self.name = encoder_name
        self.min_pat = min_patients_per_site
        self.C = C
        self.n_folds = n_folds
        self.seed = seed
        self.verbose = verbose
        self.report = {}

        if not (len(self.X) == len(self.y) == len(self.pids) == len(self.sites)):
            raise ValueError("X, y, patient_ids and site_labels must be the same length")

        # sites with enough patients to be usable as held-out folds
        pat_site = pd.DataFrame({"p": self.pids, "s": self.sites}).drop_duplicates()
        cnt = pat_site["s"].value_counts()
        self.usable = sorted(cnt[cnt >= self.min_pat].index.tolist())

    # ── helpers ──────────────────────────────────────────────────────────
    def _log(self, m):
        if self.verbose:
            print(m)

    def _fit_predict(self, Xtr, ytr, Xte):
        sc = StandardScaler()
        clf = LogisticRegression(C=self.C, max_iter=1000, solver="lbfgs",
                                 class_weight="balanced", random_state=self.seed)
        clf.fit(sc.fit_transform(Xtr), ytr)
        return clf.predict_proba(sc.transform(Xte))[:, 1]

    @staticmethod
    def _site_standardize(X, sites):
        """TRANSDUCTIVE: align per-site mean and variance, restore grand stats."""
        X = X.astype(np.float64)
        gm, gs = X.mean(0), X.std(0) + 1e-8
        Xh = X.copy()
        for s in np.unique(sites):
            m = sites == s
            mu = Xh[m].mean(0, keepdims=True)
            sd = Xh[m].std(0, keepdims=True) + 1e-8
            Xh[m] = (Xh[m] - mu) / sd * gs + gm
        return Xh

    @staticmethod
    def _ece(y, p, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        e, n = 0.0, len(y)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
            if m.sum():
                e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
        return float(e)

    # ── 1. site leakage ──────────────────────────────────────────────────
    def site_leakage(self, X=None):
        X = self.X if X is None else X
        keep = np.isin(self.sites, self.usable)
        if keep.sum() < 30 or len(self.usable) < 2:
            return {"site_leakage": np.nan, "site_bacc": np.nan, "chance": np.nan}
        Xs, ps, ss = X[keep], self.pids[keep], self.sites[keep]
        uq = sorted(np.unique(ss)); s2i = {s: i for i, s in enumerate(uq)}
        ys = np.array([s2i[s] for s in ss])
        chance = 1.0 / len(uq)
        skf = StratifiedGroupKFold(n_splits=min(self.n_folds, len(uq)),
                                   shuffle=True, random_state=self.seed)
        pred = np.full(len(ys), -1)
        for tr, te in skf.split(Xs, ys, groups=ps):
            sc = StandardScaler()
            clf = LogisticRegression(C=1.0, max_iter=1500, solver="lbfgs",
                                     class_weight="balanced", random_state=self.seed)
            clf.fit(sc.fit_transform(Xs[tr]), ys[tr])
            pred[te] = clf.predict(sc.transform(Xs[te]))
        m = pred >= 0
        b = balanced_accuracy_score(ys[m], pred[m])
        return {"site_leakage": float((b - chance) / (1 - chance)),
                "site_bacc": float(b), "chance": float(chance),
                "n_sites": len(uq)}

    # ── 2. random patient-grouped CV (the optimistic benchmark) ──────────
    def random_cv(self, X=None):
        X = self.X if X is None else X
        skf = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True,
                                   random_state=self.seed)
        prob = np.full(len(self.y), np.nan)
        for tr, te in skf.split(X, self.y, groups=self.pids):
            if len(np.unique(self.y[tr])) < 2:
                continue
            prob[te] = self._fit_predict(X[tr], self.y[tr], X[te])
        m = ~np.isnan(prob)
        if m.sum() < 10 or len(np.unique(self.y[m])) < 2:
            return {"random_cv_auroc": np.nan, "random_cv_ece": np.nan}
        return {"random_cv_auroc": float(roc_auc_score(self.y[m], prob[m])),
                "random_cv_ece": self._ece(self.y[m], prob[m])}

    # ── 3. LOSO (the honest benchmark) ──────────────────────────────────
    def loso(self, X=None):
        X = self.X if X is None else X
        probs, trues, per_site = [], [], []
        for s in self.usable:
            te = np.where(self.sites == s)[0]
            tr = np.where(self.sites != s)[0]
            if len(np.unique(self.y[tr])) < 2 or len(np.unique(self.y[te])) < 2:
                continue
            p = self._fit_predict(X[tr], self.y[tr], X[te])
            probs.append(p); trues.append(self.y[te])
            per_site.append({"site": s, "n": int(len(te)),
                             "auroc": float(roc_auc_score(self.y[te], p)),
                             "prevalence": float(self.y[te].mean())})
        if not probs:
            return {"loso_auroc": np.nan, "loso_ece": np.nan, "per_site": []}
        yt = np.concatenate(trues); yp = np.concatenate(probs)
        return {"loso_auroc": float(roc_auc_score(yt, yp)),
                "loso_ece": self._ece(yt, yp),
                "per_site": sorted(per_site, key=lambda d: d["auroc"])}

    # ── 4. run everything ───────────────────────────────────────────────
    def run(self):
        self._log(f"\nrobustness report v{__version__} — {self.name}")
        self._log(f"  {len(self.X)} slides · {len(np.unique(self.pids))} patients · "
                  f"{len(self.usable)} usable sites (of {len(np.unique(self.sites))})")
        if len(self.usable) < 3:
            self._log("  !! fewer than 3 usable sites — LOSO not meaningful")

        self._log("  [1/5] site leakage ...")
        leak = self.site_leakage()
        self._log("  [2/5] random patient-grouped CV ...")
        rcv = self.random_cv()
        self._log("  [3/5] leave-one-site-out ...")
        lo = self.loso()
        self._log("  [4/5] transductive harmonization ...")
        Xh = self._site_standardize(self.X, self.sites)
        harm_leak = self.site_leakage(Xh)
        harm_loso = self.loso(Xh)
        self._log("  [5/5] verdict ...")

        gap = (rcv["random_cv_auroc"] - lo["loso_auroc"]
               if not (np.isnan(rcv["random_cv_auroc"]) or np.isnan(lo["loso_auroc"]))
               else np.nan)
        recov = (harm_loso["loso_auroc"] - lo["loso_auroc"]
                 if not np.isnan(lo["loso_auroc"]) else np.nan)

        r = {
            "tool": "robustness report", "version": __version__,
            "encoder": self.name,
            "n_slides": int(len(self.X)), "n_patients": int(len(np.unique(self.pids))),
            "n_sites_total": int(len(np.unique(self.sites))),
            "n_sites_usable": len(self.usable), "feat_dim": int(self.X.shape[1]),
            **leak, **rcv, **lo,
            "generalization_gap": None if np.isnan(gap) else round(float(gap), 4),
            "harmonized_loso_auroc": harm_loso["loso_auroc"],
            "harmonized_site_leakage": harm_leak["site_leakage"],
            "harmonization_recovery": None if np.isnan(recov) else round(float(recov), 4),
        }
        r["verdict"] = self._verdict(r)
        self.report = r
        return r

    # ── 5. verdict ──────────────────────────────────────────────────────
    @staticmethod
    def _verdict(r):
        flags, rec = [], []
        L, G, A = r.get("site_leakage"), r.get("generalization_gap"), r.get("loso_auroc")

        if L is not None and not np.isnan(L):
            if L > 0.90:
                flags.append(f"SEVERE site encoding (leakage {L:.3f}) — hospital "
                             f"identity is almost perfectly recoverable")
            elif L > 0.60:
                flags.append(f"HIGH site encoding (leakage {L:.3f})")
            else:
                flags.append(f"MODERATE site encoding (leakage {L:.3f})")

        if G is not None:
            if G > 0.08:
                flags.append(f"LARGE generalization gap ({G:+.3f}) — standard CV "
                             f"substantially overstates cross-hospital performance")
                rec.append("Do NOT rely on random-CV numbers for deployment planning")
            elif G > 0.03:
                flags.append(f"Moderate generalization gap ({G:+.3f})")
            else:
                flags.append(f"Small generalization gap ({G:+.3f}) — rankings appear "
                             f"stable across protocols")

        if A is not None and not np.isnan(A):
            if A < 0.65:
                rec.append("LOSO AUROC below 0.65 — not deployment-ready to unseen sites")
            elif A < 0.75:
                rec.append("Moderate cross-site performance; validate locally before use")
            else:
                rec.append("Reasonable cross-site performance")

        rv = r.get("harmonization_recovery")
        if rv is not None:
            if rv > 0.01:
                rec.append(f"Transductive harmonization RECOVERS {rv:+.3f} AUROC — "
                           f"collect unlabeled slides from the target hospital first")
            elif rv < -0.01:
                rec.append(f"Transductive harmonization COSTS {rv:+.3f} AUROC — "
                           f"note that site-centering also removes case-mix signal "
                           f"when prevalence differs across hospitals")
            else:
                rec.append("Transductive harmonization is roughly performance-neutral")

        return {"flags": flags, "recommendations": rec}

    # ── 6. per-encoder summary ───────────────────────────────────────────────────
    def to_markdown(self, path=None):
        if not self.report:
            self.run()
        r = self.report

        def fmt(v, n=4):
            return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"

        L = [f"# robustness report — `{r['encoder']}`", "",
             f"*Per-encoder robustness summary (v{r['version']})*", "",
             "## Dataset", "",
             f"- Slides: **{r['n_slides']}** · Patients: **{r['n_patients']}**",
             f"- Sites: **{r['n_sites_total']}** total, **{r['n_sites_usable']}** usable "
             f"(>= {self.min_pat} patients)",
             f"- Embedding dimension: **{r['feat_dim']}**", "",
             "## Headline numbers", "",
             "| Metric | Value | Reading |",
             "|---|---|---|",
             f"| **Site leakage** | **{fmt(r.get('site_leakage'))}** | "
             f"0 = hospital unrecoverable · 1 = perfectly recoverable |",
             f"| Site balanced accuracy | {fmt(r.get('site_bacc'))} | "
             f"chance = {fmt(r.get('chance'), 3)} |",
             f"| Random patient-grouped CV AUROC | {fmt(r.get('random_cv_auroc'))} | "
             f"the optimistic benchmark number |",
             f"| **Leave-one-site-out AUROC** | **{fmt(r.get('loso_auroc'))}** | "
             f"the honest number |",
             f"| **Generalization gap** | **{fmt(r.get('generalization_gap'))}** | "
             f"how much standard CV overstates |",
             f"| Calibration (LOSO ECE) | {fmt(r.get('loso_ece'))} | lower is better |", "",
             "## Transductive harmonization *(conditional)*", "",
             "Requires **unlabeled slides from the target hospital**. Not free.", "",
             "| Metric | Value |", "|---|---|",
             f"| Site leakage after harmonization | {fmt(r.get('harmonized_site_leakage'))} |",
             f"| LOSO AUROC after harmonization | {fmt(r.get('harmonized_loso_auroc'))} |",
             f"| **Net recovery** | **{fmt(r.get('harmonization_recovery'))}** |", "",
             "> **Caveat.** Site-standardization removes case-mix signal by construction: "
             "if hospitals differ in class prevalence, a site's mean embedding encodes "
             "that prevalence. Part of any cost here is a method artifact, not evidence "
             "that site signal carried biology.", ""]

        if r.get("per_site"):
            L += ["## Per-site performance (hardest first)", "",
                  "| Site | n | AUROC | Prevalence |", "|---|---|---|---|"]
            for s in r["per_site"]:
                L.append(f"| {s['site']} | {s['n']} | {s['auroc']:.4f} | "
                         f"{s['prevalence']:.3f} |")
            L.append("")

        v = r.get("verdict", {})
        if v.get("flags"):
            L += ["## Findings", ""] + [f"- {f}" for f in v["flags"]] + [""]
        if v.get("recommendations"):
            L += ["## Recommendations", ""] + [f"- {x}" for x in v["recommendations"]] + [""]

        L += ["---", "",
              "*Generated by robustness report. Standard benchmarks use random or "
              "patient-grouped CV, where the same hospitals appear in train and test. "
              "Pathology foundation model embeddings encode hospital identity, so those "
              "benchmarks overstate cross-institutional performance and their rankings "
              "are protocol-dependent.*"]

        md = "\n".join(L)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                f.write(md + "\n")
            self._log(f"  summary -> {path}")
        return md

    def to_json(self, path=None):
        if not self.report:
            self.run()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.report, f, indent=2, default=float)
        return self.report


def audit_encoder(X, y, patient_ids, site_labels, encoder_name="encoder", **kw):
    """One-call convenience wrapper."""
    a = RobustnessReport(X, y, patient_ids, site_labels, encoder_name, **kw)
    a.run()
    return a


if __name__ == "__main__":
    print(f"robustness report v{__version__}")
    print(__doc__)
