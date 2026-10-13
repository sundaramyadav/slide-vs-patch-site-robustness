# results/

Result tables behind the paper. Every number reported in the paper can be
recomputed from these files plus `predictions/`; run
`python code/verify_from_predictions.py` to check.

## Which file backs which claim

| File | Backs |
|---|---|
| `honest_leaderboard.csv` | The main results table: all 30 encoders under both protocols, the generalization gap, rank change, and site leakage |
| `loso_leaderboard.csv` | Protocol B (leave-one-site-out) metrics |
| `site_leakage_scores.csv` | Site-leakage score per encoder |
| `site_leakage_sanity_checks.csv` | The permutation test and trivial baselines that validate the leakage numbers |
| `site_distribution.csv` | Cohort composition: patients, slides and prevalence per tissue source site |
| `fold_difficulty.csv`, `fold_encoder_auroc_matrix.csv` | Per-held-out-site performance |
| `rank_stability_bootstrap.csv` | Bootstrap rank intervals per encoder |
| `spatial_summary.csv` | Per-patch site-score concentration and spatial clustering |

## Column notes for `honest_leaderboard.csv`

Columns used in the paper:

| Column | Meaning |
|---|---|
| `phase3_auroc`, `phase3_rank` | Protocol A: patient-grouped 5-fold × 3-repeat CV |
| `loso_auroc`, `loso_lo`, `loso_hi`, `loso_rank` | Protocol B: leave-one-site-out, with patient-level bootstrap CI |
| `loso_auroc_exclYL` | Protocol B excluding the extreme-prevalence site, reported as a sensitivity check |
| `generalization_gap`, `auroc_drop` | Protocol A AUROC − Protocol B AUROC |
| `rank_change` | Protocol A rank − Protocol B rank |
| `site_leakage_score` | Chance-normalised site-probe score, (bacc − 1/K)/(1 − 1/K), K = 8 |
| `site_bacc` | Raw balanced accuracy of the site probe |
| `type`, `feat_dim`, `n_slides` | Encoder family, embedding dimension, slides evaluated under Protocol B |

Columns **not** used in the paper, kept because they are part of the script's
raw output:

| Column | Note |
|---|---|
| `deployment_score`, `honest_rank` | A weighted composite of LOSO AUROC, gap and calibration, from an earlier framing of this work. It is not defined or used in the paper and should not be read as a recommendation. |
| `harmonized_auroc`, `harmonization_recovery`, `harmonized_leakage` | Populated for 12 of 30 encoders only; the harmonization sweep in the paper was run on a subset. |
| `loso_aupr`, `loso_mcc`, `loso_ece` | Secondary metrics computed for every encoder but not discussed in the paper. |
| `site_leakage_score_leak` | Duplicate of `site_leakage_score` retained from an earlier column naming. |

## Not included

The harmonization sweep and the patch-pruning pilot are described in the paper
but their raw output tables are not in this release. The code that produces them
is in `code/harmonization.py` and `code/pruning_pilot.py`.

## AUROC convention

Reported AUROC is pooled across all out-of-fold predictions, not the mean of
per-fold AUROCs. Recomputing pooled AUROC from `predictions/` reproduces the
published values to within 0.0015.
