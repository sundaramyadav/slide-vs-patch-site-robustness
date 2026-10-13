# Slide-Level Encoders Are More Robust to Site Confounding Than Patch-Level Encoders

Code and released evaluation artifacts for a leave-one-site-out study of
**30 pathology foundation models** on TCGA-PRAD.

## What this measures

Pathology foundation models encode which hospital a slide came from, and this
degrades cross-institutional generalization. Prior work established this for
patch-level encoders and left slide-level encoders unevaluated.

This repository evaluates both families on byte-identical frozen folds under two
protocols that differ only in whether the test institution appears in training:

| | Patient-grouped CV | Leave-one-site-out | Degradation |
|---|---|---|---|
| Slide-level encoders (n=4) | 0.8574 | 0.8466 | **+0.0108** |
| Patch-level encoders (n=26) | 0.8199 | 0.7639 | **+0.0560** |

Slide-level encoders degrade 5.2x less (Mann-Whitney p=0.016) and retain their
leaderboard positions exactly, while patch encoders move 7.6 positions on
average.

## Contents

```
code/           analysis pipeline, in dependency order
folds/          frozen cross-validation splits, byte-identical across encoders
predictions/    per-encoder out-of-fold predictions
results/        leaderboards, site-leakage scores, harmonization sweep
```

The frozen folds and the out-of-fold predictions are the reusable part. Adopt
the folds and your numbers are directly comparable to these. Every statistic
reported in the paper is recomputable from `predictions/` and `results/` without
access to a whole-slide image.

Extracted features are not included: they total hundreds of gigabytes and are
not redistributable. The extraction commands are below.

## Install

```bash
git clone https://github.com/<USER>/slide-vs-patch-site-robustness
cd slide-vs-patch-site-robustness
pip install -r requirements.txt
```

## Configuration

Scripts read their data root from the `PCA_ROOT` environment variable:

```bash
export PCA_ROOT=/path/to/your/project/root
```

It defaults to a Colab Drive mount path if unset.

Expected layout under `$PCA_ROOT`:

```
organized_output/<risk_group>/<patient_id>/<mag>_<size>px_0px_overlap/*.h5
manifest/        folds and slide manifest
results/         script output
```

## Running the pipeline

| # | Script | Produces |
|---|---|---|
| 1 | `build_manifest_and_folds.py` | slide manifest, patient-grouped folds |
| 2 | `run_protocol_a.py` | patient-grouped CV results |
| 3 | `site_gonogo_check.py` | site feasibility check |
| 4 | `build_cache_and_site_leakage.py` | embedding cache, site-leakage scores |
| 5 | `site_leakage_sanity_checks.py` | permutation and shuffle controls |
| 6 | `build_loso_folds.py` | frozen leave-one-site-out folds |
| 7 | `run_protocol_b.py` | LOSO results, out-of-fold predictions |
| 8 | `rank_stability.py` | bootstrap ranking analysis |
| 9 | `harmonization.py` | component-removal sweep |
| 10 | `build_leaderboard.py` | final leaderboard |

Additional analyses: `spatial_localization.py` (per-patch site maps, Moran's I),
`pruning_pilot.py` (instance-level pruning, a negative result),
`robustness_report.py` (per-encoder robustness summary).

`eval_engine.py` is the shared library: data loading, cross-validation, metrics,
bootstrap confidence intervals, calibration.

## Feature extraction

Features were extracted with [TRIDENT](https://github.com/mahmoodlab/TRIDENT)
v0.2.4, which performs tissue segmentation, patch tessellation, and encoder
inference in one pass per slide:

```bash
python run_batch_of_slides.py \
    --task all \
    --wsi_dir /path/to/TCGA-PRAD/slides \
    --job_dir /path/to/output \
    --patch_encoder <encoder_name> \
    --mag 20 --patch_size 224 \
    --gpu 0
```

Five extraction configurations are used, dictated by each model's released
preprocessing:

| Configuration | Encoders |
|---|---|
| 20x / 224 px | Virchow, Virchow2, Phikon, Phikon-v2, H-optimus-0/1, H0-mini, OpenMidnight, Midnight-12k, GPFM, Hibou-L, Kaiko (5 variants), Lunit-ViTS8, GenBio-PathFM |
| 20x / 256 px | UNI-v1, UNI-v2, Prov-GigaPath, ResNet-50 |
| 20x / 384 px | MUSK |
| 20x / 512 px | CONCH v1, CONCH v1.5 |
| 10x / 256 px | CTransPath |

Slide-level encoders consume a designated backbone's patch features: CHIEF via
CTransPath, TITAN and FEATHER via CONCH v1.5, MADELEINE via CONCH v1.

Run without `--skip_errors` when diagnosing failures; that flag suppresses
exceptions in the main loop, so slides disappear from the output silently.

## Data

TCGA-PRAD: 459 diagnostic FFPE H&E whole-slide images, 401 patients, 27 tissue
source sites, with a per-slide binary risk label derived from Gleason grading
(187 high / 272 low).

Site labels come from the TCGA barcode's second field (`TCGA-KK-A8IL` -> site
`KK`). Protocol B holds out one site per fold for the 8 sites with at least 15
patients (349 slides evaluated); smaller sites are training-only.

No patient-identifying information and no raw slide data are included here.
TCGA data are public and governed by the
[NIH GDC data use policy](https://gdc.cancer.gov/access-data/data-access-policies).

## Limitations

- Single cohort, single organ, single binary task.
- Only four slide-level encoders exist publicly, and three share a CONCH
  backbone, so the group comparison has limited power.
- CHIEF and CTransPath are the only encoders extracted at 10x, so magnification
  is a confound that these experiments do not fully separate from slide-level
  pretraining.
- One site has 92.6% high-risk prevalence and high leverage; results are
  reported with and without it.
- Aggregation is mean pooling, chosen because it has no trainable parameters and
  therefore cannot itself learn a site shortcut. Patch-encoder numbers are a
  lower bound on what a learned aggregator would reach.

## Citation

```bibtex
@inproceedings{yadav2026slidevspatch,
  title     = {Slide-Level Encoders Are More Robust to Site Confounding Than
               Patch-Level Encoders: A Leave-One-Site-Out Study of 30 Pathology
               Foundation Models},
  author    = {Yadav, Sundaram and others},
  booktitle = {IEEE International Conference on Big Data (BigData) Workshops},
  year      = {2026}
}
```

## License

Code: [MIT](LICENSE). Released folds, predictions and results:
[CC-BY-4.0](LICENSE-DATA).

## Acknowledgment

Based upon data generated by the [TCGA Research Network](https://www.cancer.gov/tcga).
Feature extraction used [TRIDENT](https://github.com/mahmoodlab/TRIDENT) (Mahmood Lab).
