# Human–AI Deferral under Limited Expert Availability: A Study in Intraoperative Ischemia Detection

Code for the paper **"Human–AI Deferral under Limited Expert Availability: A Study in Intraoperative Ischemia Detection"**, *Proceedings of Machine Learning Research* **340**:1–36, 2026 (Machine Learning for Healthcare, MLHC 2026).

Nihal Murali (University of Pittsburgh), Karim Elzokm (Boston University), Parthasarathy Thirumala (University of Pittsburgh), Kayhan Batmanghelich† (Boston University), and Shyam Visweswaran† (University of Pittsburgh). *(† equal senior contribution. Contact: Nihal Murali, nihal.murali@pitt.edu.)*

This repository contains the implementation of **Learning to Augment (L2A)** — a soft-deferral method that selectively blends novice human predictions with an AI model to detect cerebral ischemia during carotid endarterectomy (CEA) from continuous EEG (cEEG) — along with the deferral baselines, the cEEG feature-extraction pipeline, and the analysis and plotting code used to produce the figures in the paper.

> **Status:** This repository is being organized for public release alongside the camera-ready paper. The core method, baselines, and analysis scripts are included. Some supporting components (AI-model training and dataset assembly, cross-novice transfer, monitor-onboarding data-efficiency, novice-agreement analysis, and a synthetic-data generator) are being finalized, and the scripts currently reference absolute paths that need to be set for a given environment. See [Reproducibility notes](#reproducibility-notes).

## Overview

Reliable interpretation of intraoperative cEEG depends on scarce expert neurophysiologists. Rather than relying on AI alone in this safety-critical setting, we study **human–AI deferral**: methods that decide *when* and *how* to bring a human into the loop. We use **novices with only brief training** as a conservative lower-bound test of whether non-expert input carries complementary signal — not as a proposed replacement for clinically trained personnel.

**L2A** learns a per-interval weight `w ∈ [0, 1]` estimating the probability that the novice prediction improves on the AI. Given a deferral threshold `τ`, the final prediction is:

```
ŷ = w · h + (1 − w) · p     if w > τ
ŷ = p                        otherwise
```

where `p` is the AI probability of ischemia and `h ∈ {0, 1}` is the novice prediction. The weight model takes as input `z = (p, u, v)`: the AI probability `p`, the AI uncertainty `u` (Shannon entropy of `p`), and an EEG variability score `v` derived from the interval's feature vector. Because these inputs carry no novice-specific signal, the learned policy is monitor-agnostic — which is also why it transfers well across novices (see below).

With only 5–10% novice involvement, L2A improves AUPRC by ~20% and precision and sensitivity by ~40% over the AI alone, while improving calibration relative to prior deferral methods.

## Repository structure

```
MLHC2026-Human-AI-Deferral/
├── features/
│   └── extract_features.py        # cEEG → 111 quantitative features (Appendix A/B)
├── deferral/
│   └── l2a.py                     # L2A: weight model, gating, nested-CV evaluation
├── baselines/
│   └── run_baselines.py           # RS, OvA, CC, SP, MoE + row-level deferral / sweep CSVs
├── analysis/
│   ├── plot_discrimination.py     # AUPRC/AUROC/F1/precision/sensitivity/specificity vs deferral
│   ├── plot_calibration.py        # ECE / Brier / log-loss vs deferral
│   └── contiguity.py              # interruptions & longest uninterrupted free time
├── configs/                       # extraction / training configuration
├── requirements.txt
├── LICENSE
└── README.md
```

Components being finalized and not yet included: AI-model training with patient-level cross-validation (produces the out-of-fold `RF_prob` / `RF_pred` columns), assembly of the per-case feature files into the master analysis table, the DaF-base / DaF-ext soft-deferral baselines, the cross-novice transfer experiment (Appendix F), the monitor-onboarding data-efficiency analysis (Appendix F.1), the novice label-agreement analysis (Appendix E), and a synthetic-data generator that exercises the deferral and analysis code without patient data.

## Method and evaluation

- **AI backbone (fixed).** A Random Forest with 1000 trees, maximum depth 15, seed 42, and no class weighting, trained on the 111 cEEG features. It performs well on tabular data under heavy class imbalance (~2.4% positive intervals). Its out-of-fold probabilities are used as fixed inputs to L2A, Compare Confidence, and Selective Prediction, so those methods require no AI retraining.
- **End-to-end backbone.** RS, OvA, MoE, DaF-base, and DaF-ext learn their own AI components with a two-layer MLP (256 and 128 hidden units; batch norm, ReLU, dropout 0.3; Kaiming init). Training uses Adam (lr 5e-4), batch size 256, 100 epochs, elementwise gradient clipping to [−1, 1], and seed 42.
- **Cross-validation.** 10-fold, patient-level — all intervals from a patient stay in the same fold.
- **Deferral sweep.** Each method is evaluated across deferral fractions 0–100%; at each level the top-ranked intervals (by a method-specific score) receive human input. For L2A the score is the learned novice weight.
- **Operating point.** At this prevalence a 0.5 threshold predicts all-negative, so binary metrics are reported at the Youden operating point.
- **Method-matched comparison.** Because RF and MLP differ, absolute cross-backbone comparisons confound the deferral method with the AI model. Each method is therefore compared to its own 0%-deferral baseline; the cleanest matched comparisons for L2A are CC and SP, which share the same RF predictions.
- **Transfer and onboarding.** L2A transfers across novices with little degradation (at 10% deferral, within-novice vs transferred AUPRC 0.636 vs 0.641; AUROC 0.961 for both), and a new monitor's policy can be fit from a modest labeled set (~72 cases recovers ~96% of the full-data AUPRC gain), since the AI model never needs retraining.

## Requirements

Core:

```
numpy
pandas
scikit-learn
matplotlib
tqdm
```

Additional, by component:

- **L2A / baselines:** `xgboost`, `lightgbm` (selectable AI backbones), `torch` (MLP-based baselines)
- **Feature extraction:** `scipy`, `bottleneck`, `joblib`, `pyyaml`, `tsai`

A `requirements.txt` is provided. A pinned environment file will be added.

Reproducing the RS/OvA/CC/SP/MoE and DaF baselines additionally requires the two upstream repositories they are adapted from (see [Acknowledgments](#acknowledgments)); clone those separately and point the baseline script's path variables at them.

## Data availability

Patient-level cEEG data **cannot be shared** due to privacy and institutional restrictions. The dataset comprises intraoperative recordings from 400 CEA patients (2009–2017) collected in a large U.S. academic health system. The study was approved by the Institutional Review Board (protocol STUDY18120055) and determined to involve no greater than minimal risk; all data were de-identified before research analysis.

The pipeline is documented in enough detail — acquisition setup, preprocessing, and the 111-feature construction (Appendices A, B) — that the approach can be reproduced on an independently approved cEEG dataset. A synthetic-data generator matching the analysis schema will be added so the deferral and analysis code can be run without access to patient data.

## Reproducibility notes

- Scripts currently use absolute cluster paths (e.g. `/jet/home/...`); set these to your environment before running.
- The analysis and plotting scripts consume intermediate CSVs (the assembled feature table with AI probabilities, and per-method baseline outputs). The scripts that produce those tables are part of the components being finalized above.
- Baseline reproduction requires cloning the two upstream repositories; this repository provides the adapter and configuration, not a vendored copy (see Acknowledgments).
- Expect to run per-annotator steps once per novice.

## Citation

```bibtex
@inproceedings{murali2026humanaideferral,
  title     = {Human--AI Deferral under Limited Expert Availability:
               A Study in Intraoperative Ischemia Detection},
  author    = {Murali, Nihal and Elzokm, Karim and Thirumala, Parthasarathy
               and Batmanghelich, Kayhan and Visweswaran, Shyam},
  booktitle = {Proceedings of the Machine Learning for Healthcare Conference (MLHC)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {340},
  pages     = {1--36},
  year      = {2026},
  publisher = {PMLR}
}
```

## License

Original code in this repository (L2A, analysis, and plotting) is released under the MIT License — see [LICENSE](LICENSE).

The deferral **baselines are not redistributed here**. As of this writing, neither upstream repository (see Acknowledgments) carries an open-source license, which by default means standard copyright applies and their code cannot be relicensed or re-published without permission. This repository therefore references those repositories and provides only the adapter/configuration layer; users should obtain the baseline code from the original sources. The cEEG feature-extraction code originates with prior work from the same group (Mina et al., 2024) and is included with attribution.

## Acknowledgments

- The cEEG feature-extraction pipeline builds on prior work on machine-learning–based ischemia detection during CEA by A. Mina and colleagues (Mina et al., 2024; Mina, 2024).
- Baselines RS, OvA, CC, SP, and MoE are adapted from the human–AI deferral benchmark of Mozannar et al. (2023), *Who Should Predict? Exact Algorithms for Learning to Defer to Humans* (AISTATS 2023): https://github.com/clinicalml/human_ai_deferral
- DaF-base and DaF-ext are adapted from the BeyondDefer repository of Charusaie et al. (2024), where the `BeyondDefer` and `AdditionalBeyond` classes correspond to DaF-base and DaF-ext respectively: https://github.com/AminChrs/BeyondDefer
