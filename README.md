# Human–AI Deferral under Limited Expert Availability

Code for the paper **"Human–AI Deferral under Limited Expert Availability: A Study in Intraoperative Ischemia Detection"**, *Proceedings of Machine Learning Research* **340**:1–36, 2026 (Machine Learning for Healthcare, MLHC 2026).

Nihal Murali (University of Pittsburgh), Karim Elzokm (Boston University), Parthasarathy Thirumala (University of Pittsburgh), Kayhan Batmanghelich† (Boston University), and Shyam Visweswaran† (University of Pittsburgh). *(† equal senior contribution. Contact: Nihal Murali, nihal.murali@pitt.edu.)*

This repository contains the implementation of **Learning to Augment (L2A)** — a soft-deferral method that selectively blends novice human predictions with an AI model to detect cerebral ischemia during carotid endarterectomy (CEA) from continuous EEG (cEEG) — along with the deferral baselines, the cEEG feature-extraction pipeline, and the analysis and plotting code used to produce the figures in the paper.

> **Status.** The scripts included here are cleaned for release: each is driven by command-line flags (or a YAML config for feature extraction), with no hardcoded paths. Some supporting components referenced in the paper are still being finalized and are **not yet included** — see [Components not yet included](#components-not-yet-included). As a result the pipeline is not yet runnable end-to-end: in particular, the step that trains the AI backbone and assembles the master analysis table (`data_mlhc.csv`) is not part of this repository.

## Overview

Reliable interpretation of intraoperative cEEG depends on scarce expert neurophysiologists. Rather than relying on AI alone in this safety-critical setting, we study **human–AI deferral**: methods that decide *when* and *how* to bring a human into the loop. We use **novices with only brief training** as a conservative lower-bound test of whether non-expert input carries complementary signal — not as a proposed replacement for clinically trained personnel.

**L2A** learns a per-interval weight `w ∈ [0, 1]` estimating the probability that the novice prediction improves on the AI. Given a deferral threshold `τ`, the final prediction is:

```
ŷ = w · h + (1 − w) · p     if w > τ
ŷ = p                        otherwise
```

where `p` is the AI probability of ischemia and `h ∈ {0, 1}` is the novice prediction. The weight model takes as input `z = (p, u, v)`: the AI probability `p`, the AI uncertainty `u` (Shannon entropy of `p`), and an EEG variability score `v` derived from the interval's feature vector. Because these inputs carry no novice-specific signal, the learned policy is monitor-agnostic — which is also why it transfers well across novices.

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
│   ├── plot_calibration.py        # ECE / log-loss / Brier vs deferral
│   └── contiguity.py              # interruptions & longest uninterrupted free time
├── configs/
│   └── feature_extraction.yaml    # config for extract_features.py (the only YAML-driven script)
├── scripts/
│   └── run_pipeline.sh            # end-to-end command reference (template; marks missing stages)
├── environment.yml                # conda environment (pinned)
├── requirements.txt               # pip dependencies (grouped by component)
├── LICENSE
└── README.md
```

All other scripts are configured through command-line flags (`--help` on any of them lists the options); feature extraction is the only script that reads a YAML config.

## Installation

Conda:

```bash
conda env create -f environment.yml
conda activate mlhc2026-deferral
```

or pip (Python 3.11):

```bash
pip install -r requirements.txt
```

`requirements.txt` groups dependencies by component, so you can install a subset — for example, skip the feature-extraction group (`tsai`, `bottleneck`) if you are only running L2A, the baselines, and the figures on an existing data table.

Reproducing the RS/OvA/CC/SP/MoE baselines additionally requires a local clone of the upstream benchmark they are adapted from (see [Acknowledgments](#acknowledgments)); pass its path to `run_baselines.py --repo_path`. DaF-base / DaF-ext are produced from a separate repository.

## Quickstart

Annotators are referred to as `novice_1 … novice_4` (the paper's anonymized novices); pass the relevant one with `--annotator`. The fixed AI-backbone columns are `RF_prob` / `RF_pred` (`--ai_name RF`).

```bash
# 1) cEEG -> 111 features (edit configs/feature_extraction.yaml first)
python features/extract_features.py \
    --config configs/feature_extraction.yaml --output_dir output/features

# 2) [NOT INCLUDED] assemble data_mlhc.csv and train the AI backbone
#    (see "Components not yet included")

# 3) L2A (per annotator)
python deferral/l2a.py \
    --csv data_mlhc.csv --output_dir output/l2a --annotator novice_1 --ai_name RF

# 4) hard/soft baselines (per annotator)
python baselines/run_baselines.py \
    --csv data_mlhc.csv --repo_path /path/to/human_ai_deferral \
    --output_dir output/baselines_novice_1 --annotator novice_1 --ai_name RF

# 5) figures
python analysis/plot_discrimination.py \
    --deferral_csv output/l2a/deferral_novice_1.csv \
    --baselines_csv output/baselines_novice_1/results_standard_sweep.csv \
    --output_dir figures --annotator novice_1

python analysis/plot_calibration.py \
    --l2a_csv output/l2a/deferral_novice_1.csv \
    --baselines_csv output/baselines_novice_1/results_calibration_sweep.csv \
    --output_dir figures --annotator novice_1

python analysis/contiguity.py \
    --base_dir output/contiguity_inputs --output_dir figures --deferral_frac 0.10
```

`scripts/run_pipeline.sh` captures this full sequence and marks the two stages that are not included. Note that steps 3–5 depend on `data_mlhc.csv` from step 2, which is being finalized (below).

## Method and evaluation

- **AI backbone (fixed).** A Random Forest with 1000 trees, maximum depth 15, seed 42, and no class weighting, trained on the 111 cEEG features. It performs well on tabular data under heavy class imbalance (~2.4% positive intervals). Its out-of-fold probabilities are used as fixed inputs to L2A, Compare Confidence, and Selective Prediction, so those methods require no AI retraining.
- **End-to-end backbone.** RS, OvA, MoE, DaF-base, and DaF-ext learn their own AI components with a two-layer MLP (256 and 128 hidden units; batch norm, ReLU, dropout 0.3; Kaiming init). Training uses Adam (lr 5e-4), batch size 256, 100 epochs, elementwise gradient clipping to [−1, 1], and seed 42.
- **Cross-validation.** 10-fold, patient-level — all intervals from a patient stay in the same fold.
- **Deferral sweep.** Each method is evaluated across deferral fractions 0–100%; at each level the top-ranked intervals (by a method-specific score) receive human input. For L2A the score is the learned novice weight.
- **Operating point.** At this prevalence a 0.5 threshold predicts all-negative, so binary metrics are reported at the Youden operating point.
- **Method-matched comparison.** Because RF and MLP differ, absolute cross-backbone comparisons confound the deferral method with the AI model. Each method is therefore compared to its own 0%-deferral baseline; the cleanest matched comparisons for L2A are CC and SP, which share the same RF predictions.
- **Transfer and onboarding.** L2A transfers across novices with little degradation (at 10% deferral, within-novice vs transferred AUPRC 0.636 vs 0.641; AUROC 0.961 for both), and a new monitor's policy can be fit from a modest labeled set (~72 cases recovers ~96% of the full-data AUPRC gain), since the AI model never needs retraining.

## Data availability

Patient-level cEEG data **cannot be shared** due to privacy and institutional restrictions. The dataset comprises intraoperative recordings from 400 CEA patients (2009–2017) collected in a large U.S. academic health system. The study was approved by the Institutional Review Board (protocol STUDY18120055) and determined to involve no greater than minimal risk; all data were de-identified before research analysis.

The pipeline is documented in enough detail — acquisition setup, preprocessing, and the 111-feature construction (Appendices A, B) — that the approach can be reproduced on an independently approved cEEG dataset. A synthetic-data generator matching the analysis schema will be added so the deferral and analysis code can be run without access to patient data.

## Components not yet included

The following are referenced in the paper and are being finalized for release:

- AI-model training with patient-level cross-validation (produces the out-of-fold `RF_prob` / `RF_pred` columns) and assembly of the per-case feature files into the master analysis table (`data_mlhc.csv`). This is the glue between feature extraction and the deferral/analysis code.
- The DaF-base / DaF-ext soft-deferral baselines, produced from the BeyondDefer repository (see Acknowledgments).
- The cross-novice transfer experiment (Appendix F) and the monitor-onboarding data-efficiency analysis (Appendix F.1).
- The novice label-agreement analysis (Appendix E).
- An adapter to include L2A in the contiguity analysis (the contiguity script reads per-method `row_deferral_*.csv` files; L2A currently writes its decisions in a different layout).
- A synthetic-data generator that exercises the deferral and analysis code without patient data.

## Reproducibility notes

- Scripts are configured via command-line flags; run any with `--help` to see options. Feature extraction additionally reads `configs/feature_extraction.yaml`.
- The analysis and plotting scripts consume intermediate CSVs: the assembled table with AI probabilities (from the training/assembly step above) and the per-method baseline outputs (from `run_baselines.py` and the DaF run). The baseline figures accept whatever methods are present in the supplied CSVs, so they render even if some methods are absent.
- Reproducing the hard/soft baselines requires cloning the upstream repositories; this repository references them and provides only the adapter/configuration, not a vendored copy (see Acknowledgments and License).
- Run per-annotator steps once per novice (`novice_1 … novice_4`).

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

Original code in this repository (L2A, feature extraction, analysis, and plotting) is released under the MIT License — see [LICENSE](LICENSE).

The deferral **baselines are not redistributed here**. As of this writing, neither upstream repository (see Acknowledgments) carries an open-source license, which by default means standard copyright applies and their code cannot be relicensed or re-published without permission. This repository therefore references those repositories and provides only the adapter/configuration layer; obtain the baseline code from the original sources. The cEEG feature-extraction code originates with prior work from the same group (Mina et al., 2024) and is included with attribution.

## Acknowledgments

- The cEEG feature-extraction pipeline builds on prior work on machine-learning–based ischemia detection during CEA by A. Mina and colleagues (Mina et al., 2024; Mina, 2024).
- Baselines RS, OvA, CC, SP, and MoE are adapted from the human–AI deferral benchmark of Mozannar et al. (2023), *Who Should Predict? Exact Algorithms for Learning to Defer to Humans* (AISTATS 2023): https://github.com/clinicalml/human_ai_deferral
- DaF-base and DaF-ext are adapted from the BeyondDefer repository of Charusaie et al. (2024), where the `BeyondDefer` and `AdditionalBeyond` classes correspond to DaF-base and DaF-ext respectively: https://github.com/AminChrs/BeyondDefer
