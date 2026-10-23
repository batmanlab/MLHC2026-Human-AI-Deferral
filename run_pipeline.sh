#!/usr/bin/env bash
# =============================================================================
# End-to-end command reference — MLHC 2026 Human-AI Deferral pipeline.
#
# This is a TEMPLATE, not a turnkey script. Two stages are not part of this
# repository and are marked "(NOT IN REPO)" below:
#   * Stage 2: training the AI backbone and assembling data_mlhc.csv
#   * Stage 5: the DaF-base / DaF-ext baselines (from the BeyondDefer repo)
# Edit the paths at the top, run the stages you have, and supply the gap
# stages yourself (see the README).
#
# Conventions: annotators are novice_1..novice_4; the AI backbone column prefix
# is RF (RF_prob / RF_pred).
# =============================================================================

set -euo pipefail

# --- paths (edit these) ------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_CSV="./data_mlhc.csv"                 # assembled interval table (see Stage 2)
HAD_REPO="/path/to/human_ai_deferral"      # local clone of clinicalml baseline repo
OUT="./output"                             # per-run outputs
FIGS="./figures"                           # figures
ANNOTATORS=(novice_1 novice_2 novice_3 novice_4)

mkdir -p "$OUT" "$FIGS"

# --- Stage 1: cEEG -> 111 features -------------------------------------------
# Edit configs/feature_extraction.yaml to point at your raw case directories.
python "$REPO_ROOT/features/extract_features.py" \
    --config "$REPO_ROOT/configs/feature_extraction.yaml" \
    --output_dir "$OUT/features"

# --- Stage 2 (NOT IN REPO): assemble data_mlhc.csv + train the AI ------------
# Concatenate the per-case feature tables from Stage 1, attach the four novice
# label columns and a patient-level `fold` column, train the Random Forest
# backbone (1000 trees), and write its out-of-fold RF_prob / RF_pred columns.
# The result is "$DATA_CSV". See the README (components being finalized).
echo "Stage 2 (AI training + data_mlhc.csv assembly) is not included; see README." >&2

# --- Stage 3: L2A (per annotator) --------------------------------------------
for a in "${ANNOTATORS[@]}"; do
    python "$REPO_ROOT/deferral/l2a.py" \
        --csv "$DATA_CSV" \
        --output_dir "$OUT/l2a" \
        --annotator "$a" \
        --ai_name RF
done

# --- Stage 4: clinicalml baselines, RS/OvA/CC/SP/MoE (per annotator) ----------
for a in "${ANNOTATORS[@]}"; do
    python "$REPO_ROOT/baselines/run_baselines.py" \
        --csv "$DATA_CSV" \
        --repo_path "$HAD_REPO" \
        --output_dir "$OUT/baselines_$a" \
        --annotator "$a" \
        --ai_name RF
done

# --- Stage 5 (NOT IN REPO): DaF-base / DaF-ext -------------------------------
# Produce row_deferral_BeyondDefer.csv / row_deferral_AdditionalBeyond.csv and
# the matching sweep rows using https://github.com/AminChrs/BeyondDefer .
echo "Stage 5 (DaF via BeyondDefer) is not included; see README." >&2

# --- Stage 6: arrange inputs for the figures ---------------------------------
# The plotting scripts expect:
#   plot_discrimination.py --baselines_csv : standard sweep (optionally + DaF rows)
#   plot_calibration.py    --baselines_csv : calibration sweep (optionally + DaF rows)
#   contiguity.py          --base_dir      : contiguity_<annotator>/row_deferral_<method>.csv
# Merge the Stage 4 (and Stage 5) outputs accordingly. For contiguity, place
# each annotator's row_deferral_<method>.csv under
# "$OUT/contiguity_inputs/contiguity_<annotator>/".

# --- Stage 7: figures --------------------------------------------------------
for a in "${ANNOTATORS[@]}"; do
    python "$REPO_ROOT/analysis/plot_discrimination.py" \
        --deferral_csv "$OUT/l2a/deferral_$a.csv" \
        --baselines_csv "$OUT/baselines_$a/results_standard_sweep.csv" \
        --output_dir "$FIGS" \
        --annotator "$a" \
        --ai_name RF

    python "$REPO_ROOT/analysis/plot_calibration.py" \
        --l2a_csv "$OUT/l2a/deferral_$a.csv" \
        --baselines_csv "$OUT/baselines_$a/results_calibration_sweep.csv" \
        --output_dir "$FIGS" \
        --annotator "$a" \
        --ai_name RF
done

python "$REPO_ROOT/analysis/contiguity.py" \
    --base_dir "$OUT/contiguity_inputs" \
    --output_dir "$FIGS" \
    --deferral_frac 0.10

echo "Done (stages present in this repo)."
