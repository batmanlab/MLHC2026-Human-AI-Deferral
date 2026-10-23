"""
Human-workload / contiguity analysis for the human-AI deferral study.

For each (annotator, method) pair, reads the per-row deferral decisions at a
fixed deferral fraction and summarizes how the deferred intervals are
distributed over time within each 10-minute case:

  1. Longest contiguous free time (minutes): the longest run of AI-handled
     intervals (max 10 min, the whole case).
  2. Interruptions per case: the number of distinct hand-offs to the human
     (runs of deferred intervals).

Outputs mean-and-std tables (CSV + LaTeX) and a Cleveland dot plot per metric,
pooling cases across all annotators so each method gets one grand-mean dot with
a 95% confidence interval.

Expected input layout
---------------------
    <base_dir>/
        contiguity_<annotator>/
            row_deferral_<method>.csv    one row per interval, with a
                                         deferred_<frac> column (1 = deferred to
                                         the human, 0 = AI handles it), keyed by
                                         Case ID and Interval.

Per-method row_deferral_*.csv files are produced by run_baselines.py (for the
five clinicalml baselines) and by the BeyondDefer run (for DaF-base / DaF-ext).
Missing files are reported and skipped, so the plot simply omits any method
whose file is absent.

Usage
-----
    python contiguity.py \\
        --base_dir ./contiguity_inputs \\
        --output_dir ./figures \\
        --deferral_frac 0.10
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")


# =============================================================================
# FIXED ANALYSIS CONSTANTS
# =============================================================================

N_INTERVALS = 30                               # intervals per case
INTERVAL_SEC = 20                              # seconds per interval
CASE_DURATION_MIN = N_INTERVALS * INTERVAL_SEC / 60   # = 10.0 minutes

# Methods to include; each is loaded from row_deferral_<method>.csv if present.
METHODS = [
    "CompareConfidence",
    "MixOfExps",
    "OvASurrogate",
    "RealizableSurrogate",
    "SelectivePrediction",
    "BeyondDefer",        # soft-deferral baseline (DaF-base); skipped if absent
    "AdditionalBeyond",   # soft-deferral baseline (DaF-ext); skipped if absent
    "L2A",                # our method
]

# Display labels.
HUMAN_LABELS = {
    "novice_1": "Novice 1",
    "novice_2": "Novice 2",
    "novice_3": "Novice 3",
    "novice_4": "Novice 4",
}
METHOD_LABELS = {
    "CompareConfidence":   "CC",
    "MixOfExps":           "MoE",
    "OvASurrogate":        "OvA",
    "RealizableSurrogate": "RS",
    "SelectivePrediction": "SP",
    "BeyondDefer":         "DaF-base",
    "AdditionalBeyond":    "DaF-ext",
    "L2A":                 "L2A",
}

SAVE_FIGS = True

# Data-dependent globals (populated in main from the command line).
BASE_DIR = "."
FILE_TEMPLATE = os.path.join(BASE_DIR, "contiguity_{human}", "row_deferral_{method}.csv")
DEFERRAL_FRAC = 0.10
HUMANS = ["novice_1", "novice_2", "novice_3", "novice_4"]
OUTPUT_DIR = "."


# =============================================================================
# COLOR PALETTE (matches the discrimination/calibration figures)
# =============================================================================

METHOD_COLORS: Dict[str, str] = {
    "CompareConfidence":   "#43A047",   # green
    "MixOfExps":           "#8E24AA",   # purple
    "OvASurrogate":        "#FFB300",   # amber
    "RealizableSurrogate": "#E53935",   # red
    "SelectivePrediction": "#00897B",   # teal
    "BeyondDefer":         "#FF7043",   # deep orange (soft deferral)
    "AdditionalBeyond":    "#9A6324",   # brown       (soft deferral)
    "L2A":                 "#0288D1",   # sky blue    (ours)
}


# =============================================================================
# GLOBAL AESTHETICS
# =============================================================================

FONT_BASE = 13
BG_AXES = "#F7F9FC"
BG_FIG = "#FFFFFF"

plt.rcParams.update({
    "font.family":           "DejaVu Sans",
    "axes.titlesize":        FONT_BASE + 3,
    "axes.labelsize":        FONT_BASE + 1,
    "xtick.labelsize":       FONT_BASE,
    "ytick.labelsize":       FONT_BASE,
    "legend.fontsize":       FONT_BASE - 1,
    "legend.title_fontsize": FONT_BASE,
    "figure.titlesize":      FONT_BASE + 6,
    "axes.titlepad":         10,
    "axes.labelpad":         6,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.facecolor":        BG_AXES,
    "figure.facecolor":      BG_FIG,
    "axes.grid":             True,
    "grid.linestyle":        "--",
    "grid.alpha":            0.45,
    "axes.linewidth":        1.2,
    "xtick.major.width":     1.2,
    "ytick.major.width":     1.2,
})


# =============================================================================
# METRIC CONFIGS
# (column, x-axis label, section title, higher_is_better, value format)
# =============================================================================

METRIC_CONFIGS = [
    ("longest_free_min",
     "Longest Free Time (min, /10)",
     "Longest contiguous free time",
     True,
     "{:.1f}"),
    ("n_interruptions",
     "Interruptions (per case)",
     "Number of interruptions per case",
     False,
     "{:.1f}"),
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_runs(binary_seq: list) -> List[Tuple]:
    """Run-length encode a binary sequence into (value, length) tuples."""
    runs = []
    if not binary_seq:
        return runs
    cur_val, cur_len = binary_seq[0], 1
    for v in binary_seq[1:]:
        if v == cur_val:
            cur_len += 1
        else:
            runs.append((cur_val, cur_len))
            cur_val, cur_len = v, 1
    runs.append((cur_val, cur_len))
    return runs


def compute_metrics_for_case(defer_seq) -> Tuple[float, int]:
    """Per-case contiguity metrics from one binary deferral sequence.

    Returns the longest free-time run in minutes (longest run of 0s) and the
    number of interruptions (distinct runs of 1s).
    """
    runs = get_runs(list(defer_seq))
    free_chunks = [length for val, length in runs if val == 0]
    defer_chunks = [length for val, length in runs if val == 1]

    longest_free_min = (max(free_chunks) * INTERVAL_SEC / 60) if free_chunks else 0.0
    n_interruptions = len(defer_chunks)

    return longest_free_min, n_interruptions


def load_and_compute_metrics(human: str, method: str) -> Optional[pd.DataFrame]:
    """Load one (annotator, method) CSV and compute per-case metrics.

    Returns a DataFrame of per-case metrics, or None if the file or the
    expected deferral column is missing.
    """
    fpath = FILE_TEMPLATE.format(human=human, method=method)
    if not Path(fpath).exists():
        print(f"  [MISSING FILE]   {fpath}")
        return None

    df = pd.read_csv(fpath)
    defer_col = f"deferred_{DEFERRAL_FRAC:.2f}"

    if defer_col not in df.columns:
        print(f"  [MISSING COLUMN] '{defer_col}' not in {fpath}")
        avail = [c for c in df.columns if c.startswith("deferred_")]
        print(f"    Available deferred columns: {avail}")
        return None

    results = []
    for case_id, grp in df.groupby("Case ID"):
        grp_sorted = grp.sort_values("Interval")
        defer_seq = grp_sorted[defer_col].values.astype(int)

        if len(defer_seq) != N_INTERVALS:
            print(f"  [WARNING] Case {case_id}: {len(defer_seq)} intervals "
                  f"(expected {N_INTERVALS})")

        lf_min, n_int = compute_metrics_for_case(defer_seq)
        results.append({
            "case_id":          case_id,
            "longest_free_min": lf_min,
            "n_interruptions":  n_int,
        })

    return pd.DataFrame(results)


def summary_stats(series: pd.Series, label: str) -> None:
    q25, q75 = np.percentile(series, [25, 75])
    print(f"    {label}:")
    print(f"      mean={series.mean():.2f}  median={series.median():.2f}"
          f"  IQR=[{q25:.2f}, {q75:.2f}]  std={series.std():.2f}")


# =============================================================================
# LONG-FORM DATAFRAME
# =============================================================================

def build_long_df(all_metrics: Dict) -> pd.DataFrame:
    rows = []
    for human, methods_dict in all_metrics.items():
        for method, df_m in methods_dict.items():
            if df_m is None:
                continue
            tmp = df_m.copy()
            tmp["human"] = human
            tmp["human_label"] = HUMAN_LABELS.get(human, human)
            tmp["method"] = method
            tmp["method_label"] = METHOD_LABELS.get(method, method)
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _available_methods(long_df: pd.DataFrame) -> List[str]:
    present = set(long_df["method"].unique()) if not long_df.empty else set()
    return [m for m in METHODS if m in present]


def _human_display_labels() -> List[str]:
    return [HUMAN_LABELS.get(h, h) for h in HUMANS]


def _save(fname: str) -> None:
    if SAVE_FIGS:
        path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor=BG_FIG, edgecolor="none")
        print(f"  Saved: {path}")


# =============================================================================
# AGGREGATED CLEVELAND DOT PLOT (one figure per metric)
# =============================================================================

def plot_mean_dots(long_df: pd.DataFrame) -> None:
    """One dot plot per metric, pooling cases across all annotators.

    Each method gets a grand-mean dot with a +/- 1.96 SE interval from the
    pooled distribution; methods are sorted best-to-worst (best at top). L2A
    gets a ring marker and a vertical reference line.
    """
    avail_methods = _available_methods(long_df)
    n_methods = len(avail_methods)

    for col, xlabel, title, higher_is_better, fmt_str in METRIC_CONFIGS:

        fig, ax = plt.subplots(
            1, 1,
            figsize=(8, max(5.0, n_methods * 0.85 + 2.0)),
            facecolor=BG_FIG,
        )
        fig.subplots_adjust(top=0.94, bottom=0.14, left=0.18,
                            right=0.88, wspace=0.0)

        # Shared x-axis label with a direction-of-improvement arrow.
        arrow = " \u2191" if higher_is_better else " \u2193"
        ax.set_xlabel(xlabel + arrow, fontsize=19, fontweight="bold",
                      color="#1a1a2e", labelpad=14)

        # Grand-mean stats, pooled across annotators.
        stats = {}
        for m_key in avail_methods:
            vals = long_df.loc[long_df["method"] == m_key, col].dropna()
            if vals.empty:
                stats[m_key] = (np.nan, np.nan)
            else:
                mu = vals.mean()
                sem = vals.std(ddof=1) / np.sqrt(len(vals))
                stats[m_key] = (mu, 1.96 * sem)

        # Sort best-to-worst (best at the top).
        sorted_methods = sorted(
            avail_methods,
            key=lambda m: stats[m][0] if not np.isnan(stats[m][0]) else -np.inf,
            reverse=higher_is_better,
        )
        y_pos = {m: (n_methods - 1 - i) for i, m in enumerate(sorted_methods)}

        l2a_mean = stats.get("L2A", (np.nan, np.nan))[0]

        # Metric-specific x-axis limits.
        if col == "longest_free_min":
            ax.set_xlim(7.0, 10.2)
            ax.set_xticks([7, 8, 9, 10])
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
            label_side = "right"
        elif col == "n_interruptions":
            ax.set_xlim(0.0, 2.2)
            ax.set_xticks([0, 1, 2])
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
            label_side = "right"
        else:
            all_means = [v for v, _ in stats.values() if not np.isnan(v)]
            if all_means:
                d_min, d_max = min(all_means), max(all_means)
                span = d_max - d_min
                ax.set_xlim(d_min - max(span * 0.50, 0.20),
                            d_max + max(span * 1.40, 0.70))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
            label_side = "right"

        ax.set_ylim(-0.65, n_methods - 0.35)
        ax.set_facecolor(BG_AXES)

        x_min_data, x_max_data = ax.get_xlim()
        span_x = x_max_data - x_min_data

        # Alternating row bands.
        for i, m in enumerate(sorted_methods):
            ax.axhspan(y_pos[m] - 0.46, y_pos[m] + 0.46,
                       color="#FFFFFF" if i % 2 == 0 else "#E8EDF4",
                       zorder=0, alpha=0.55)

        # L2A vertical reference line.
        if not np.isnan(l2a_mean):
            ax.axvline(l2a_mean, color="#0288D1", linewidth=1.8,
                       linestyle=(0, (5, 3)), alpha=0.65, zorder=1)

        # Draw each method's dot and confidence interval.
        for m_key in sorted_methods:
            mu, ci = stats[m_key]
            y = y_pos[m_key]
            color = METHOD_COLORS[m_key]
            is_l2a = (m_key == "L2A")

            if np.isnan(mu):
                ax.text(x_min_data + span_x * 0.05, y, "n/a",
                        va="center", fontsize=10, color="#aaa", zorder=3)
                continue

            ci_val = ci if not np.isnan(ci) else 0.0

            ax.plot([mu - ci_val, mu + ci_val], [y, y],
                    color=color, linewidth=2.2, alpha=0.65,
                    solid_capstyle="round", zorder=2)
            for xc in [mu - ci_val, mu + ci_val]:
                ax.plot([xc, xc], [y - 0.16, y + 0.16],
                        color=color, linewidth=1.8, alpha=0.65, zorder=2)

            if is_l2a:
                ax.scatter([mu], [y], s=320, color="white",
                           edgecolors=color, linewidths=3.0, zorder=4)
                ax.scatter([mu], [y], s=100, color=color,
                           edgecolors="none", zorder=5)
            else:
                ax.scatter([mu], [y], s=160, color=color,
                           edgecolors="white", linewidths=1.2, zorder=4)

            val_str = (fmt_str.format(mu) + " min"
                       if col == "longest_free_min"
                       else fmt_str.format(mu))

            if label_side == "left":
                ax.text(mu - ci_val - span_x * 0.022, y, val_str,
                        va="center", ha="right", fontsize=14,
                        color=color, fontweight="bold", zorder=5)
            else:
                ax.text(mu + ci_val + span_x * 0.022, y, val_str,
                        va="center", ha="left", fontsize=14,
                        color=color, fontweight="bold", zorder=5)

        # Y-axis: method labels, colored to match.
        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(
            [METHOD_LABELS.get(m, m) for m in sorted_methods],
            fontsize=17, fontweight="bold",
        )
        for tick, m in zip(ax.get_yticklabels(), sorted_methods):
            tick.set_color(METHOD_COLORS.get(m, "#333"))

        ax.tick_params(axis="x", labelsize=15)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")

        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", linestyle="--", alpha=0.40, zorder=0)
        ax.grid(axis="y", visible=False)

        safe_col = col.replace("_", "-")
        _save(f"contiguity_{safe_col}_dotplot_mean.pdf")
        plt.close(fig)


# =============================================================================
# PAPER-READY TABLES (mean +/- std; CSV and LaTeX)
# =============================================================================

def build_summary_table(long_df: pd.DataFrame, col: str) -> pd.DataFrame:
    avail_methods = _available_methods(long_df)
    records = []
    for method in avail_methods:
        m_lbl = METHOD_LABELS.get(method, method)
        row = {"Method": m_lbl}
        for human in HUMANS:
            h_lbl = HUMAN_LABELS.get(human, human)
            vals = long_df.loc[
                (long_df["method"] == method) & (long_df["human"] == human),
                col
            ].dropna()
            row[h_lbl] = f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}" \
                if not vals.empty else "—"
        records.append(row)
    return pd.DataFrame(records).set_index("Method")


def _to_latex(df_tbl: pd.DataFrame, col: str, metric_name: str,
              unit: str) -> str:
    higher_is_better = (col == "longest_free_min")

    def _extract(cell):
        try:
            return float(str(cell).split()[0])
        except Exception:
            return np.nan

    df_num = df_tbl.map(_extract)
    best = df_num.max(axis=0) if higher_is_better else df_num.min(axis=0)

    col_header = " & ".join(df_tbl.columns.tolist())
    lines = [
        r"\begin{table}[h]",
        r"\centering", r"\small",
        rf"\caption{{{metric_name} (mean $\pm$ std{', ' + unit if unit else ''})"
        rf" at {DEFERRAL_FRAC*100:.0f}\% deferral"
        rf" (case = {CASE_DURATION_MIN:.0f} min, {N_INTERVALS} intervals)}}",
        r"\begin{tabular}{l" + "c" * len(df_tbl.columns) + "}",
        r"\toprule",
        r"Method & " + col_header + r" \\",
        r"\midrule",
    ]
    for method, row_data in df_tbl.iterrows():
        cells = []
        for human in df_tbl.columns:
            val = row_data[human]
            nval = _extract(val)
            cells.append(rf"\textbf{{{val}}}"
                         if (not np.isnan(nval) and nval == best[human])
                         else str(val))
        lines.append(f"{method} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def print_and_save_tables(long_df: pd.DataFrame) -> None:
    DIVIDER = "=" * 72

    TABLE_CFGS = [
        ("longest_free_min", "Longest contiguous free time", "min",   True),
        ("n_interruptions",  "Number of interruptions per case", "count", False),
    ]

    print(f"\n{DIVIDER}")
    print(f"  TABLES — mean +/- std   |   Deferral: {DEFERRAL_FRAC*100:.0f}%   "
          f"|   Case: {CASE_DURATION_MIN:.0f} min ({N_INTERVALS} x {INTERVAL_SEC}s)")
    print(DIVIDER)

    for col, metric_name, unit, higher_is_better in TABLE_CFGS:
        df_tbl = build_summary_table(long_df, col)
        print(f"\n  {metric_name}  [{unit}]")
        print("  " + "-" * 68)
        print(df_tbl.to_string())
        print(f"\n  best = {'highest' if higher_is_better else 'lowest'} "
              f"per annotator\n")

        if SAVE_FIGS:
            safe_col = col.replace("_", "-")
            csv_path = os.path.join(OUTPUT_DIR, f"table_{safe_col}_mean.csv")
            df_tbl.to_csv(csv_path)
            print(f"  Saved CSV  : {csv_path}")

            tex_path = os.path.join(OUTPUT_DIR, f"table_{safe_col}_mean.tex")
            with open(tex_path, "w") as fh:
                fh.write(_to_latex(df_tbl, col, metric_name, unit))
            print(f"  Saved LaTeX: {tex_path}")

    # Compact aggregate table across both metrics.
    print(f"\n{DIVIDER}")
    print("  AGGREGATE (both metrics, all annotators)")
    print(DIVIDER)
    avail_methods = _available_methods(long_df)
    agg = []
    for method in avail_methods:
        row = {"Method": METHOD_LABELS.get(method, method)}
        for col, metric_name, unit, _ in TABLE_CFGS:
            short = metric_name[:16]
            for human in HUMANS:
                h_lbl = HUMAN_LABELS.get(human, human)
                vals = long_df.loc[
                    (long_df["method"] == method) & (long_df["human"] == human),
                    col
                ].dropna()
                row[f"{short}|{h_lbl}"] = \
                    f"{vals.mean():.2f}±{vals.std(ddof=1):.2f}" \
                    if not vals.empty else "—"
        agg.append(row)
    df_agg = pd.DataFrame(agg).set_index("Method")
    print(df_agg.to_string())
    if SAVE_FIGS:
        agg_csv = os.path.join(OUTPUT_DIR, "table_aggregate_mean.csv")
        df_agg.to_csv(agg_csv)
        print(f"\n  Saved aggregate CSV: {agg_csv}")


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Contiguity / workload analysis: interruptions and longest "
                    "uninterrupted free time per case."
    )
    parser.add_argument(
        "--base_dir", required=True,
        help="Directory containing contiguity_<annotator>/row_deferral_<method>.csv.",
    )
    parser.add_argument(
        "--output_dir", default=".",
        help="Directory for the output tables and figures.",
    )
    parser.add_argument(
        "--deferral_frac", type=float, default=0.10,
        help="Deferral fraction to analyze (must match a 'deferred_<frac>' column).",
    )
    parser.add_argument(
        "--annotators", nargs="+",
        default=["novice_1", "novice_2", "novice_3", "novice_4"],
        help="Annotator ids (directory suffixes under --base_dir).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global BASE_DIR, FILE_TEMPLATE, DEFERRAL_FRAC, HUMANS, OUTPUT_DIR
    BASE_DIR = os.path.abspath(args.base_dir)
    FILE_TEMPLATE = os.path.join(BASE_DIR, "contiguity_{human}", "row_deferral_{method}.csv")
    DEFERRAL_FRAC = args.deferral_frac
    HUMANS = args.annotators
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    defer_col_name = f"deferred_{DEFERRAL_FRAC:.2f}"
    print("=" * 65)
    print("  Contiguity Analysis")
    print(f"  Case structure    : {N_INTERVALS} intervals x {INTERVAL_SEC}s "
          f"= {CASE_DURATION_MIN:.0f} min per case")
    print(f"  Deferral fraction : {DEFERRAL_FRAC * 100:.0f}%  "
          f"(column: '{defer_col_name}')")
    print(f"  Annotators        : {[HUMAN_LABELS.get(h, h) for h in HUMANS]}")
    print(f"  Methods           : {[METHOD_LABELS.get(m, m) for m in METHODS]}")
    print("  Note: methods whose row_deferral CSV is absent are skipped.")
    print("=" * 65)

    all_metrics: Dict = {}

    for human in HUMANS:
        h_lbl = HUMAN_LABELS.get(human, human)
        print(f"\n-- {h_lbl} " + "-" * 50)
        all_metrics[human] = {}

        for method in METHODS:
            m_lbl = METHOD_LABELS.get(method, method)
            print(f"  Method: {m_lbl}")
            df_m = load_and_compute_metrics(human, method)
            all_metrics[human][method] = df_m

            if df_m is not None:
                print(f"    Cases loaded: {len(df_m)}")
                summary_stats(df_m["longest_free_min"],
                              "Longest contiguous free time (min)")
                summary_stats(df_m["n_interruptions"],
                              "Number of interruptions per case")

    long_df = build_long_df(all_metrics)

    if long_df.empty:
        print("\n[ERROR] No data loaded — check file paths and column names.")
        return

    n_m = long_df["method"].nunique()
    print(f"\n  Long-form DataFrame: {long_df.shape[0]} rows  "
          f"({long_df['human'].nunique()} annotators x {n_m} methods x cases)")

    print("\n[1/2] Generating paper-ready tables ...")
    print_and_save_tables(long_df)

    print("\n[2/2] Cleveland dot plots (pooled across annotators) ...")
    plot_mean_dots(long_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
