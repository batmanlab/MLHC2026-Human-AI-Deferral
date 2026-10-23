"""
Calibration-vs-deferral figure for the human-AI deferral study.

Plots ECE, log loss, and Brier score as a function of deferral fraction, with
L2A highlighted against the hard- and soft-deferral baselines. Output is a
single 1x3 figure saved as PDF.

Inputs
------
    --l2a_csv        L2A out-of-fold results (deferral_<annotator>.csv from
                     l2a.py). Must contain the ground-truth column, the AI
                     probability column, and the per-fraction gated_prob_dNN
                     columns.
    --baselines_csv  Baseline calibration sweep with a 'Method' column, a
                     'TargetFraction' column, and Uncal_/Cal_ columns for ECE,
                     MCE, Brier, and LogLoss (results_calibration_sweep.csv from
                     run_baselines.py, combined with the DaF rows from the
                     BeyondDefer run).

Usage
-----
    python plot_calibration.py \\
        --l2a_csv deferral_novice_1.csv \\
        --baselines_csv baselines_novice_1_cal.csv \\
        --output_dir ./figures \\
        --annotator novice_1
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
from sklearn.metrics import brier_score_loss, log_loss

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

CALIB_FRACTIONS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
                   0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                   0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
                   0.90, 0.95, 1.00]
CALIB_N_BINS = 15

# =============================================================================
# FONT SIZES
# =============================================================================

FS_TICK    = 30    # x and y tick labels
FS_LABEL   = 30    # axis labels
FS_LEGEND  = 14.5  # legend text
FS_LEG_TTL = 15    # legend title

# =============================================================================
# METHOD CONFIG
# =============================================================================

SOFT_DEFER_KEYS = {'BeyondDefer', 'AdditionalBeyond'}

METHOD_CONFIG = {
    'RealizableSurrogate':  dict(enabled=True,  label='RS', color='#E53935', ls='--', marker='o', lw=2, markersize=9),
    'LceSurrogate':         dict(enabled=False, label='LCE',        color='#F4511E', ls='--', marker='o', lw=2, markersize=9),
    'OvASurrogate':         dict(enabled=True,  label='OvA',        color='#FFB300', ls='--', marker='o', lw=2, markersize=9),
    'CompareConfidence':    dict(enabled=True,  label='CC',   color='#43A047', ls='--', marker='o', lw=2, markersize=9),
    'SelectivePrediction':  dict(enabled=True,  label='SP', color='#00897B', ls='--', marker='o', lw=2, markersize=9),
    'MixOfExps':            dict(enabled=True,  label='MoE',       color='#8E24AA', ls='--', marker='o', lw=2, markersize=9),
    'DifferentiableTriage': dict(enabled=False, label='DT',color='#D81B60', ls='--', marker='o', lw=2, markersize=9),
    'BeyondDefer':          dict(enabled=True,  label='DaF-base',            color='#FF7043', ls='-',  marker='o', lw=3,
                                 markerfacecolor='white', markeredgecolor='#FF7043', markeredgewidth=2.0, markersize=10),
    'AdditionalBeyond':     dict(enabled=True,  label='DaF-ext',         color='#9A6324', ls='-',  marker='o', lw=3,
                                 markerfacecolor='white', markeredgecolor='#9A6324', markeredgewidth=2.0, markersize=10),
}

# Fallback styles for any method present in the CSV but not configured above.
_FALLBACK_COLORS  = ["#332288", "#AA4499", "#44AA99", "#117733", "#882255", "#DDCC77", "#CC6677"]
_FALLBACK_LS      = [(0, (4, 2)), (0, (3, 2, 1, 2)), (0, (5, 3)), (0, (2, 2)), (0, (6, 1)), (0, (1, 2))]
_FALLBACK_MARKERS = ["p", "H", "8", "<", ">", "1", "2"]

CAL_ALPHA   = 0.50   # alpha for (optional) recalibrated curves
UNCAL_ALPHA = 0.88   # alpha for uncalibrated curves

# =============================================================================
# DESIGN TOKENS
# =============================================================================

COLOR_MAIN = '#0288D1'
BG_AXES    = '#F7F9FC'
BG_FIGURE  = '#FFFFFF'

L2A_GLOW_LAYERS = [(14, 0.04), (9, 0.08), (5, 0.14)]
L2A_CORE_LW     = 3.6

LEG_STYLE = dict(
    fontsize=FS_LEGEND, framealpha=0.3,
    facecolor='#F5F5F5', edgecolor='#999999',
    fancybox=False,
    borderpad=0.9, labelspacing=0.52,
    handlelength=3.2, handleheight=1.6,
)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    "#CCCCCC",
    "axes.linewidth":    1.0,
    "grid.color":        "#E3E8EF",
    "grid.linestyle":    "-",
    "grid.linewidth":    0.8,
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "xtick.direction":   "out",
    "ytick.direction":   "out",
})

# Panels and display titles (arrows indicate "lower is better").
CALIB_METRICS = [
    ("ECE",     "ECE \u2193"),
    ("LogLoss", "Log Loss \u2193"),
    ("Brier",   "Brier Score \u2193"),
]

# Data-dependent globals (populated in main from the input CSVs).
df_l2a = None
df_bl = None
y_true = None
l2a_calib = None
l2a_fracs = None
methods_to_plot = []
effective_mode = "uncal"
AI_PROB_COL = "RF_prob"
GROUNDTRUTH_COL = "Event"


# =============================================================================
# CALIBRATION METRIC FUNCTIONS
# =============================================================================

def ece_score(y_true, y_prob, n_bins=CALIB_N_BINS):
    """Expected calibration error."""
    y_true = np.array(y_true); y_prob = np.clip(np.array(y_prob), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1); ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum():
            ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def mce_score(y_true, y_prob, n_bins=CALIB_N_BINS):
    """Maximum calibration error."""
    y_true = np.array(y_true); y_prob = np.clip(np.array(y_prob), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1); mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum():
            mce = max(mce, abs(y_true[mask].mean() - y_prob[mask].mean()))
    return mce


def compute_all_calib(y_true, y_prob):
    y_prob = np.clip(np.array(y_prob), 1e-7, 1 - 1e-7)
    return {"ECE": ece_score(y_true, y_prob), "MCE": mce_score(y_true, y_prob),
            "Brier": brier_score_loss(y_true, y_prob), "LogLoss": log_loss(y_true, y_prob)}


def frac_to_col(frac):
    """Map a deferral fraction to the L2A gated-probability column name."""
    return f"gated_prob_d{int(round(frac * 100)):02d}"


# =============================================================================
# HELPERS
# =============================================================================

def get_bl_curve(method, metric_col):
    sub = df_bl[df_bl["Method"] == method].sort_values("TargetFraction").reset_index(drop=True)
    xs = (sub["TargetFraction"] * 100).tolist()
    ys = sub[metric_col].tolist() if metric_col in sub.columns else [np.nan] * len(xs)
    return xs, ys


def _method_line2d(cfg, is_cal=False):
    return mlines.Line2D(
        [0], [0],
        color=cfg["color"],
        linewidth=cfg["lw"] * (0.7 if is_cal else 1.0),
        linestyle=(0, (2, 1.5)) if is_cal else cfg["ls"],
        marker=cfg["marker"] if not is_cal else None,
        markersize=cfg.get("markersize", 8),
        markerfacecolor=cfg.get("markerfacecolor", cfg["color"]),
        markeredgecolor=cfg.get("markeredgecolor", "white"),
        markeredgewidth=cfg.get("markeredgewidth", 0.8),
        alpha=CAL_ALPHA if is_cal else UNCAL_ALPHA,
    )


def _l2a_handle():
    return mlines.Line2D(
        [0], [0],
        color=COLOR_MAIN, linewidth=4.0, linestyle="-",
        marker="o", markersize=10,
        markerfacecolor="white", markeredgecolor=COLOR_MAIN, markeredgewidth=2.0,
    )


def _style_legend_title(leg):
    t = leg.get_title()
    t.set_fontweight("bold"); t.set_fontstyle("italic")
    t.set_fontsize(FS_LEG_TTL); t.set_color("#1a1a2e")


# =============================================================================
# TWO LEGEND BOXES — hard (left) + soft (right)
# =============================================================================

def place_two_legends(ax, fig, methods_in_plot):
    soft_order = ['BeyondDefer', 'AdditionalBeyond']
    soft_active = [m for m in soft_order if m in methods_in_plot]
    hard_active = [m for m in methods_in_plot if m not in SOFT_DEFER_KEYS]

    sh, sl = [_l2a_handle()], ["L2A (Ours)"]
    for m in soft_active:
        if effective_mode in ("uncal", "both"):
            lbl = METHOD_CONFIG[m]["label"] + (" (uncal)" if effective_mode == "both" else "")
            sh.append(_method_line2d(METHOD_CONFIG[m])); sl.append(lbl)
        if effective_mode in ("cal", "both"):
            sh.append(_method_line2d(METHOD_CONFIG[m], is_cal=True))
            sl.append(METHOD_CONFIG[m]["label"] + " (cal)")

    hh, hl = [], []
    for m in hard_active:
        if effective_mode in ("uncal", "both"):
            lbl = METHOD_CONFIG[m]["label"] + (" (uncal)" if effective_mode == "both" else "")
            hh.append(_method_line2d(METHOD_CONFIG[m])); hl.append(lbl)
        if effective_mode in ("cal", "both"):
            hh.append(_method_line2d(METHOD_CONFIG[m], is_cal=True))
            hl.append(METHOD_CONFIG[m]["label"] + " (cal)")

    # Hard-deferral box anchored top-left.
    if hh:
        leg_hard = ax.legend(hh, hl, title="\u25cf  Hard Deferral",
                             loc="upper left",
                             bbox_to_anchor=(0.01, 0.99),
                             bbox_transform=ax.transAxes,
                             **LEG_STYLE)
        _style_legend_title(leg_hard)
        leg_hard.get_frame().set_linewidth(1.0)
        for txt in leg_hard.get_texts():
            txt.set_color("#222222")
        ax.add_artist(leg_hard)

        # Measure the hard box so the soft box can sit just to its right.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transAxes.inverted()
        hard_bb = leg_hard.get_window_extent(renderer)
        soft_left, _ = inv.transform((hard_bb.x1 + 8, 0))   # 8px gap
    else:
        soft_left = 0.01

    # Soft-deferral box to the right of the hard box (same top anchor).
    leg_soft = ax.legend(sh, sl, title="\u25cb  Soft Deferral",
                         loc="upper left",
                         bbox_to_anchor=(soft_left, 0.99),
                         bbox_transform=ax.transAxes,
                         **LEG_STYLE)
    _style_legend_title(leg_soft)
    leg_soft.get_frame().set_linewidth(1.0)
    leg_soft.get_texts()[0].set_fontweight("bold")
    for txt in leg_soft.get_texts():
        txt.set_color("#222222")


# =============================================================================
# FIGURE BUILDER
# =============================================================================

def make_calib_figure(metrics_subset, out_path, legend_idx=1):
    """Render the calibration panels and save as PDF (legend on one panel)."""
    n_panels = len(metrics_subset)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 8), facecolor=BG_FIGURE)
    if n_panels == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.93, bottom=0.13, left=0.07, right=0.97, wspace=0.36)

    for panel_idx, (ax, (metric_key, panel_title)) in enumerate(zip(axes, metrics_subset)):
        ax.set_facecolor(BG_AXES)
        ax.grid(True, zorder=0)

        # Baseline curves.
        for method in methods_to_plot:
            cfg = METHOD_CONFIG[method]
            mk_face = cfg.get("markerfacecolor", cfg["color"])
            mk_edge = cfg.get("markeredgecolor", "white")
            mk_ew = cfg.get("markeredgewidth", 0.8)
            mk_size = cfg.get("markersize", 8)

            if effective_mode in ("uncal", "both"):
                xs, ys = get_bl_curve(method, f"Uncal_{metric_key}")
                ax.plot(xs, ys, color=cfg["color"], linewidth=cfg["lw"],
                        linestyle=cfg["ls"], marker=cfg["marker"], markersize=mk_size,
                        markerfacecolor=mk_face, markeredgecolor=mk_edge,
                        markeredgewidth=mk_ew, alpha=UNCAL_ALPHA, zorder=3)

            if effective_mode in ("cal", "both"):
                xs, ys = get_bl_curve(method, f"Cal_{metric_key}")
                ax.plot(xs, ys, color=cfg["color"], linewidth=cfg["lw"] * 0.7,
                        linestyle=(0, (2, 1.5)), alpha=CAL_ALPHA, zorder=3)

        # Y limits from all drawn series.
        all_y = list(l2a_calib[metric_key])
        pfxs = ["Uncal_", "Cal_"] if effective_mode == "both" \
            else [f"{'Uncal' if effective_mode == 'uncal' else 'Cal'}_"]
        for method in methods_to_plot:
            for pfx in pfxs:
                _, ys = get_bl_curve(method, f"{pfx}{metric_key}")
                all_y.extend(v for v in ys if not np.isnan(v))
        data_min, data_max = min(all_y), max(all_y)
        span = max(data_max - data_min, 0.01)
        y_lo = max(0.0, data_min - span * 0.10)
        y_hi = min(data_max + span * 0.28, data_max * 1.5 + 0.02)
        ax.set_ylim(y_lo, y_hi)

        # L2A: glow halo, then the solid core.
        x_arr = np.array(l2a_fracs)
        y_arr = np.array(l2a_calib[metric_key])

        ax.fill_between(x_arr, y_arr, y_lo, alpha=0.07, color=COLOR_MAIN, zorder=2)
        for glow_lw, glow_alpha in L2A_GLOW_LAYERS:
            ax.plot(x_arr, y_arr, color=COLOR_MAIN, linewidth=glow_lw,
                    linestyle="-", alpha=glow_alpha, zorder=4, solid_capstyle="round")
        ax.plot(x_arr, y_arr, color=COLOR_MAIN, linewidth=L2A_CORE_LW, zorder=5,
                solid_capstyle="round", solid_joinstyle="round")
        ax.scatter(x_arr, y_arr, s=100, zorder=6,
                   facecolors="white", edgecolors=COLOR_MAIN, linewidths=2.2)

        # Axes.
        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"], fontsize=FS_TICK, fontweight="bold")

        fmt = "%.1f" if metric_key == "LogLoss" else "%.2f"
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        ax.yaxis.set_major_locator(mticker.LinearLocator(numticks=3))
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.tick_params(axis="x", labelsize=FS_TICK)

        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")

        ax.set_title('')
        if panel_idx == legend_idx:
            ax.set_xlabel("Deferral Fraction (%)", fontsize=FS_LABEL,
                          labelpad=14, color="#333", fontweight="bold")
        else:
            ax.set_xlabel("")
        ax.set_ylabel(panel_title, fontsize=FS_LABEL, labelpad=14,
                      color="#1a1a2e", fontweight="bold")

        if panel_idx == legend_idx:
            place_two_legends(ax, fig, methods_to_plot)

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=BG_FIGURE, edgecolor="none")
    plt.close(fig)
    print(f"[ok] Saved: {out_path}")


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot calibration metrics (ECE / log loss / Brier) vs "
                    "deferral fraction (L2A vs baselines)."
    )
    parser.add_argument("--l2a_csv", required=True,
                        help="L2A out-of-fold results (deferral_<annotator>.csv).")
    parser.add_argument("--baselines_csv", required=True,
                        help="Baseline calibration sweep CSV "
                             "(Method + TargetFraction + Uncal_/Cal_ metrics).")
    parser.add_argument("--output_dir", default="./figures",
                        help="Directory for the output PDF (created if absent).")
    parser.add_argument("--annotator", default="novice_1",
                        help="Annotator id, used in the output filename.")
    parser.add_argument("--ai_name", default="RF",
                        help="Prefix of the AI columns, e.g. 'RF' -> RF_prob.")
    parser.add_argument("--groundtruth_col", default="Event",
                        help="Ground-truth label column.")
    parser.add_argument("--baseline_mode", default="uncal",
                        choices=["uncal", "cal", "both"],
                        help="Which baseline calibration curves to draw.")
    return parser.parse_args()


def main():
    args = parse_args()

    global df_l2a, df_bl, y_true, l2a_calib, l2a_fracs
    global methods_to_plot, effective_mode, AI_PROB_COL, GROUNDTRUTH_COL

    AI_PROB_COL = f"{args.ai_name}_prob"
    GROUNDTRUTH_COL = args.groundtruth_col

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df_l2a = pd.read_csv(args.l2a_csv)
    df_bl = pd.read_csv(args.baselines_csv)
    y_true = df_l2a[GROUNDTRUTH_COL].values

    # Register fallback styling for any method present in the CSV but not configured.
    all_methods_in_csv = df_bl["Method"].unique().tolist()
    _fb_c = iter(_FALLBACK_COLORS)
    _fb_l = iter(_FALLBACK_LS)
    _fb_m = iter(_FALLBACK_MARKERS)
    for _m in all_methods_in_csv:
        if _m not in METHOD_CONFIG:
            METHOD_CONFIG[_m] = dict(enabled=True, label=_m.replace("_", " "),
                                     color=next(_fb_c, "#AAAAAA"), ls=next(_fb_l, "--"),
                                     marker=next(_fb_m, "o"), lw=1.8, markersize=8)

    methods_to_plot = [m for m in all_methods_in_csv if METHOD_CONFIG[m]["enabled"]]

    # Decide which baseline curves can actually be drawn given the CSV columns.
    bl_cols = df_bl.columns.tolist()
    has_uncal = any(c.startswith("Uncal_") for c in bl_cols)
    has_cal = any(c.startswith("Cal_") for c in bl_cols)

    effective_mode = args.baseline_mode
    if effective_mode == "both" and not (has_uncal and has_cal):
        effective_mode = "uncal" if has_uncal else "cal"
    if effective_mode == "uncal" and not has_uncal:
        effective_mode = "cal"
    if effective_mode == "cal" and not has_cal:
        effective_mode = "uncal"

    print(f"Methods in CSV       : {all_methods_in_csv}")
    print(f"Methods being plotted: {methods_to_plot}")
    print(f"Baseline mode        : {effective_mode}")

    # Compute the L2A calibration curve across deferral fractions.
    l2a_calib = {k: [] for k in ("ECE", "MCE", "Brier", "LogLoss")}
    l2a_fracs = []
    for frac in CALIB_FRACTIONS:
        col = AI_PROB_COL if frac == 0.0 else frac_to_col(frac)
        if col not in df_l2a.columns:
            print(f"  warning: '{col}' not found — skipping {frac:.2f}")
            continue
        m = compute_all_calib(y_true, df_l2a[col].values)
        for k in m:
            l2a_calib[k].append(m[k])
        l2a_fracs.append(frac * 100)
    print(f"L2A calibration at {len(l2a_fracs)} points")

    make_calib_figure(
        CALIB_METRICS,
        out_path=os.path.join(output_dir, f"calibration_all_{args.annotator}.pdf"),
        legend_idx=1,
    )


if __name__ == "__main__":
    main()
