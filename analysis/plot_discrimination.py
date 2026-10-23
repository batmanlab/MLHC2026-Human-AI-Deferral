"""
Discrimination-vs-deferral figure for the human-AI deferral study.

Produces the 2x3 metric grid as a function of deferral fraction, with L2A
highlighted against the hard- and soft-deferral baselines and the AI-only and
novice reference lines. Output is two panels saved as PDF:

    Panel A: AUPRC, AUROC, F1
    Panel B: precision, sensitivity, specificity

Inputs
------
    --deferral_csv   L2A out-of-fold results (deferral_<annotator>.csv from
                     l2a.py). Must contain the ground-truth column, the AI
                     probability/prediction columns, the novice column, and the
                     per-fraction gated_prob_d* / f1_pred_d* columns.
    --baselines_csv  Baseline metric sweep with a 'Method' column, a 'DeferRate'
                     column, and AUPRC/AUROC/F1/Precision/Recall/Specificity
                     columns (results_standard_sweep.csv from run_baselines.py,
                     combined with the DaF rows from the BeyondDefer run).

Usage
-----
    python plot_discrimination.py \\
        --deferral_csv deferral_novice_1.csv \\
        --baselines_csv baselines_novice_1.csv \\
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
from matplotlib.lines import Line2D
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

# =============================================================================
# DISPLAY CONFIGURATION
# =============================================================================

METHOD_MAIN_NAME = 'L2A (Ours)'

# Deferral fractions drawn for the L2A curve (0% is prepended as the AI-only point).
DEFERRAL_FRACTIONS_COARSE = [0.05, 0.10, 0.20, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Whether to draw AI-only / novice horizontal reference lines, per metric.
SHOW_ANNOTATIONS = False

SHOW_AI_BASELINE_AUPRC       = False
SHOW_AI_BASELINE_AUROC       = False
SHOW_AI_BASELINE_F1          = False
SHOW_AI_BASELINE_PRECISION   = False
SHOW_AI_BASELINE_RECALL      = False
SHOW_AI_BASELINE_SPECIFICITY = False

SHOW_HUMAN_BASELINE_F1          = True
SHOW_HUMAN_BASELINE_PRECISION   = True
SHOW_HUMAN_BASELINE_RECALL      = True
SHOW_HUMAN_BASELINE_SPECIFICITY = True

# =============================================================================
# METHOD CONFIG
# =============================================================================

SOFT_DEFER_KEYS = {'BeyondDefer', 'AdditionalBeyond'}

METHOD_CONFIG = {
    'RealizableSurrogate':  dict(enabled=True,  label='RS', color='#E53935', ls='--', marker='o', lw=2, markeredgewidth=2.0, markersize=10),
    'LceSurrogate':         dict(enabled=False, label='LCE',        color='#F4511E', ls='--', marker='o', lw=2, markeredgewidth=2.0, markersize=10),
    'OvASurrogate':         dict(enabled=True,  label='OvA',        color='#FFB300', ls='--', marker='o', lw=2, markeredgewidth=2.0, markersize=10),
    'CompareConfidence':    dict(enabled=True,  label='CC',   color='#43A047', ls='--', marker='o', lw=2, markeredgewidth=2.0, markersize=10),
    'SelectivePrediction':  dict(enabled=True,  label='SP', color='#00897B', ls='--', marker='o', lw=2, markeredgewidth=2.0, markersize=10),
    'MixOfExps':            dict(enabled=True,  label='MoE',       color='#8E24AA', ls='--', marker='o', lw=2, markeredgewidth=2.0, markersize=10),
    'DifferentiableTriage': dict(enabled=False, label='DT',color='#D81B60', ls='--', marker='o', lw=2, markeredgewidth=2.0, markersize=10),
    'BeyondDefer':          dict(enabled=True,  label='DaF-base',            color='#FF7043', ls='-',  marker='o', lw=3,
                                 markerfacecolor='white', markeredgecolor='#FF7043', markeredgewidth=2.0, markersize=10),
    'AdditionalBeyond':     dict(enabled=True,  label='DaF-ext',         color='#9A6324', ls='-',  marker='o', lw=3,
                                 markerfacecolor='white', markeredgecolor='#9A6324', markeredgewidth=2.0, markersize=10),
}

methods_to_plot = [k for k, v in METHOD_CONFIG.items() if v['enabled']]

# =============================================================================
# DESIGN TOKENS
# =============================================================================

COLOR_MAIN   = '#0288D1'
COLOR_AI     = '#C0392B'
COLOR_HUMAN  = '#1A7A4A'
COLOR_ANNOT  = '#2C3E6B'
BG_AXES      = '#F7F9FC'
BG_FIGURE    = '#FFFFFF'

COLOR_AI_LABEL    = '#C0392B'
COLOR_HUMAN_LABEL = '#1A7A4A'

REF_LINESTYLE  = (0, (7, 2, 1, 2))
REF_LINEWIDTH  = 2.6
REF_BAND_ALPHA = 0.06

L2A_GLOW_LAYERS = [(14, 0.04), (9, 0.08), (5, 0.14)]
L2A_CORE_LW     = 3.6

# Font sizes (tuned for a printed figure).
FS_TITLE      = 30   # metric name at the top of each subplot
FS_TICK       = 25   # x and y tick labels
FS_XLABEL     = 27   # "Deferral Fraction (%)"
FS_ANNOT_REF  = 20   # "human ->" / "^ AI"
FS_LEGEND     = 15   # legend text
FS_LEGEND_TTL = 15   # legend title

LEG_STYLE = dict(
    fontsize=FS_LEGEND, framealpha=0.93,
    facecolor='#F5F5F5', edgecolor='#999999',
    fancybox=False,
    borderpad=1.0, labelspacing=0.55,
    handlelength=3.5, handleheight=1.6,
)

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.edgecolor':     '#CCCCCC',
    'axes.linewidth':     1.0,
    'grid.color':         '#E3E8EF',
    'grid.linestyle':     '-',
    'grid.linewidth':     0.8,
    'xtick.color':        '#444444',
    'ytick.color':        '#444444',
    'xtick.direction':    'out',
    'ytick.direction':    'out',
})

# =============================================================================
# PLOT CONFIGURATION
# "Recall" is displayed as "Sensitivity"; the internal metric key stays 'Recall'.
# =============================================================================

plot_cfg_all = [
    ('AUPRC',       'AUPRC',        SHOW_AI_BASELINE_AUPRC,       False),
    ('AUROC',       'AUROC',        SHOW_AI_BASELINE_AUROC,       False),
    ('F1',          'F1 SCORE',     SHOW_AI_BASELINE_F1,          SHOW_HUMAN_BASELINE_F1),
    ('Precision',   'PRECISION',    SHOW_AI_BASELINE_PRECISION,   SHOW_HUMAN_BASELINE_PRECISION),
    ('Recall',      'SENSITIVITY',  SHOW_AI_BASELINE_RECALL,      SHOW_HUMAN_BASELINE_RECALL),
    ('Specificity', 'SPECIFICITY',  SHOW_AI_BASELINE_SPECIFICITY, SHOW_HUMAN_BASELINE_SPECIFICITY),
]
plot_cfg_A = plot_cfg_all[:3]   # AUPRC | AUROC | F1               (legend on the middle subplot)
plot_cfg_B = plot_cfg_all[3:]   # Precision | Sensitivity | Specificity (legend on the middle subplot)

# Metrics that get sparser y-tick labels (same treatment as Specificity).
SPARSE_YTICK_KEYS = {'AUPRC', 'F1', 'Precision'}

METRIC_COLS = ['AUPRC', 'AUROC', 'F1', 'Precision', 'Recall', 'Specificity']

# Data-dependent globals (populated in main from the input CSVs).
df = None
df_baselines = None
y_true = None
ai_base = None
human_base = None
HUMAN_COL = 'novice_1'
AI_PROB_COL = 'RF_prob'
AI_PRED_COL = 'RF_pred'
GROUNDTRUTH_COL = 'Event'
HUMAN_LABEL = 'novice_1'
AI_LABEL = 'RF (AI)'


# =============================================================================
# HELPERS
# =============================================================================

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def threshold_metrics(y_true, y_pred):
    return {
        'F1':          f1_score(y_true, y_pred, zero_division=0),
        'Precision':   precision_score(y_true, y_pred, zero_division=0),
        'Recall':      recall_score(y_true, y_pred, zero_division=0),
        'Specificity': specificity_score(y_true, y_pred),
    }


def _style_legend_title(leg):
    t = leg.get_title()
    t.set_fontweight('bold'); t.set_fontstyle('italic')
    t.set_fontsize(FS_LEGEND_TTL); t.set_color('#1a1a2e')


# =============================================================================
# L2A METRICS
# =============================================================================

def compute_metrics(deferral_fractions):
    """Compute L2A metrics across deferral fractions, with the AI-only point first.

    Uses the gated probability for ranking metrics (AUPRC/AUROC) and the
    F1-threshold prediction for the classification metrics.
    """
    m = {k: [] for k in ['AUPRC', 'AUROC', 'F1', 'Precision', 'Recall', 'Specificity']}
    for d in deferral_fractions:
        suffix = f"{d:.2f}".replace('0.', '').replace('.', '')
        col_prob = f'gated_prob_d{suffix}'
        col_pred = f'f1_pred_d{suffix}'
        missing = [c for c in (col_prob, col_pred) if c not in df.columns]
        if missing:
            raise ValueError(
                f"Deferral CSV is missing expected column(s) for fraction {d:.2f}: "
                f"{missing}. Was it produced by l2a.py including this fraction?"
            )
        y_prob = df[col_prob].values
        y_pred = df[col_pred].values
        m['AUPRC'].append(average_precision_score(y_true, y_prob))
        m['AUROC'].append(roc_auc_score(y_true, y_prob))
        tm = threshold_metrics(y_true, y_pred)
        for k in ['F1', 'Precision', 'Recall', 'Specificity']:
            m[k].append(tm[k])
    for k in m:
        m[k] = [ai_base[k]] + m[k]
    x_pct = [0.0] + [d * 100 for d in deferral_fractions]
    return m, x_pct


# =============================================================================
# BASELINE METHOD CURVES (with interpolation onto the plotted fractions)
# =============================================================================

def get_baseline_curve(method_name, deferral_fractions):
    sub = df_baselines[df_baselines['Method'] == method_name].copy()
    sub = sub.sort_values('DeferRate').reset_index(drop=True)
    targets = [0.0] + list(deferral_fractions)
    result = {k: [] for k in METRIC_COLS}
    for t in targets:
        row = sub[np.isclose(sub['DeferRate'].astype(float), t, atol=1e-6)]
        if not row.empty:
            for k in METRIC_COLS:
                result[k].append(float(row.iloc[0][k]))
        else:
            lower = sub[sub['DeferRate'] <= t]
            upper = sub[sub['DeferRate'] >= t]
            if lower.empty:
                for k in METRIC_COLS:
                    result[k].append(float(upper.iloc[0][k]))
            elif upper.empty:
                for k in METRIC_COLS:
                    result[k].append(float(lower.iloc[-1][k]))
            else:
                r0, r1 = lower.iloc[-1], upper.iloc[0]
                x0, x1 = r0['DeferRate'], r1['DeferRate']
                w = (t - x0) / (x1 - x0) if x1 != x0 else 0.0
                for k in METRIC_COLS:
                    result[k].append(float(r0[k]) + w * (float(r1[k]) - float(r0[k])))
    x_pct = [v * 100 for v in targets]
    return x_pct, result


# =============================================================================
# CORE SUBPLOT DRAWING
# =============================================================================

def draw_subplot(ax, key, title, show_ai, show_human,
                 metrics, x_pct, baseline_curves, is_fine,
                 show_legend=False, show_xlabel=False):
    """Draw one metric panel: baselines, reference lines, and the L2A hero curve.

    show_xlabel draws the "Deferral Fraction (%)" axis label (only the middle
    subplot in each panel uses it).
    """
    vals = metrics[key]
    x_arr = np.array(x_pct)
    v_arr = np.array(vals)
    x_max = max(x_pct)
    fine_label_every = 5

    ax.set_facecolor(BG_AXES)
    ax.grid(True, zorder=0)

    # Y limits from all series drawn in this panel.
    all_vals = list(v_arr)
    if show_ai:                          all_vals.append(ai_base[key])
    if show_human and key in human_base: all_vals.append(human_base[key])
    for mth in methods_to_plot:
        all_vals.extend(baseline_curves[mth][1][key])
    data_min = min(all_vals)
    data_max = max(all_vals)
    span = max(data_max - data_min, 0.05)
    y_lo = max(0.0, data_min - span * 0.12)
    y_hi = min(1.00, data_max + span * 0.22)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-x_max * 0.04, x_max * 1.05)
    x_left, _ = ax.get_xlim()

    # Vertical separator at the 0% (AI-only) operating point.
    ax.axvline(x=0, color=COLOR_AI_LABEL, linewidth=1.6, linestyle='--', alpha=0.45, zorder=1)

    # AI-only / novice reference lines.
    if show_ai:
        bval = ai_base[key]
        ax.axhspan(bval - span*0.008, bval + span*0.008,
                   color=COLOR_AI, alpha=REF_BAND_ALPHA, zorder=1)
        ax.axhline(bval, color=COLOR_AI, linewidth=REF_LINEWIDTH,
                   linestyle=REF_LINESTYLE, alpha=0.95, zorder=3,
                   label=f'{AI_LABEL} ({bval:.3f})')

    if show_human and key in human_base:
        bval = human_base[key]
        ax.axhspan(bval - span*0.008, bval + span*0.008,
                   color=COLOR_HUMAN, alpha=REF_BAND_ALPHA, zorder=1)
        ax.axhline(bval, color=COLOR_HUMAN, linewidth=REF_LINEWIDTH,
                   linestyle=REF_LINESTYLE, alpha=0.95, zorder=3,
                   label=f'{HUMAN_LABEL} ({bval:.3f})')
        ax.annotate(
            'human \u2192',
            xy=(x_left, bval),
            xytext=(-8, 0), textcoords='offset points',
            fontsize=FS_ANNOT_REF, color=COLOR_HUMAN_LABEL,
            fontweight='bold', fontstyle='italic',
            ha='right', va='center', annotation_clip=False,
        )

    # Comparison baseline method curves.
    for method in methods_to_plot:
        cfg = METHOD_CONFIG[method]
        bx, bm = baseline_curves[method]
        ax.plot(np.array(bx), np.array(bm[key]),
                color=cfg['color'], linewidth=cfg['lw'],
                linestyle=cfg['ls'],
                marker=cfg['marker'],
                markersize=cfg.get('markersize', 7),
                markerfacecolor=cfg.get('markerfacecolor', cfg['color']),
                markeredgecolor=cfg.get('markeredgecolor', 'white'),
                markeredgewidth=cfg.get('markeredgewidth', 0.8),
                label=cfg['label'],
                zorder=3, alpha=0.88)

    # Soft fill under the L2A curve.
    ax.fill_between(x_arr, v_arr, y_lo, alpha=0.07, color=COLOR_MAIN, zorder=2)

    # L2A: glow halo layers, then the solid core.
    for glow_lw, glow_alpha in L2A_GLOW_LAYERS:
        ax.plot(x_arr, v_arr, color=COLOR_MAIN, linewidth=glow_lw,
                linestyle='-', alpha=glow_alpha, zorder=4, solid_capstyle='round')

    ax.plot(x_arr, v_arr,
            color=COLOR_MAIN, linewidth=L2A_CORE_LW, zorder=5,
            label=METHOD_MAIN_NAME,
            solid_capstyle='round', solid_joinstyle='round')

    if is_fine:
        milestone_x = [xv for xv in x_pct if xv % 5 == 0]
        milestone_v = [vals[x_pct.index(xv)] for xv in milestone_x]
        ax.scatter(milestone_x, milestone_v, s=100, zorder=6,
                   facecolors='white', edgecolors=COLOR_MAIN, linewidths=2.2)
    else:
        ax.scatter(x_arr, v_arr, s=100, zorder=6,
                   facecolors='white', edgecolors=COLOR_MAIN, linewidths=2.4)

    # Optional value annotations.
    if SHOW_ANNOTATIONS:
        pts = ([(xv, v) for xv, v in zip(x_pct, vals) if xv % 5 == 0]
               if is_fine else list(zip(x_pct, vals)))
        for xv, v in pts:
            ax.annotate(f'{v:.2f}', xy=(xv, v),
                        xytext=(0, 14), textcoords='offset points',
                        ha='center', fontsize=13, color=COLOR_ANNOT, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.65))

    # X-axis ticks.
    if is_fine:
        maj = [x for x in np.arange(0, x_max + 0.01, fine_label_every) if x <= x_max + 0.01]
        ax.set_xticks(maj)
        ax.set_xticklabels([f'{int(x)}%' for x in maj],
                           fontsize=FS_TICK, fontweight='bold', rotation=0, ha='center')
    else:
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels([f'{x}%' for x in [0, 25, 50, 75, 100]],
                           fontsize=FS_TICK, fontweight='bold', rotation=0, ha='center')

    # Y-axis: 3 decimals for Specificity; sparser ticks for AUPRC/F1/Precision.
    y_fmt = '%.3f' if key == 'Specificity' else '%.2f'
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(y_fmt))
    if key in {'Specificity'} | SPARSE_YTICK_KEYS:
        current_ticks = ax.yaxis.get_major_locator().tick_values(y_lo, y_hi)
        ax.yaxis.set_major_locator(
            mticker.MaxNLocator(
                nbins=max(1, len(current_ticks) // 2),
                steps=[1, 2, 5, 10]
            )
        )
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.tick_params(axis='x', labelsize=FS_TICK)

    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=14,
                 color='#1a1a2e', loc='left')

    if show_xlabel:
        ax.set_xlabel('Deferral Fraction (%)', fontsize=FS_XLABEL, labelpad=16,
                      color='#333', fontweight='bold')
    else:
        ax.set_xlabel('')

    ax.set_ylabel('')

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

    ax.annotate(
        '\u2191\nAI',
        xy=(0, y_lo), xytext=(0, -38), textcoords='offset points',
        fontsize=FS_ANNOT_REF, color=COLOR_AI_LABEL,
        fontweight='bold', fontstyle='italic',
        ha='center', va='top', annotation_clip=False,
    )

    # Legends are drawn only on the designated middle subplot.
    if not show_legend:
        if ax.get_legend():
            ax.legend().set_visible(False)
        return

    all_h, all_l = ax.get_legend_handles_labels()

    def _rebuild(h, lbl):
        if lbl == METHOD_MAIN_NAME:
            return Line2D([0], [0], color=COLOR_MAIN, linewidth=4.0, linestyle='-',
                          marker='o', markersize=10,
                          markerfacecolor='white', markeredgecolor=COLOR_MAIN,
                          markeredgewidth=2.0)
        if lbl.startswith(HUMAN_LABEL):
            return Line2D([0], [0], color=COLOR_HUMAN,
                          linewidth=REF_LINEWIDTH, linestyle=REF_LINESTYLE)
        if lbl.startswith(AI_LABEL):
            return Line2D([0], [0], color=COLOR_AI,
                          linewidth=REF_LINEWIDTH, linestyle=REF_LINESTYLE)
        mk = next((k for k, v in METHOD_CONFIG.items() if v['label'] == lbl), None)
        if mk:
            c = METHOD_CONFIG[mk]
            return Line2D([0], [0],
                          color=c['color'], linewidth=c['lw'], linestyle=c['ls'],
                          marker=c['marker'],
                          markersize=c.get('markersize', 8),
                          markerfacecolor=c.get('markerfacecolor', c['color']),
                          markeredgecolor=c.get('markeredgecolor', 'white'),
                          markeredgewidth=c.get('markeredgewidth', 0.8))
        return h

    soft_h, soft_l = [], []
    hard_h, hard_l = [], []

    l2a_idx = next((i for i, l in enumerate(all_l) if l == METHOD_MAIN_NAME), None)
    if l2a_idx is not None:
        soft_h.append(_rebuild(all_h[l2a_idx], METHOD_MAIN_NAME))
        soft_l.append(METHOD_MAIN_NAME)

    for h, lbl in zip(all_h, all_l):
        if lbl == METHOD_MAIN_NAME:
            continue
        if lbl.startswith(HUMAN_LABEL):
            continue
        mk = next((k for k, v in METHOD_CONFIG.items() if v['label'] == lbl), None)
        if mk and mk in SOFT_DEFER_KEYS:
            soft_h.append(_rebuild(h, lbl)); soft_l.append(lbl)
        else:
            hard_h.append(_rebuild(h, lbl)); hard_l.append(lbl)

    fig = ax.get_figure()

    leg_soft = ax.legend(soft_h, soft_l, title='\u25cb  Soft Deferral',
                         loc='lower right',
                         bbox_to_anchor=(0.95, 0.0),
                         bbox_transform=ax.transAxes,
                         **LEG_STYLE)
    _style_legend_title(leg_soft)
    leg_soft.get_frame().set_linewidth(1.0)
    leg_soft.get_texts()[0].set_fontweight('bold')
    for txt in leg_soft.get_texts():
        txt.set_color('#222222')
    ax.add_artist(leg_soft)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    soft_bb = leg_soft.get_window_extent(renderer)
    x_left_leg, _ = ax.transAxes.inverted().transform((soft_bb.x0, 0))

    if hard_h:
        leg_hard = ax.legend(hard_h, hard_l, title='\u25cf  Hard Deferral',
                             loc='lower right',
                             bbox_to_anchor=(x_left_leg - 0.01, 0.0),
                             bbox_transform=ax.transAxes,
                             **LEG_STYLE)
        _style_legend_title(leg_hard)
        leg_hard.get_frame().set_linewidth(1.0)
        for txt in leg_hard.get_texts():
            txt.set_color('#222222')


# =============================================================================
# FIGURE BUILDER
# =============================================================================

def make_plot(metrics, x_pct, baseline_curves,
              output_path_A, output_path_B, is_fine=False):
    """Render panels A and B and save each as PDF."""
    for cfg, out_path, legend_idx in [
        (plot_cfg_A, output_path_A, 1),
        (plot_cfg_B, output_path_B, 1),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(26, 9), facecolor=BG_FIGURE)
        fig.subplots_adjust(top=0.92, bottom=0.14, left=0.07, right=0.97,
                            hspace=0.4, wspace=0.38)

        for idx, (ax, (key, title, show_ai, show_human)) in enumerate(zip(axes, cfg)):
            draw_subplot(
                ax, key, title, show_ai, show_human,
                metrics, x_pct, baseline_curves, is_fine,
                show_legend=(idx == legend_idx),
                show_xlabel=(idx == legend_idx),
            )

        fig.savefig(out_path, dpi=300, bbox_inches='tight',
                    facecolor=BG_FIGURE, edgecolor='none')
        plt.close(fig)
        print(f"[ok] Saved: {out_path}")


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot discrimination metrics vs deferral fraction "
                    "(L2A vs baselines)."
    )
    parser.add_argument("--deferral_csv", required=True,
                        help="L2A out-of-fold results (deferral_<annotator>.csv).")
    parser.add_argument("--baselines_csv", required=True,
                        help="Baseline metric sweep CSV (Method + DeferRate + metrics).")
    parser.add_argument("--output_dir", default="./figures",
                        help="Directory for the output PDFs (created if absent).")
    parser.add_argument("--annotator", default="novice_1",
                        help="Novice column name in the deferral CSV; also used "
                             "in output filenames and the reference-line label.")
    parser.add_argument("--ai_name", default="RF",
                        help="Prefix of the AI columns, e.g. 'RF' -> RF_prob, RF_pred.")
    parser.add_argument("--groundtruth_col", default="Event",
                        help="Ground-truth label column.")
    return parser.parse_args()


def main():
    args = parse_args()

    global df, df_baselines, y_true, ai_base, human_base
    global HUMAN_COL, AI_PROB_COL, AI_PRED_COL, GROUNDTRUTH_COL, HUMAN_LABEL, AI_LABEL

    HUMAN_COL = args.annotator
    GROUNDTRUTH_COL = args.groundtruth_col
    AI_PROB_COL = f"{args.ai_name}_prob"
    AI_PRED_COL = f"{args.ai_name}_pred"
    HUMAN_LABEL = args.annotator
    AI_LABEL = f"{args.ai_name} (AI)"

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.deferral_csv)
    y_true = df[GROUNDTRUTH_COL].values
    df_baselines = pd.read_csv(args.baselines_csv)

    # Fixed AI-only and novice-only reference metrics.
    ai_base = {
        'AUPRC': average_precision_score(y_true, df[AI_PROB_COL].values),
        'AUROC': roc_auc_score(y_true, df[AI_PROB_COL].values),
        **threshold_metrics(y_true, df[AI_PRED_COL].values),
    }
    human_base = threshold_metrics(y_true, df[HUMAN_COL].values)

    metrics_coarse, x_coarse = compute_metrics(DEFERRAL_FRACTIONS_COARSE)
    baseline_curves_coarse = {m: get_baseline_curve(m, DEFERRAL_FRACTIONS_COARSE)
                              for m in methods_to_plot}

    make_plot(
        metrics_coarse, x_coarse, baseline_curves_coarse,
        output_path_A=os.path.join(output_dir, f'deferral_metrics_coarse_{args.annotator}_A.pdf'),
        output_path_B=os.path.join(output_dir, f'deferral_metrics_coarse_{args.annotator}_B.pdf'),
        is_fine=False,
    )


if __name__ == "__main__":
    main()
