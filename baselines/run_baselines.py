"""
Human-AI deferral baselines for intraoperative ischemia detection.

Runs five deferral baselines on the interval-level cEEG table and, for each,
sweeps the deferral fraction from 0% to 100%:

    RS   Realizable Surrogate
    OvA  One-vs-All Surrogate
    CC   Compare Confidence
    SP   Selective Prediction
    MoE  Mixture-of-Experts

These implementations are **not vendored here**. They are imported from a local
clone of the human-AI deferral benchmark, passed via ``--repo_path``:

    https://github.com/clinicalml/human_ai_deferral

(As of this writing that repository carries no open-source license, so it
cannot be redistributed; clone it yourself and point ``--repo_path`` at it.)

The two soft-deferral baselines reported in the paper, **DaF-base** and
**DaF-ext**, are produced separately from the BeyondDefer repository
(the ``BeyondDefer`` and ``AdditionalBeyond`` classes correspond to DaF-base and
DaF-ext respectively) and are out of scope for this script:

    https://github.com/AminChrs/BeyondDefer

AI backbone
-----------
SP and CC use the fixed, precomputed AI probabilities already in the input
table (``<ai_name>_prob`` / ``<ai_name>_pred``; Random Forest in the paper) and
do not retrain the AI. RS, OvA, and MoE learn their own AI component with a
two-hidden-layer MLP (256 and 128 units). This mirrors the paper's
method-matched evaluation (Appendix G).

Inputs (CSV)
------------
    Case ID, Interval        composite key (unique per row)
    Event                    ground-truth ischemia label (0/1)
    fold                     patient-level CV fold id (expects folds 0..N-1)
    <ai_name>_prob/_pred     fixed AI probability / hard prediction
    <novice column>          novice binary prediction (0/1)
    111 cEEG feature columns the MLP-based methods train on

Outputs (under --output_dir)
----------------------------
    row_deferral_<method>.csv       per-row deferral decisions for every
                                    fraction (consumed by the contiguity analysis)
    results_standard_sweep.csv      discrimination metrics vs deferral fraction
    results_calibration_sweep.csv   calibration metrics vs deferral fraction
    plots/                          diagnostic curves (headless PNGs)

Each method is run for a single novice per invocation (``--annotator``); run it
once per novice.

Usage
-----
    python run_baselines.py \\
        --csv data_mlhc.csv \\
        --repo_path /path/to/human_ai_deferral \\
        --output_dir ./output/baselines_novice_1 \\
        --annotator novice_1

Requires PyTorch (for the MLP-based methods).
"""

import argparse
import logging
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix,
    brier_score_loss, log_loss,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# RUN CONFIGURATION
# =============================================================================
# Paths and the per-run annotator / AI-backbone name come from the command
# line (see parse_args). The settings below control which outputs are produced
# and the training hyperparameters; edit them here if needed.

SEED = 42

# Composite primary key in the input table.
CASE_ID_COL = "Case ID"
INTERVAL_COL = "Interval"

# Which outputs to produce.
RUN_STANDARD_METRICS = True
STANDARD_METRIC_PLOTS = ["AUPRC", "AUROC", "F1", "Accuracy", "Precision", "Recall", "Specificity"]
RUN_CALIB_UNCAL = True    # uncalibrated calibration metrics
RUN_CALIB_CAL = False     # beta-recalibrated calibration metrics
CALIB_USE_INNER_CV = False
CALIB_N_BINS = 15

# Which baselines to run. LceSurrogate and DifferentiableTriage are off by
# default (not reported in the paper).
RUN_METHODS = {
    "RealizableSurrogate":  True,
    "LceSurrogate":         False,
    "OvASurrogate":         True,
    "CompareConfidence":    True,
    "SelectivePrediction":  True,
    "MixOfExps":            True,
    "DifferentiableTriage": False,
}

# AI-backbone column names in the input table (bound from --ai_name in main).
AI_PROB_COL = "RF_prob"
AI_PRED_COL = "RF_pred"

# Methods that use the fixed precomputed AI probabilities rather than training
# their own AI component (informational; dispatch below is explicit).
FIXED_AI_METHODS = {"SelectivePrediction", "CompareConfidence"}

# Column names bound from the command line in main().
LABEL_COL = "Event"
HUMAN_COL = "novice_1"
FOLD_COL = "fold"

# Model / training settings.
N_FOLDS = 10
N_CLASSES = 2
INPUT_DIM = 111
HIDDEN_DIM = 256
BATCH_SIZE = 256
LR = 5e-4
EPOCHS = 100
TEST_INTV = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# DEFERRAL / CALIBRATION FRACTIONS
# =============================================================================

DEFERRAL_FRACTIONS = [
    0.0, 0.05, 0.10, 0.15, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 1.00,
]

CALIB_FRACTIONS = [round(i * 0.05, 2) for i in range(21)]


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# UPSTREAM REPOSITORY IMPORTS
# =============================================================================

def _import_repo():
    """Import the enabled baseline classes from the cloned upstream repo.

    ``--repo_path`` must already be on ``sys.path`` (set up in main). Methods
    whose import fails are reported and skipped rather than aborting the run.
    """
    errors, modules = [], {}

    try:
        from methods.realizable_surrogate import RealizableSurrogate
        modules["RealizableSurrogate"] = RealizableSurrogate
    except ImportError as e:
        errors.append(f"RealizableSurrogate: {e}")
    try:
        from baselines.lce_surrogate import LceSurrogate
        modules["LceSurrogate"] = LceSurrogate
    except ImportError as e:
        errors.append(f"LceSurrogate: {e}")
    try:
        from baselines.one_v_all import OVASurrogate
        modules["OvASurrogate"] = OVASurrogate
    except ImportError as e:
        errors.append(f"OvASurrogate: {e}")
    try:
        from baselines.compare_confidence import CompareConfidence
        modules["CompareConfidence"] = CompareConfidence
    except ImportError as e:
        errors.append(f"CompareConfidence: {e}")
    try:
        from baselines.selective_prediction import SelectivePrediction
        modules["SelectivePrediction"] = SelectivePrediction
    except ImportError as e:
        errors.append(f"SelectivePrediction: {e}")
    try:
        from baselines.mix_of_exps import MixtureOfExperts
        modules["MixOfExps"] = MixtureOfExperts
    except ImportError as e:
        errors.append(f"MixOfExps: {e}")
    for mod_name in ("baselines.differentiable_triage", "baselines.diff_triage"):
        try:
            import importlib
            m = importlib.import_module(mod_name)
            modules["DifferentiableTriage"] = m.DifferentiableTriage
            break
        except (ImportError, AttributeError):
            pass
    if "DifferentiableTriage" not in modules:
        errors.append("DifferentiableTriage: could not import")

    if errors:
        print("Some imports failed (those methods will be skipped):")
        for e in errors:
            print(f"   {e}")

    modules = {k: v for k, v in modules.items() if RUN_METHODS.get(k, False)}
    skipped = [k for k, v in RUN_METHODS.items() if not v]
    print(f"Active methods      : {list(modules.keys())}")
    print(f"Skipped (toggle off): {skipped}\n")
    return modules


# =============================================================================
# FEATURE COLUMNS
# =============================================================================
# The 111 cEEG features, in the order the MLP-based methods expect.

FEATURE_COLS = [
    "Delta_F3-P3", "Delta_P3-O1", "Delta_F3-T3", "Delta_T3-O1",
    "Delta_F4-P4", "Delta_P4-O2", "Delta_F4-T4", "Delta_T4-O2",
    "Theta_F3-P3", "Theta_P3-O1", "Theta_F3-T3", "Theta_T3-O1",
    "Theta_F4-P4", "Theta_P4-O2", "Theta_F4-T4", "Theta_T4-O2",
    "Alpha_F3-P3", "Alpha_P3-O1", "Alpha_F3-T3", "Alpha_T3-O1",
    "Alpha_F4-P4", "Alpha_P4-O2", "Alpha_F4-T4", "Alpha_T4-O2",
    "Beta_F3-P3", "Beta_P3-O1", "Beta_F3-T3", "Beta_T3-O1",
    "Beta_F4-P4", "Beta_P4-O2", "Beta_F4-T4", "Beta_T4-O2",
    "Gamma_F3-P3", "Gamma_P3-O1", "Gamma_F3-T3", "Gamma_T3-O1",
    "Gamma_F4-P4", "Gamma_P4-O2", "Gamma_F4-T4", "Gamma_T4-O2",
    "ADR_F3-P3", "ADR_P3-O1", "ADR_F3-T3", "ADR_T3-O1",
    "ADR_F4-P4", "ADR_P4-O2", "ADR_F4-T4", "ADR_T4-O2",
    "BDR_F3-P3", "BDR_P3-O1", "BDR_F3-T3", "BDR_T3-O1",
    "BDR_F4-P4", "BDR_P4-O2", "BDR_F4-T4", "BDR_T4-O2",
    "ABDTR_F3-P3", "ABDTR_P3-O1", "ABDTR_F3-T3", "ABDTR_T3-O1",
    "ABDTR_F4-P4", "ABDTR_P4-O2", "ABDTR_F4-T4", "ABDTR_T4-O2",
    "SEF90_F3-P3", "SEF90_P3-O1", "SEF90_F3-T3", "SEF90_T3-O1",
    "SEF90_F4-P4", "SEF90_P4-O2", "SEF90_F4-T4", "SEF90_T4-O2",
    "aEEG_F3-P3", "aEEG_P3-O1", "aEEG_F3-T3", "aEEG_T3-O1",
    "aEEG_F4-P4", "aEEG_P4-O2", "aEEG_F4-T4", "aEEG_T4-O2",
    "Delta_1", "Theta_1", "Alpha_1", "Beta_1", "Gamma_1",
    "ADR_1", "BDR_1", "ABDTR_1", "SEF90_1", "aEEG_1",
    "Delta_2", "Theta_2", "Alpha_2", "Beta_2", "Gamma_2",
    "ADR_2", "BDR_2", "ABDTR_2", "SEF90_2", "aEEG_2",
    "Delta_0", "Theta_0", "Alpha_0", "Beta_0", "Gamma_0",
    "ADR_0", "BDR_0", "ABDTR_0", "SEF90_0", "aEEG_0",
    "Asymmetry",
]


# =============================================================================
# MODEL
# =============================================================================

class MLPClassifier(nn.Module):
    """Two-hidden-layer MLP (256 -> 128) with batch norm, ReLU, and dropout."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.BatchNorm1d(HIDDEN_DIM // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM // 2, output_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def _add_grad_clip(model, max_norm=1.0):
    """Register a hook clamping each parameter's gradient to [-max_norm, max_norm]."""
    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(lambda g: torch.clamp(g, -max_norm, max_norm))


def _make_surrogate_model():
    m = MLPClassifier(N_CLASSES + 1).to(DEVICE); _add_grad_clip(m); return m


def _make_task_classifier():
    m = MLPClassifier(N_CLASSES).to(DEVICE); _add_grad_clip(m); return m


def _make_binary_head():
    m = MLPClassifier(2).to(DEVICE); _add_grad_clip(m); return m


# =============================================================================
# DATA UTILITIES
# =============================================================================

def make_loader(X, y, h, shuffle=False, drop_last=False):
    ds = TensorDataset(
        torch.FloatTensor(X), torch.LongTensor(y), torch.LongTensor(h)
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=drop_last)


def get_splits(df, test_fold):
    """Train / validation / test split for one outer fold.

    The validation fold is the next fold (cyclically); the rest are training.
    """
    val_fold = (test_fold + 1) % N_FOLDS
    return (df[~(df[FOLD_COL].isin([test_fold, val_fold]))],
            df[df[FOLD_COL] == val_fold],
            df[df[FOLD_COL] == test_fold])


def _impute_with_medians(arr, medians):
    arr = arr.copy().astype(float)
    arr[~np.isfinite(arr)] = np.nan
    r, c = np.where(np.isnan(arr))
    arr[r, c] = medians[c]
    return arr


def build_loaders(train_df, val_df, test_df):
    """Build train/val/test loaders with median imputation and standardization.

    Returns the three loaders, the test/val labels and test novice labels, and
    the standardized test feature matrix (used by the fixed-backbone methods).
    """
    raw = {s: df[FEATURE_COLS].values.astype(float)
           for s, df in [("tr", train_df), ("va", val_df), ("te", test_df)]}

    X_nan = raw["tr"].copy(); X_nan[~np.isfinite(X_nan)] = np.nan
    meds = np.nanmedian(X_nan, axis=0)
    meds = np.where(np.isfinite(meds), meds, 0.0)

    scaler = StandardScaler().fit(_impute_with_medians(raw["tr"], meds))
    X = {s: np.clip(scaler.transform(_impute_with_medians(raw[s], meds)), -10, 10)
         for s in ("tr", "va", "te")}

    y = {s: df[LABEL_COL].values.astype(int)
         for s, df in [("tr", train_df), ("va", val_df), ("te", test_df)]}
    h = {s: df[HUMAN_COL].values.astype(int)
         for s, df in [("tr", train_df), ("va", val_df), ("te", test_df)]}

    return (
        make_loader(X["tr"], y["tr"], h["tr"], shuffle=True, drop_last=True),
        make_loader(X["va"], y["va"], h["va"]),
        make_loader(X["te"], y["te"], h["te"]),
        y["te"], h["te"], y["va"],
        X["te"],
    )


# =============================================================================
# STANDARD METRICS
# =============================================================================

def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp + 1e-9)


def compute_metrics_dict(y_true, y_pred_hard, y_prob_soft):
    y_true = np.array(y_true)
    y_pred_hard = np.clip(np.array(y_pred_hard), 0, N_CLASSES - 1)
    y_prob_soft = np.clip(np.array(y_prob_soft), 0.0, 1.0)
    avg = "binary" if N_CLASSES == 2 else "macro"
    try:
        auroc = roc_auc_score(y_true, y_prob_soft)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, y_prob_soft)
    except Exception:
        auprc = float("nan")
    return {
        "AUPRC":       auprc,
        "AUROC":       auroc,
        "Accuracy":    accuracy_score(y_true, y_pred_hard),
        "F1":          f1_score(y_true, y_pred_hard, average=avg, zero_division=0),
        "Precision":   precision_score(y_true, y_pred_hard, average=avg, zero_division=0),
        "Recall":      recall_score(y_true, y_pred_hard, average=avg, zero_division=0),
        "Specificity": specificity_score(y_true, y_pred_hard),
    }


# =============================================================================
# CALIBRATION METRICS
# =============================================================================

def ece_score(y_true, y_prob, n_bins=CALIB_N_BINS):
    """Expected calibration error."""
    y_true = np.array(y_true); y_prob = np.clip(np.array(y_prob), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def mce_score(y_true, y_prob, n_bins=CALIB_N_BINS):
    """Maximum calibration error."""
    y_true = np.array(y_true); y_prob = np.clip(np.array(y_prob), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(y_true[mask].mean() - y_prob[mask].mean()))
    return mce


def compute_calib_metrics(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.clip(np.array(y_prob), 1e-7, 1 - 1e-7)
    return {
        "ECE":     ece_score(y_true, y_prob),
        "MCE":     mce_score(y_true, y_prob),
        "Brier":   brier_score_loss(y_true, y_prob),
        "LogLoss": log_loss(y_true, y_prob),
    }


# =============================================================================
# BETA CALIBRATION
# =============================================================================

class BetaCalibrator:
    """Beta calibration via logistic regression on log-probabilities."""

    def __init__(self):
        self._lr = None
        self._fitted = False

    def fit(self, probs, labels):
        probs = np.clip(np.array(probs, dtype=float), 1e-7, 1 - 1e-7)
        labels = np.array(labels, dtype=int)
        if len(np.unique(labels)) < 2:
            return self
        X = np.column_stack([np.log(probs), np.log(1 - probs)])
        try:
            self._lr = LogisticRegression(max_iter=1000, C=1.0)
            self._lr.fit(X, labels)
            self._fitted = True
        except Exception:
            pass
        return self

    def calibrate(self, probs):
        probs = np.clip(np.array(probs, dtype=float), 1e-7, 1 - 1e-7)
        if not self._fitted:
            return probs
        X = np.column_stack([np.log(probs), np.log(1 - probs)])
        return self._lr.predict_proba(X)[:, 1]


def _get_val_fold_calib_data(method, val_loader, y_val):
    """Validation-fold class-1 probabilities used to fit the beta calibrator."""
    test_out = method.test(val_loader)
    raw = np.array(test_out.get("class_probs",
                                test_out.get("model_probs", [])))
    if raw.ndim == 2 and raw.shape[1] >= N_CLASSES:
        cp = raw[:, :N_CLASSES]
        sums = np.where(cp.sum(1, keepdims=True) < 1e-10, 1.0, cp.sum(1, keepdims=True))
        return (cp / sums)[:, 1], y_val
    preds = np.clip(np.array(test_out["preds"]), 0, N_CLASSES - 1)
    return preds.astype(float), y_val


# =============================================================================
# PARSE UPSTREAM TEST OUTPUT
# =============================================================================

def parse_test_output(test_out, h_test):
    """Extract (hard preds, class-1 AI probs, rejection score) from a method.

    Falls back to deriving probabilities / rejection scores from the hard
    predictions when a method does not expose class probabilities or a
    rejection score.
    """
    model_preds = np.clip(np.array(test_out["preds"]), 0, N_CLASSES - 1)
    raw = np.array(test_out.get("class_probs",
                                test_out.get("model_probs", [])))

    if raw.ndim == 2 and raw.shape[1] >= N_CLASSES:
        cp = raw[:, :N_CLASSES]
        sums = np.where(cp.sum(1, keepdims=True) < 1e-10, 1.0, cp.sum(1, keepdims=True))
        ai_probs = (cp / sums)[:, 1]
    else:
        ai_probs = model_preds.astype(float)

    if "rej_score" in test_out and len(test_out["rej_score"]) == len(model_preds):
        rej_score = np.array(test_out["rej_score"])
    else:
        if raw.ndim == 2 and raw.shape[1] >= N_CLASSES:
            cp = raw[:, :N_CLASSES]
            sums = np.where(cp.sum(1, keepdims=True) < 1e-10, 1.0, cp.sum(1, keepdims=True))
            rej_score = 1.0 - (cp / sums).max(1)
        else:
            rej_score = 1.0 - ai_probs

    return model_preds, ai_probs, rej_score


# =============================================================================
# FIXED-AI BACKBONE — SELECTIVE PREDICTION
# =============================================================================

def run_fixed_ai_selective_prediction(test_df, val_df, y_val):
    """SP on the fixed AI probabilities: no training; defer on low AI confidence."""
    ai_probs = test_df[AI_PROB_COL].values.astype(float)
    model_preds = test_df[AI_PRED_COL].values.astype(int)
    rej_score = 1.0 - np.maximum(ai_probs, 1.0 - ai_probs)
    val_ai_probs = val_df[AI_PROB_COL].values.astype(float)
    return model_preds, ai_probs, rej_score, val_ai_probs, y_val


# =============================================================================
# FIXED-AI BACKBONE — COMPARE CONFIDENCE
# =============================================================================

def _train_expert_model(train_loader, val_loader):
    """Train a network to predict the novice's label (the 'expert' proxy)."""
    import copy
    set_seed(SEED)
    model = _make_binary_head()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X_batch, _y, h_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            h_batch = h_batch.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X_batch), h_batch).backward()
            optimizer.step()

        if epoch % TEST_INTV == 0 and epoch > 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, _y, h_batch in val_loader:
                    logits = model(X_batch.to(DEVICE))
                    preds = logits.argmax(1).cpu()
                    correct += (preds == h_batch).sum().item()
                    total += len(h_batch)
            val_acc = correct / total
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _expert_confidence(model, X):
    """Max softmax probability of the expert-proxy model over inputs X."""
    model.eval()
    ds = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    confs = []
    for (X_batch,) in loader:
        probs = torch.softmax(model(X_batch.to(DEVICE)), dim=1).cpu().numpy()
        confs.extend(probs.max(1).tolist())
    return np.array(confs)


def run_fixed_ai_compare_confidence(train_loader, val_loader, test_df, val_df,
                                    y_val, X_te):
    """CC with the fixed AI backbone: defer where the expert proxy is more
    confident than the AI."""
    expert_model = _train_expert_model(train_loader, val_loader)
    ai_probs = test_df[AI_PROB_COL].values.astype(float)
    model_preds = test_df[AI_PRED_COL].values.astype(int)
    ai_conf = np.maximum(ai_probs, 1.0 - ai_probs)
    expert_conf = _expert_confidence(expert_model, X_te)
    rej_score = expert_conf - ai_conf
    val_ai_probs = val_df[AI_PROB_COL].values.astype(float)
    return model_preds, ai_probs, rej_score, val_ai_probs, y_val


# =============================================================================
# METHOD RUNNER (MLP-based)
# =============================================================================

FIT_KWARGS = dict(
    epochs=EPOCHS, optimizer=optim.Adam, scheduler=None,
    lr=LR, verbose=False, test_interval=TEST_INTV,
)
FIT_KWARGS_PLAIN = dict(
    epochs=EPOCHS, optimizer=optim.Adam, lr=LR,
    scheduler=None, verbose=False, test_interval=TEST_INTV,
)


def _build_and_train(name, cls, train_loader, val_loader, test_loader):
    """Construct and fit an MLP-based baseline, handling constructor variants."""
    if name == "CompareConfidence":
        method = cls(model_class=_make_task_classifier(),
                     model_expert=_make_binary_head(),
                     device=DEVICE, plotting_interval=300)
        method.fit(train_loader, val_loader, test_loader, **FIT_KWARGS_PLAIN)
    elif name == "DifferentiableTriage":
        method = cls(model_class=_make_task_classifier(),
                     model_rejector=_make_binary_head(),
                     device=DEVICE, plotting_interval=300)
        method.fit(train_loader, val_loader, test_loader, **FIT_KWARGS_PLAIN)
    elif name == "SelectivePrediction":
        method = cls(model_class=_make_task_classifier(),
                     device=DEVICE, plotting_interval=300)
        method.fit(train_loader, val_loader, test_loader, **FIT_KWARGS_PLAIN)
    else:
        model = _make_surrogate_model()
        constructed = False
        for kw in [
            dict(alpha=1, plotting_interval=300, model=model, device=DEVICE, fit_on_val=True),
            dict(alpha=1, plotting_interval=300, model=model, device=DEVICE),
            dict(alpha=1, model=model, device=DEVICE),
            dict(model=model, device=DEVICE),
        ]:
            try:
                method = cls(**kw); constructed = True; break
            except TypeError:
                continue
        if not constructed:
            raise RuntimeError(f"No matching constructor for {name}")
        method.fit(train_loader, val_loader, test_loader, **FIT_KWARGS)
    return method


def run_one_method(name, cls, train_loader, val_loader, test_loader,
                   y_test, h_test, y_val):
    try:
        set_seed(SEED)
        method = _build_and_train(name, cls, train_loader, val_loader, test_loader)
        test_out = method.test(test_loader)
        model_preds, ai_probs, rej_score = parse_test_output(test_out, h_test)
        val_ai_probs, val_labels = _get_val_fold_calib_data(method, val_loader, y_val)
        return model_preds, ai_probs, rej_score, val_ai_probs, val_labels
    except Exception as exc:
        print(f"\n      [ERROR] {name}: {exc}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def get_inner_cv_calib_data(name, cls, df, outer_test_fold):
    """Pool inner-CV AI probabilities (used for inner-CV beta calibration)."""
    inner_folds = [f for f in range(N_FOLDS) if f != outer_test_fold]
    all_probs, all_labels = [], []

    for inner_val_fold in inner_folds:
        inner_train = [f for f in inner_folds if f != inner_val_fold]
        i_i_val = inner_train[-1]
        i_i_train = inner_train[:-1]

        tr_df_i = df[df[FOLD_COL].isin(i_i_train)]
        va_df_i = df[df[FOLD_COL] == i_i_val]
        te_df_i = df[df[FOLD_COL] == inner_val_fold]

        if name == "SelectivePrediction":
            ai_probs_i = te_df_i[AI_PROB_COL].values.astype(float)
        elif name == "CompareConfidence":
            tr_ld, va_ld, _, _, _, _, X_te_i = build_loaders(tr_df_i, va_df_i, te_df_i)
            _, ai_probs_i, _, _, _ = run_fixed_ai_compare_confidence(
                tr_ld, va_ld, te_df_i, va_df_i,
                te_df_i[LABEL_COL].values.astype(int), X_te_i,
            )
        else:
            tr_ld, va_ld, te_ld, y_te, h_te, y_va, _ = build_loaders(
                tr_df_i, va_df_i, te_df_i
            )
            _, ai_probs_i, _, _, _ = run_one_method(
                name, cls, tr_ld, va_ld, te_ld, y_te, h_te, y_va
            )

        if ai_probs_i is not None:
            all_probs.extend(ai_probs_i.tolist())
            all_labels.extend(te_df_i[LABEL_COL].values.tolist())

    return np.array(all_probs), np.array(all_labels)


# =============================================================================
# DEFERRAL HELPER (aggregate metrics)
# =============================================================================

def apply_fraction(rej_score, model_preds, ai_probs, h_test, fraction,
                   cal_ai_probs=None):
    """Defer the top ``fraction`` of intervals by rejection score to the human.

    Returns the system hard prediction, uncalibrated and (optional) calibrated
    system probabilities, and the binary deferral indicator.
    """
    n = len(model_preds)
    n_defer = int(np.round(fraction * n))
    rank_order = np.argsort(rej_score)[::-1]
    defers = np.zeros(n, dtype=int)
    if n_defer > 0:
        defers[rank_order[:n_defer]] = 1

    h_clip = np.clip(h_test, 0, N_CLASSES - 1)
    sys_pred = np.where(defers == 1, h_clip, model_preds)
    sys_prob_uncal = np.where(defers == 1, h_clip.astype(float), ai_probs)
    sys_prob_cal = (np.where(defers == 1, h_clip.astype(float), cal_ai_probs)
                    if cal_ai_probs is not None else None)

    return sys_pred, sys_prob_uncal, sys_prob_cal, defers


# =============================================================================
# ROW-LEVEL DEFERRAL CSV WRITER
# =============================================================================

def save_row_deferral_csv(method_name, row_records, df_orig, out_dir):
    """Write one CSV of per-row deferral decisions for ``method_name``.

    Rows are keyed on the composite primary key (Case ID, Interval). For each
    deferral fraction f, deferral is ranked and applied *within each fold's
    test set* separately (consistent with the CV experiment), adding columns:

        deferred_{f:.2f}    1 = deferred to the human, 0 = AI decides
        final_pred_{f:.2f}  prediction actually used
        final_prob_{f:.2f}  probability actually used

    Output is sorted to match the original CSV's row order.
    """
    if not row_records:
        print(f"  [SKIP] {method_name}: no row records collected.")
        return

    base_df = pd.DataFrame(row_records)

    # Every (Case ID, Interval) pair should appear exactly once.
    n_dupes = base_df.duplicated(subset=[CASE_ID_COL, INTERVAL_COL]).sum()
    if n_dupes > 0:
        print(f"  [WARN] {method_name}: {n_dupes} duplicate (Case ID, Interval) "
              f"pairs found — check fold assignment.")

    for frac in DEFERRAL_FRACTIONS:
        col_def = f"deferred_{frac:.2f}"
        col_pred = f"final_pred_{frac:.2f}"
        col_prob = f"final_prob_{frac:.2f}"

        deferred_arr = np.zeros(len(base_df), dtype=int)
        final_pred = np.empty(len(base_df), dtype=int)
        final_prob = np.empty(len(base_df), dtype=float)

        for fold_id in range(N_FOLDS):
            fold_mask = (base_df["fold"] == fold_id).values
            fold_pos = np.where(fold_mask)[0]

            if len(fold_pos) == 0:
                continue

            fold_rej = base_df["rej_score"].values[fold_pos]
            fold_ai_pred = base_df["ai_pred"].values[fold_pos]
            fold_ai_prob = base_df["ai_prob"].values[fold_pos]
            fold_human = base_df["human_label"].values[fold_pos]

            n_fold = len(fold_pos)
            n_defer = int(np.round(frac * n_fold))
            rank_order = np.argsort(fold_rej)[::-1]
            fold_deferred = np.zeros(n_fold, dtype=int)
            if n_defer > 0:
                fold_deferred[rank_order[:n_defer]] = 1

            h_clip = np.clip(fold_human, 0, N_CLASSES - 1)
            deferred_arr[fold_pos] = fold_deferred
            final_pred[fold_pos] = np.where(fold_deferred, h_clip, fold_ai_pred)
            final_prob[fold_pos] = np.where(fold_deferred,
                                            h_clip.astype(float), fold_ai_prob)

        base_df[col_def] = deferred_arr
        base_df[col_pred] = final_pred
        base_df[col_prob] = final_prob

    # Restore the original CSV row order via an explicit key merge.
    orig_order = (df_orig[[CASE_ID_COL, INTERVAL_COL]]
                  .reset_index(drop=True)
                  .reset_index()
                  .rename(columns={"index": "_orig_row_order"}))
    base_df = base_df.merge(orig_order, on=[CASE_ID_COL, INTERVAL_COL], how="left")
    base_df = base_df.sort_values("_orig_row_order").drop(columns=["_orig_row_order"])
    base_df = base_df.reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"row_deferral_{method_name}.csv")
    base_df.to_csv(fname, index=False)
    print(f"  -> Saved row-level deferral CSV ({len(base_df)} rows, "
          f"{len(DEFERRAL_FRACTIONS)} fractions): {fname}")


# =============================================================================
# PLOTTING
# =============================================================================

def _base_plot(ax, method_names, xs_dict, ys_dict, ylabel, title):
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(method_names), 1)))
    for i, name in enumerate(method_names):
        xs = xs_dict.get(name, [])
        ys = ys_dict.get(name, [])
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5,
                    label=name, color=colors[i])
    ax.set_xlabel("Deferral Fraction", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)


def plot_standard_curves(results, method_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for metric in STANDARD_METRIC_PLOTS:
        fig, ax = plt.subplots(figsize=(9, 6))
        xs_d, ys_d = {}, {}
        for name in method_names:
            if name not in results:
                continue
            xs, ys = [], []
            for frac in DEFERRAL_FRACTIONS:
                entry = results[name].get(frac)
                if entry and not np.isnan(entry.get(metric, float("nan"))):
                    xs.append(entry["DeferRate"])
                    ys.append(entry[metric])
            xs_d[name], ys_d[name] = xs, ys
        _base_plot(ax, method_names, xs_d, ys_d, metric,
                   f"{metric} vs Deferral Fraction")
        fname = os.path.join(out_dir, f"std_{metric}.png")
        fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)
        print(f"  -> {fname}")


def plot_calib_curves(calib_results, method_names, out_dir, tag="uncal"):
    os.makedirs(out_dir, exist_ok=True)
    calib_metrics = ["ECE", "MCE", "Brier", "LogLoss"]
    for metric in calib_metrics:
        fig, ax = plt.subplots(figsize=(9, 6))
        xs_d, ys_d = {}, {}
        for name in method_names:
            if name not in calib_results:
                continue
            xs, ys = [], []
            for frac in CALIB_FRACTIONS:
                entry = calib_results[name].get(frac)
                if entry and not np.isnan(entry.get(metric, float("nan"))):
                    xs.append(frac)
                    ys.append(entry[metric])
            xs_d[name], ys_d[name] = xs, ys
        label = "Uncalibrated" if tag == "uncal" else "Beta-Calibrated"
        _base_plot(ax, method_names, xs_d, ys_d, metric,
                   f"{metric} vs Deferral Fraction ({label})")
        fname = os.path.join(out_dir, f"calib_{tag}_{metric}.png")
        fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)
        print(f"  -> {fname}")


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RS/OvA/CC/SP/MoE human-AI deferral baselines and write "
                    "per-row deferral CSVs and metric sweeps."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to the assembled interval-level table (e.g. data_mlhc.csv).",
    )
    parser.add_argument(
        "--repo_path", required=True,
        help="Path to a local clone of clinicalml/human_ai_deferral "
             "(provides the baseline implementations).",
    )
    parser.add_argument(
        "--output_dir", default="./output_baselines",
        help="Directory for the output CSVs and plots (created if absent).",
    )
    parser.add_argument(
        "--annotator", default="novice_1",
        help="Column name of the novice's binary predictions in the CSV.",
    )
    parser.add_argument(
        "--ai_name", default="RF",
        help="Prefix of the fixed AI columns, e.g. 'RF' -> RF_prob, RF_pred.",
    )
    parser.add_argument("--label_col", default="Event",
                        help="Ground-truth label column.")
    parser.add_argument("--fold_col", default="fold",
                        help="Patient-level CV fold column (expects folds 0..N-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Bind run-wide column names / seed from the command line. The helper
    # functions above reference these as module-level names.
    global HUMAN_COL, AI_PROB_COL, AI_PRED_COL, LABEL_COL, FOLD_COL, SEED
    HUMAN_COL = args.annotator
    AI_PROB_COL = f"{args.ai_name}_prob"
    AI_PRED_COL = f"{args.ai_name}_pred"
    LABEL_COL = args.label_col
    FOLD_COL = args.fold_col
    SEED = args.seed

    # Resolve paths to absolute before changing directory.
    csv_path = os.path.abspath(args.csv)
    repo_path = os.path.abspath(args.repo_path)
    out_dir = os.path.abspath(args.output_dir)

    # Make the cloned upstream repo importable, then mirror its expected working
    # directory before importing the baseline classes.
    sys.path.insert(0, repo_path)
    os.chdir(repo_path)

    set_seed(SEED)
    METHODS = _import_repo()
    os.makedirs(out_dir, exist_ok=True)

    need_calib = RUN_CALIB_UNCAL or RUN_CALIB_CAL

    print(f"{'='*65}")
    print("  Human-AI Deferral Baselines — cEEG")
    print(f"  Device : {DEVICE}  |  Seed: {SEED}  |  Epochs: {EPOCHS}")
    print(f"  Methods: {list(METHODS.keys())}")
    print("  SelectivePrediction : fixed AI backbone (no training)")
    print("  CompareConfidence   : fixed AI backbone + trained expert proxy")
    print(f"  Standard metrics : {RUN_STANDARD_METRICS}  "
          f"Calib uncal: {RUN_CALIB_UNCAL}  Calib cal: {RUN_CALIB_CAL}")
    print(f"{'='*65}\n")

    # ---- Load and validate the input table ---------------------------------
    df = pd.read_csv(csv_path)
    for col in (LABEL_COL, HUMAN_COL, FOLD_COL, AI_PROB_COL, AI_PRED_COL):
        assert col in df.columns, f"Missing column '{col}'"
    fold_vals = sorted(df[FOLD_COL].unique())
    assert fold_vals == list(range(N_FOLDS)), \
        f"Expected folds 0-{N_FOLDS-1}, got {fold_vals}"
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    n_missing = df[HUMAN_COL].isna().sum()
    if n_missing > 0:
        raise ValueError(f"'{HUMAN_COL}' has {n_missing} missing values.")
    df[HUMAN_COL] = df[HUMAN_COL].astype(int).clip(0, 1)
    df[AI_PRED_COL] = df[AI_PRED_COL].astype(int).clip(0, 1)
    df[AI_PROB_COL] = df[AI_PROB_COL].astype(float).clip(0.0, 1.0)
    print(f"Dataset : {df.shape[0]} rows | Label: {dict(df[LABEL_COL].value_counts().sort_index())}")
    print(f"Human   : {dict(df[HUMAN_COL].value_counts().sort_index())}\n")

    # ---- Storage ------------------------------------------------------------
    std_pool = {name: {frac: {"y_true": [], "y_pred": [], "y_prob": [], "deferred": []}
                       for frac in DEFERRAL_FRACTIONS}
                for name in METHODS}

    cal_pool = {name: {frac: {"y_true": [], "y_prob_uncal": [], "y_prob_cal": []}
                       for frac in CALIB_FRACTIONS}
                for name in METHODS}

    # Per-row records, one list per method. Deferral columns are computed at
    # save time, after all folds.
    row_records = {name: [] for name in METHODS}

    # ---- 10-fold cross-validation -------------------------------------------
    for fold in range(N_FOLDS):
        print(f"{'-'*65}")
        print(f"  Fold {fold}  (test={fold}, val={(fold+1)%N_FOLDS}, train=rest)")
        print(f"{'-'*65}")

        train_df, val_df, test_df = get_splits(df, fold)
        tr_ld, va_ld, te_ld, y_test, h_test, y_val, X_te = build_loaders(
            train_df, val_df, test_df
        )

        # Primary keys for the test fold, read positionally (row i of test_df
        # aligns with index i of the arrays returned by build_loaders).
        test_case_ids = test_df[CASE_ID_COL].tolist()
        test_intervals = test_df[INTERVAL_COL].tolist()

        print(f"  Sizes -> train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

        for name, cls in METHODS.items():
            print(f"  > {name:<25}", end=" ", flush=True)

            # Dispatch by backbone type.
            if name == "SelectivePrediction":
                out = run_fixed_ai_selective_prediction(test_df, val_df, y_val)
                backbone_tag = "fixed-AI"
            elif name == "CompareConfidence":
                out = run_fixed_ai_compare_confidence(
                    tr_ld, va_ld, test_df, val_df, y_val, X_te
                )
                backbone_tag = "fixed-AI+expert"
            else:
                out = run_one_method(name, cls, tr_ld, va_ld, te_ld,
                                     y_test, h_test, y_val)
                backbone_tag = "MLP"

            model_preds, ai_probs, rej_score, val_ai_probs, val_labels = out

            if model_preds is None:
                print("FAILED"); continue

            # Record per-row base outputs for this fold.
            for i in range(len(model_preds)):
                row_records[name].append({
                    CASE_ID_COL:  test_case_ids[i],
                    INTERVAL_COL: test_intervals[i],
                    "fold":       fold,
                    "y_true":     int(y_test[i]),
                    "human_label": int(h_test[i]),
                    "ai_pred":    int(model_preds[i]),
                    "ai_prob":    float(ai_probs[i]),
                    "rej_score":  float(rej_score[i]),
                })

            # Beta calibrator (only when calibrated metrics are requested).
            calibrator = None
            if need_calib and RUN_CALIB_CAL:
                if CALIB_USE_INNER_CV:
                    cv_probs, cv_labels = get_inner_cv_calib_data(
                        name, cls, df, outer_test_fold=fold
                    )
                    cal_probs_src, cal_labels_src = cv_probs, cv_labels
                else:
                    cal_probs_src, cal_labels_src = val_ai_probs, val_labels
                calibrator = BetaCalibrator().fit(cal_probs_src, cal_labels_src)

            cal_ai_probs = calibrator.calibrate(ai_probs) if calibrator else None

            # Standard metrics across deferral fractions.
            if RUN_STANDARD_METRICS:
                for frac in DEFERRAL_FRACTIONS:
                    sys_pred, sys_prob_u, _, defers = apply_fraction(
                        rej_score, model_preds, ai_probs, h_test, frac
                    )
                    std_pool[name][frac]["y_true"].extend(y_test.tolist())
                    std_pool[name][frac]["y_pred"].extend(sys_pred.tolist())
                    std_pool[name][frac]["y_prob"].extend(sys_prob_u.tolist())
                    std_pool[name][frac]["deferred"].extend(defers.tolist())

            # Calibration metrics across deferral fractions.
            if need_calib:
                for frac in CALIB_FRACTIONS:
                    _, sys_prob_u, sys_prob_c, defers = apply_fraction(
                        rej_score, model_preds, ai_probs, h_test, frac,
                        cal_ai_probs=cal_ai_probs
                    )
                    cal_pool[name][frac]["y_true"].extend(y_test.tolist())
                    if RUN_CALIB_UNCAL:
                        cal_pool[name][frac]["y_prob_uncal"].extend(sys_prob_u.tolist())
                    if RUN_CALIB_CAL and sys_prob_c is not None:
                        cal_pool[name][frac]["y_prob_cal"].extend(sys_prob_c.tolist())

            # Quick preview at 10% deferral.
            _pf = 0.10 if 0.10 in DEFERRAL_FRACTIONS else DEFERRAL_FRACTIONS[len(DEFERRAL_FRACTIONS) // 2]
            _, sp_u, _, _ = apply_fraction(rej_score, model_preds, ai_probs, h_test, _pf)
            sp_u = np.clip(sp_u, 0, 1)
            try:
                auprc_preview = average_precision_score(y_test, sp_u)
            except Exception:
                auprc_preview = float("nan")
            print(f"[{backbone_tag}] @{int(_pf*100)}%-> auprc={auprc_preview:.3f}")

        print()

    # ---- Save per-row deferral CSVs -----------------------------------------
    print(f"\n{'='*65}")
    print("  Saving per-row deferral CSVs ...")
    print(f"{'='*65}")
    for name in METHODS:
        save_row_deferral_csv(name, row_records[name], df, out_dir)

    # ---- Compute and save aggregate results ---------------------------------
    std_results = {}
    std_rows = []

    if RUN_STANDARD_METRICS:
        print("\nComputing standard metric curves ...")
        for name in METHODS:
            std_results[name] = {}
            for frac in DEFERRAL_FRACTIONS:
                d = std_pool[name][frac]
                if not d["y_true"]:
                    continue
                yt = np.array(d["y_true"])
                yp = np.clip(np.array(d["y_pred"]), 0, N_CLASSES - 1)
                ypr = np.clip(np.array(d["y_prob"]), 0, 1)
                yd = np.array(d["deferred"])
                m = compute_metrics_dict(yt, yp, ypr)
                m["DeferRate"] = float(yd.mean())
                std_results[name][frac] = m
                std_rows.append({"Method": name, "TargetFraction": frac, **m})

        pd.DataFrame(std_rows).to_csv(
            os.path.join(out_dir, "results_standard_sweep.csv"), index=False)
        print("-> Saved: results_standard_sweep.csv")

    calib_results_uncal = {}
    calib_results_cal = {}
    calib_rows = []

    if need_calib:
        print("Computing calibration metric curves ...")
        for name in METHODS:
            calib_results_uncal[name] = {}
            calib_results_cal[name] = {}
            for frac in CALIB_FRACTIONS:
                d = cal_pool[name][frac]
                yt = np.array(d["y_true"])
                if not len(yt):
                    continue
                row = {"Method": name, "TargetFraction": frac}
                if RUN_CALIB_UNCAL and d["y_prob_uncal"]:
                    m_u = compute_calib_metrics(yt, d["y_prob_uncal"])
                    calib_results_uncal[name][frac] = m_u
                    row.update({f"Uncal_{k}": v for k, v in m_u.items()})
                if RUN_CALIB_CAL and d["y_prob_cal"]:
                    m_c = compute_calib_metrics(yt, d["y_prob_cal"])
                    calib_results_cal[name][frac] = m_c
                    row.update({f"Cal_{k}": v for k, v in m_c.items()})
                calib_rows.append(row)

        pd.DataFrame(calib_rows).to_csv(
            os.path.join(out_dir, "results_calibration_sweep.csv"), index=False)
        print("-> Saved: results_calibration_sweep.csv")

    for frac in [0.0, 0.05, 0.10, 0.25, 0.50]:
        if not RUN_STANDARD_METRICS or frac not in DEFERRAL_FRACTIONS:
            continue
        print(f"\n{'='*65}")
        print(f"  STANDARD METRICS @ {int(frac*100)}% deferral (pooled, sorted by AUPRC)")
        print(f"{'='*65}")
        col_keys = ["AUPRC", "AUROC", "Accuracy", "F1", "Precision", "Recall", "Specificity", "DeferRate"]
        col_hdrs = ["AUPRC", "AUROC", "Acc", "F1", "Prec", "Rec", "Spec", "Defer"]
        hdr = f"{'Method':<26}" + "".join(f"{h:>10}" for h in col_hdrs)
        print(hdr); print("-" * len(hdr))
        rows_frac = [(n, std_results[n][frac]) for n in METHODS
                     if n in std_results and frac in std_results[n]]
        rows_frac.sort(key=lambda x: x[1]["AUPRC"], reverse=True)
        for n, m in rows_frac:
            print(f"{n:<26}" + "".join(f"{m[k]:>10.3f}" for k in col_keys))

    plots_dir = os.path.join(out_dir, "plots")
    print(f"\nGenerating plots -> {plots_dir}")
    if RUN_STANDARD_METRICS and std_results:
        plot_standard_curves(std_results, list(METHODS.keys()), plots_dir)
    if RUN_CALIB_UNCAL and calib_results_uncal:
        plot_calib_curves(calib_results_uncal, list(METHODS.keys()), plots_dir, tag="uncal")
    if RUN_CALIB_CAL and calib_results_cal:
        plot_calib_curves(calib_results_cal, list(METHODS.keys()), plots_dir, tag="cal")

    print("\nDone.")


if __name__ == "__main__":
    main()
