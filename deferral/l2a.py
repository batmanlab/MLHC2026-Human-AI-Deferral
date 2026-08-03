"""
Learning to Augment (L2A): a soft human-AI deferral system.

L2A learns, per interval, a weight ``w`` in [0, 1] estimating the probability
that a novice's prediction improves on the AI. Given a deferral threshold, the
final prediction blends the two sources only where the human is expected to
help::

    y_hat = w * h + (1 - w) * p     if the interval is deferred
    y_hat = p                        otherwise

where ``p`` is the AI probability of ischemia and ``h`` is the novice's binary
prediction. See the paper (Methods, Section 3) for the formulation.

This script consumes an interval-level table with precomputed AI predictions
and evaluates L2A with nested, patient-level cross-validation:

  * Outer folds come from a ``fold`` column (all intervals of a case share a
    fold). Each case must belong to exactly one fold.
  * An inner cross-validation over the outer-training cases learns the F1- and
    Youden-optimal classification thresholds for every deferral fraction.
  * The gating threshold that selects which intervals are deferred is derived
    from the final model's weights on the outer-training set, then applied to
    the held-out fold (so no test information leaks into thresholding).

Note that the AI backbone (a Random Forest trained on the 111 cEEG features)
is fit elsewhere; this script uses its precomputed out-of-fold probabilities.
The L2A weight model here is a separate, lightweight estimator (Random Forest
by default).

Expected input columns (CSV)
----------------------------
    Case ID, Interval          composite key (unique per row)
    Event                      ground-truth ischemia label (0/1)
    fold                       patient-level CV fold id
    <ai_name>_prob             AI probability of ischemia
    <ai_name>_pred             AI hard prediction (0/1)
    <novice column>            novice binary prediction (0/1)
    111 cEEG feature columns   used for the EEG-variability feature

Outputs (under --output_dir)
----------------------------
    deferral_<annotator>.csv        out-of-fold predictions for every fold
    deferral_<annotator>_full.csv   the input table with results merged back

For each deferral fraction ``d`` (encoded as a suffix, e.g. 0.10 -> "10"):
    gated_prob_d<dd>    gated probability (hybrid where deferred, else AI prob)
    f1_pred_d<dd>       binary prediction at the inner-CV F1-optimal threshold
    youden_pred_d<dd>   binary prediction at the inner-CV Youden threshold
    deferred_d<dd>      1 if the interval was deferred to the novice, else 0

Usage
-----
    python l2a.py --csv data_mlhc.csv --output_dir ./output --annotator novice_1

    # data-efficiency / onboarding experiment (subsample training cases):
    python l2a.py --csv data_mlhc.csv --output_dir ./output \\
        --annotator novice_1 --data_frac 0.2
"""

import argparse
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Default grid of deferral fractions (dense at low fractions for resolution
# around the 5-10% operating range reported in the paper).
DEFAULT_DEFERRAL_FRACTIONS = [
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
    0.21, 0.22, 0.23, 0.24, 0.25,
    0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
    0.8, 0.85, 0.9, 0.95, 1.0,
]


# =============================================================================
# THRESHOLD UTILITIES
# =============================================================================

def find_youden_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """Find the threshold maximizing the Youden index (sensitivity + specificity - 1).

    Non-finite thresholds returned by ``roc_curve`` (e.g. the leading +inf) are
    excluded.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # roc_curve's leading threshold can be +inf; keep only finite thresholds.
    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        return {'threshold': 0.5, 'youden_index': 0.0,
                'precision': 0.0, 'recall': 0.0, 'specificity': 0.0,
                'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    fpr        = fpr[finite_mask]
    tpr        = tpr[finite_mask]
    thresholds = thresholds[finite_mask]

    youden_scores = tpr - fpr
    optimal_idx = np.argmax(youden_scores)
    optimal_threshold = thresholds[optimal_idx]

    preds = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        'threshold':    optimal_threshold,
        'youden_index': youden_scores[optimal_idx],
        'precision':    tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall':       tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity':  tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def find_f1_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Dict:
    """Find the threshold maximizing F1.

    Searches over the unique score values (the only thresholds that can change
    the confusion matrix), plus a sentinel just below the minimum so that a
    threshold below all scores yields an all-positive prediction.
    """
    unique_scores = np.unique(y_score)
    thresholds = np.concatenate([[unique_scores[0] - 1e-9], unique_scores])

    best_f1, best_threshold, best_stats = 0.0, 0.5, {}

    for thr in thresholds:
        preds = (y_score >= thr).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            best_stats = {
                'threshold': thr,
                'f1_score':  f1,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                'recall':    tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            }

    return best_stats if best_stats else {'threshold': 0.5, 'f1_score': 0.0}


# =============================================================================
# SPLIT UTILITIES
# =============================================================================

def verify_case_splits(case_splits: List[List], df: pd.DataFrame,
                       groundtruth_col: str, split_name: str = "CASE SPLIT",
                       case_id_col: str = 'Case ID'):
    """Sanity-check case splits: no empty folds, no overlap, complete coverage."""
    case_labels = df.groupby(case_id_col)[groundtruth_col].any().astype(int)
    all_cases = case_labels.index.values

    print(f"\n{'='*60}")
    print(f"{split_name} VERIFICATION (n_folds={len(case_splits)})")
    print(f"{'='*60}")
    print(f"Total cases: {len(all_cases)} | "
          f"Pos: {np.sum(case_labels)} ({np.mean(case_labels)*100:.1f}%) | "
          f"Neg: {len(all_cases)-np.sum(case_labels)} "
          f"({(1-np.mean(case_labels))*100:.1f}%)")

    for i, fold_cases in enumerate(case_splits):
        if len(fold_cases) == 0:
            raise ValueError(f"Fold {i} is empty.")
        fl = case_labels.loc[fold_cases].values
        print(f"  Fold {i}: {len(fold_cases)} cases | "
              f"Pos: {np.sum(fl)} ({np.sum(fl)/len(fl)*100:.1f}%) | "
              f"Neg: {len(fl)-np.sum(fl)} "
              f"({(1-np.sum(fl)/len(fl))*100:.1f}%)")

    all_in_splits = [c for fold in case_splits for c in fold]
    if len(all_in_splits) != len(set(all_in_splits)):
        raise ValueError("Duplicate cases detected across folds — data leakage!")
    if set(all_in_splits) != set(all_cases):
        raise ValueError("Case coverage mismatch between dataset and splits!")

    print("[ok] No overlap | [ok] Complete coverage")
    print(f"{'='*60}\n")


def create_stratified_case_splits(df: pd.DataFrame, n_folds: int,
                                  groundtruth_col: str,
                                  case_id_col: str = 'Case ID') -> List[List]:
    """Build stratified K-fold splits at the case (patient) level."""
    case_labels = df.groupby(case_id_col)[groundtruth_col].any().astype(int)
    cases = case_labels.index.values
    labels = case_labels.values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    case_splits = [cases[test_idx].tolist()
                   for _, test_idx in skf.split(cases, labels)]

    verify_case_splits(case_splits, df, groundtruth_col,
                       split_name="STRATIFIED CASE SPLIT",
                       case_id_col=case_id_col)
    return case_splits


def stratified_case_sampling(df: pd.DataFrame, case_id_col: str,
                             label_col: str, frac: float,
                             random_state: Optional[int] = None) -> List:
    """Stratified fractional sampling of cases (preserving positive/negative mix).

    Used for the data-efficiency / onboarding experiments. Uses a local RNG so
    the global NumPy random state is not mutated.
    """
    if frac == 1.0:
        return df[case_id_col].unique().tolist()

    case_has_event = df.groupby(case_id_col)[label_col].any()
    pos_cases = case_has_event[case_has_event].index.values
    neg_cases = case_has_event[~case_has_event].index.values

    rng = np.random.default_rng(random_state)
    sampled_pos = rng.choice(pos_cases, max(1, int(len(pos_cases) * frac)), replace=False)
    sampled_neg = rng.choice(neg_cases, max(1, int(len(neg_cases) * frac)), replace=False)
    return np.concatenate([sampled_pos, sampled_neg]).tolist()


# =============================================================================
# DEFERRAL WEIGHT MODEL
# =============================================================================

class L2AWeightModel:
    """Estimator for the L2A deferral weight.

    Learns, per interval, whether blending the novice prediction with the AI
    improves on the AI alone. The training target is::

        human_better = |human_pred - y| < |ai_pred - y|

    and the model's predicted probability P(human_better) is used directly as
    the blend weight ``w``, clipped to [0.05, 0.95] (no additional scaling or
    calibration).

    The canonical L2A feature set is {AI probability, AI uncertainty, EEG
    variability}, matching the paper. The remaining constructor flags enable
    experimental features and are disabled in the reported results.
    """

    def __init__(self,
                 ml_model: str = 'rf',
                 use_ai_uncertainty: bool = True,
                 use_ai_probability: bool = True,
                 use_eeg_complexity: bool = True,
                 use_novice_prediction: bool = True,
                 use_ai_overconfidence: bool = False,
                 use_eeg_ai_mismatch: bool = False,
                 use_raw_eeg_features: bool = False,
                 use_ai_uncertainty_zone: bool = False,
                 use_sweet_spot_bonus: bool = False):

        self.is_fitted = False
        self.ml_model_type = ml_model
        self.ml_model = None
        self.scaler = None
        self.eeg_stats = {}
        self.fallback_mode = False
        self.fallback_weight = 0.5
        self.overconfidence_penalty = 0.0
        self.sweet_spot_bins = None
        self.sweet_spot_bonus = None

        self.use_ai_uncertainty = use_ai_uncertainty
        self.use_ai_probability = use_ai_probability
        self.use_eeg_complexity = use_eeg_complexity
        self.use_novice_prediction = use_novice_prediction
        self.use_ai_overconfidence = use_ai_overconfidence
        self.use_eeg_ai_mismatch = use_eeg_ai_mismatch
        self.use_raw_eeg_features = use_raw_eeg_features
        self.use_ai_uncertainty_zone = use_ai_uncertainty_zone
        self.use_sweet_spot_bonus = use_sweet_spot_bonus

        self.active_features = []
        if use_ai_uncertainty:      self.active_features.append('AI_uncertainty')
        if use_ai_probability:      self.active_features.append('AI_probability')
        if use_eeg_complexity:      self.active_features.append('EEG_complexity')
        if use_novice_prediction:   self.active_features.append('Novice_prediction')
        if use_ai_overconfidence:   self.active_features.append('AI_overconfidence')
        if use_eeg_ai_mismatch:     self.active_features.append('EEG_AI_mismatch')
        if use_raw_eeg_features:    self.active_features.append('Raw_EEG_features')
        if use_ai_uncertainty_zone: self.active_features.append('AI_uncertainty_zone')
        if use_sweet_spot_bonus:    self.active_features.append('Sweet_spot_bonus')

        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    def _learn_patterns(self, X, ai_probs, ai_preds, human_preds, true_labels):
        """Precompute statistics used by the optional experimental features."""
        ai_errors = np.abs(ai_preds - true_labels)
        human_errors = np.abs(human_preds - true_labels)
        human_better = human_errors < ai_errors

        if self.use_ai_overconfidence:
            ai_overconfident = (ai_probs > 0.8) | (ai_probs < 0.2)
            overconfident_wrong = ai_overconfident & (ai_errors > 0.3)
            if np.sum(overconfident_wrong) > 10:
                self.overconfidence_penalty = (
                    np.mean(human_better[overconfident_wrong]) * 0.3
                )

        if self.use_sweet_spot_bonus:
            ai_entropy = -(ai_probs * np.log(ai_probs + 1e-10) +
                           (1 - ai_probs) * np.log(1 - ai_probs + 1e-10))
            ai_uncertainty = ai_entropy / 0.693
            self.sweet_spot_bins = np.linspace(0, 1, 11)
            self.sweet_spot_bonus = np.zeros(len(self.sweet_spot_bins) - 1)
            bin_indices = np.digitize(ai_uncertainty, self.sweet_spot_bins)
            for b in range(1, len(self.sweet_spot_bins)):
                mask = (bin_indices == b)
                if np.sum(mask) >= 10 and np.mean(human_better[mask]) > 0.15:
                    self.sweet_spot_bonus[b - 1] = (
                        np.mean(human_better[mask]) * 0.5
                    )

        if ((self.use_eeg_complexity or self.use_eeg_ai_mismatch)
                and X is not None and X.shape[1] > 0):
            signal_std = np.std(X, axis=1)
            signal_range = np.ptp(X, axis=1)
            self.eeg_stats = {
                'std_min':     np.min(signal_std),
                'std_range':   np.ptp(signal_std),
                'range_min':   np.min(signal_range),
                'range_range': np.ptp(signal_range),
            }

    # ------------------------------------------------------------------
    def _compute_features(self, X, ai_probs, human_preds):
        """Assemble the model input matrix from the enabled features."""
        n = len(ai_probs)
        feature_list: List[np.ndarray] = []
        self.feature_names = []

        if self.use_ai_uncertainty:
            # Normalized Shannon entropy of the AI probability (0.693 = ln 2).
            ent = -(ai_probs * np.log(ai_probs + 1e-10) +
                    (1 - ai_probs) * np.log(1 - ai_probs + 1e-10))
            feature_list.append((ent / 0.693).reshape(-1, 1))
            self.feature_names.append('AI_uncertainty')

        if self.use_ai_probability:
            feature_list.append(ai_probs.reshape(-1, 1))
            self.feature_names.append('AI_probability')

        if self.use_eeg_complexity:
            # EEG-variability feature: mean of min-max-normalized std and range
            # across the interval's features (normalization stats from training).
            if X is not None and X.shape[1] > 0 and self.eeg_stats:
                std_n = np.clip(
                    (np.std(X, axis=1) - self.eeg_stats['std_min']) /
                    (self.eeg_stats['std_range'] + 1e-8), 0, 1)
                rng_n = np.clip(
                    (np.ptp(X, axis=1) - self.eeg_stats['range_min']) /
                    (self.eeg_stats['range_range'] + 1e-8), 0, 1)
                complexity = 0.5 * std_n + 0.5 * rng_n
            else:
                complexity = np.full(n, 0.5)
            feature_list.append(complexity.reshape(-1, 1))
            self.feature_names.append('EEG_complexity')

        if self.use_novice_prediction:
            feature_list.append(human_preds.reshape(-1, 1))
            self.feature_names.append('Novice_prediction')

        if self.use_ai_overconfidence:
            conf = np.abs(ai_probs - 0.5) * 2
            boost = np.where(conf > 0.8, self.overconfidence_penalty, 0)
            feature_list.append(boost.reshape(-1, 1))
            self.feature_names.append('AI_overconfidence')

        if self.use_eeg_ai_mismatch:
            if X is not None and X.shape[1] > 0 and self.eeg_stats:
                std_n = np.clip(
                    (np.std(X, axis=1) - self.eeg_stats['std_min']) /
                    (self.eeg_stats['std_range'] + 1e-8), 0, 1)
                rng_n = np.clip(
                    (np.ptp(X, axis=1) - self.eeg_stats['range_min']) /
                    (self.eeg_stats['range_range'] + 1e-8), 0, 1)
                complexity = 0.5 * std_n + 0.5 * rng_n
            else:
                complexity = np.full(n, 0.5)
            conf = np.abs(ai_probs - 0.5) * 2
            mismatch = np.abs(complexity - (1 - conf)) * 0.15
            feature_list.append(mismatch.reshape(-1, 1))
            self.feature_names.append('EEG_AI_mismatch')

        if self.use_raw_eeg_features:
            if X is not None and X.shape[1] > 0:
                feature_list.append(X)
                self.feature_names += [f'EEG_feature_{i}' for i in range(X.shape[1])]
            else:
                raise ValueError("Raw EEG features requested but X is empty!")

        if self.use_ai_uncertainty_zone:
            ent = -(ai_probs * np.log(ai_probs + 1e-10) +
                    (1 - ai_probs) * np.log(1 - ai_probs + 1e-10))
            unc = ent / 0.693
            zone = 1 - 4 * (unc - 0.5) ** 2
            feature_list.append(zone.reshape(-1, 1))
            self.feature_names.append('AI_uncertainty_zone')

        if self.use_sweet_spot_bonus:
            if self.sweet_spot_bins is not None:
                ent = -(ai_probs * np.log(ai_probs + 1e-10) +
                        (1 - ai_probs) * np.log(1 - ai_probs + 1e-10))
                unc = ent / 0.693
                bidx = np.clip(
                    np.digitize(unc, self.sweet_spot_bins),
                    1, len(self.sweet_spot_bonus))
                bonus = self.sweet_spot_bonus[bidx - 1]
            else:
                bonus = np.zeros(n)
            feature_list.append(bonus.reshape(-1, 1))
            self.feature_names.append('Sweet_spot_bonus')

        if not feature_list:
            raise ValueError("No features selected!")
        return np.column_stack(feature_list)

    # ------------------------------------------------------------------
    def _get_base_weights(self, X, ai_probs, human_preds) -> np.ndarray:
        if self.fallback_mode:
            return np.full(len(ai_probs), self.fallback_weight)
        feats = self._compute_features(X, ai_probs, human_preds)
        if self.scaler is not None:
            feats = self.scaler.transform(feats)
        return self.ml_model.predict_proba(feats)[:, 1]

    # ------------------------------------------------------------------
    def fit(self, X, ai_probs, ai_preds, human_preds, true_labels,
            verbose: bool = True):
        """Fit the weight model to predict P(human_better)."""
        self._learn_patterns(X, ai_probs, ai_preds, human_preds, true_labels)

        ai_errors = np.abs(ai_preds - true_labels)
        human_errors = np.abs(human_preds - true_labels)
        human_better = human_errors < ai_errors

        if verbose:
            print(f"  Model: {self.ml_model_type.upper()}")
            print(f"  Active features: {', '.join(self.active_features)}")
            print(f"  Human better: {np.sum(human_better)}/{len(human_better)} "
                  f"({np.mean(human_better)*100:.1f}%)")

        # If one class is absent or too rare to train on, fall back to a
        # constant weight rather than fitting a degenerate classifier.
        unique_cls = np.unique(human_better)
        if len(unique_cls) < 2 or np.sum(human_better) < 5 or np.sum(~human_better) < 5:
            if verbose:
                print("  Warning: Insufficient class balance -> fallback mode.")
            self.fallback_mode = True
            self.fallback_weight = (
                np.mean(human_better) if len(unique_cls) > 1
                else (0.75 if unique_cls[0] else 0.25)
            )
            self.is_fitted = True
            return self

        feats = self._compute_features(X, ai_probs, human_preds)
        if verbose:
            print(f"  Feature matrix shape: {feats.shape}")

        if self.ml_model_type == 'logistic':
            self.scaler = StandardScaler()
            feats = self.scaler.fit_transform(feats)
            self.ml_model = LogisticRegression(C=1.0, random_state=RANDOM_SEED,
                                               max_iter=1000)
        elif self.ml_model_type == 'ridge':
            self.scaler = StandardScaler()
            feats = self.scaler.fit_transform(feats)
            self.ml_model = LogisticRegression(penalty='l2', C=0.1,
                                               random_state=RANDOM_SEED, max_iter=1000)
        elif self.ml_model_type == 'lasso':
            self.scaler = StandardScaler()
            feats = self.scaler.fit_transform(feats)
            self.ml_model = LogisticRegression(penalty='l1', C=1.0,
                                               random_state=RANDOM_SEED,
                                               max_iter=1000, solver='saga')
        elif self.ml_model_type == 'elasticnet':
            self.scaler = StandardScaler()
            feats = self.scaler.fit_transform(feats)
            self.ml_model = LogisticRegression(penalty='elasticnet', C=1.0,
                                               l1_ratio=0.5, random_state=RANDOM_SEED,
                                               max_iter=1000, solver='saga')
        elif self.ml_model_type == 'rf':
            self.ml_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=20,
                random_state=RANDOM_SEED, n_jobs=-1)
        elif self.ml_model_type == 'xgb':
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise ImportError(
                    "ml_model='xgb' requires the optional 'xgboost' package."
                ) from exc
            self.ml_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_SEED, n_jobs=-1, verbosity=0)
        elif self.ml_model_type == 'lgb':
            try:
                import lightgbm as lgb
            except ImportError as exc:
                raise ImportError(
                    "ml_model='lgb' requires the optional 'lightgbm' package."
                ) from exc
            self.ml_model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1)
        else:
            raise ValueError(f"Unknown ml_model: {self.ml_model_type}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ml_model.fit(feats, human_better)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_weights(self, X, ai_probs, human_preds) -> np.ndarray:
        """Return deferral weights, clipped to [0.05, 0.95].

        The model's raw predicted probability is used directly as the blending
        weight; clipping avoids degenerate all-AI or all-human blends.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        base_w = self._get_base_weights(X, ai_probs, human_preds)
        return np.clip(base_w, 0.05, 0.95)


# =============================================================================
# GATING UTILITIES
# =============================================================================

def gate_predictions(w: np.ndarray, ai_probs: np.ndarray,
                     human_preds: np.ndarray,
                     deferred_mask: np.ndarray) -> np.ndarray:
    """Blend AI and human predictions on deferred intervals, else use the AI.

    Takes a boolean ``deferred_mask`` so the caller can implement exact top-k
    deferral. All outputs remain probabilities.
    """
    hybrid = w * human_preds + (1 - w) * ai_probs
    return np.where(deferred_mask, hybrid, ai_probs)


def compute_deferred_mask(w: np.ndarray, d: float) -> np.ndarray:
    """Exact top-k deferral: defer the ceil(d * n) highest-weight intervals.

    Using an exact count avoids the tie distortion that percentile
    thresholding can introduce at low deferral rates (1-5%).
    """
    n = len(w)
    k = max(1, int(np.ceil(d * n)))
    top_k_idx = np.argsort(w)[-k:]  # ascending argsort; last k are the largest
    mask = np.zeros(n, dtype=bool)
    mask[top_k_idx] = True
    return mask


# =============================================================================
# INNER CV: learn F1/Youden classification thresholds per deferral fraction
# =============================================================================

def find_classification_thresholds_inner_cv(
    df_train: pd.DataFrame,
    eeg_cols: List[str],
    human_col: str,
    ai_prob_col: str,
    ai_pred_col: str,
    groundtruth_col: str,
    n_inner_folds: int,
    ml_model_type: str,
    deferral_fractions: List[float],
    use_ai_uncertainty: bool,
    use_ai_probability: bool,
    use_eeg_complexity: bool,
    use_novice_prediction: bool,
    use_ai_overconfidence: bool,
    use_eeg_ai_mismatch: bool,
    use_raw_eeg_features: bool,
    use_ai_uncertainty_zone: bool,
    use_sweet_spot_bonus: bool,
    use_fold_col: bool = False,
    fold_col: str = 'fold'
) -> Dict[float, Dict[str, float]]:
    """Learn F1- and Youden-optimal classification thresholds via inner CV.

    Only the classification thresholds are learned here; the gating threshold
    that decides which intervals are deferred is derived later from the final
    outer-trained model's weights on the outer-training set.

    Returns
    -------
    dict
        ``{deferral_fraction -> {'f1_thresh': float, 'youden_thresh': float}}``
    """
    pooled_w = []
    pooled_ai_prob = []
    pooled_human = []
    pooled_true = []

    # Build inner splits from an existing fold column when available, else
    # create fresh stratified case-level splits.
    if use_fold_col and fold_col in df_train.columns:
        unique_folds = sorted(df_train[fold_col].unique())
        inner_splits = []
        for fv in unique_folds:
            inner_splits.append(
                df_train[df_train[fold_col] == fv]['Case ID'].unique().tolist()
            )
        verify_case_splits(inner_splits, df_train, groundtruth_col,
                           split_name=f"INNER CV ('{fold_col}' col)",
                           case_id_col='Case ID')
    else:
        inner_splits = create_stratified_case_splits(
            df_train, n_inner_folds, groundtruth_col, case_id_col='Case ID'
        )

    for fold_idx in range(len(inner_splits)):
        val_cases = inner_splits[fold_idx]
        train_cases = [c for i, fold in enumerate(inner_splits)
                       if i != fold_idx for c in fold]

        df_itr = df_train[df_train['Case ID'].isin(train_cases)].copy()
        df_ival = df_train[df_train['Case ID'].isin(val_cases)].copy()

        # Require both classes present in the inner train and validation splits.
        if len(df_itr[groundtruth_col].unique()) < 2:
            print(f"    Inner fold {fold_idx}: train missing a class, skipping.")
            continue
        if len(df_ival[groundtruth_col].unique()) < 2:
            print(f"    Inner fold {fold_idx}: val missing a class, skipping.")
            continue

        # Impute missing feature values with inner-training means.
        if eeg_cols and df_itr[eeg_cols].isna().any().any():
            means = df_itr[eeg_cols].mean()
            df_itr[eeg_cols] = df_itr[eeg_cols].fillna(means)
            df_ival[eeg_cols] = df_ival[eeg_cols].fillna(means)

        X_itr = df_itr[eeg_cols].values if eeg_cols else None
        X_ival = df_ival[eeg_cols].values if eeg_cols else None

        ai_p_tr = df_itr[ai_prob_col].values
        ai_d_tr = df_itr[ai_pred_col].values.astype(int)
        hu_tr = df_itr[human_col].values.astype(int)
        y_tr = df_itr[groundtruth_col].values

        ai_p_val = df_ival[ai_prob_col].values
        hu_val = df_ival[human_col].values.astype(int)
        y_val = df_ival[groundtruth_col].values

        try:
            mdl = L2AWeightModel(
                ml_model=ml_model_type,
                use_ai_uncertainty=use_ai_uncertainty,
                use_ai_probability=use_ai_probability,
                use_eeg_complexity=use_eeg_complexity,
                use_novice_prediction=use_novice_prediction,
                use_ai_overconfidence=use_ai_overconfidence,
                use_eeg_ai_mismatch=use_eeg_ai_mismatch,
                use_raw_eeg_features=use_raw_eeg_features,
                use_ai_uncertainty_zone=use_ai_uncertainty_zone,
                use_sweet_spot_bonus=use_sweet_spot_bonus
            )
            mdl.fit(X_itr, ai_p_tr, ai_d_tr, hu_tr, y_tr, verbose=False)
            w_val = mdl.predict_weights(X_ival, ai_p_val, hu_val)

            pooled_w.extend(w_val)
            pooled_ai_prob.extend(ai_p_val)
            pooled_human.extend(hu_val)
            pooled_true.extend(y_val)

        except Exception as e:
            print(f"    Inner fold {fold_idx} failed: {e}")
            continue

    if len(pooled_true) == 0:
        print("  WARNING: All inner folds failed. Using fallback thresholds.")
        return {d: {'f1_thresh': 0.5, 'youden_thresh': 0.5}
                for d in deferral_fractions}

    pooled_w = np.array(pooled_w)
    pooled_ai_prob = np.array(pooled_ai_prob)
    pooled_human = np.array(pooled_human, dtype=float)
    pooled_true = np.array(pooled_true)

    # Both classes must be present in the pooled predictions.
    if len(np.unique(pooled_true)) < 2:
        print("  WARNING: Pooled inner-CV data has only one class. "
              "Using fallback thresholds.")
        return {d: {'f1_thresh': 0.5, 'youden_thresh': 0.5}
                for d in deferral_fractions}

    results = {}
    print(f"\n  Inner CV pooled {len(pooled_true)} samples. "
          f"Computing classification thresholds for "
          f"{len(deferral_fractions)} deferral fractions...")

    for d in deferral_fractions:
        # Exact top-k deferral on the pooled inner-fold weights.
        deferred_mask = compute_deferred_mask(pooled_w, d)
        actual_inner = float(np.mean(deferred_mask))

        gated = gate_predictions(pooled_w, pooled_ai_prob, pooled_human,
                                 deferred_mask)

        f1_stats = find_f1_optimal_threshold(pooled_true, gated)
        youden_stats = find_youden_optimal_threshold(pooled_true, gated)

        results[d] = {
            'f1_thresh':     f1_stats.get('threshold', 0.5),
            'youden_thresh': youden_stats.get('threshold', 0.5),
            'actual_deferred_inner': actual_inner,
        }

        print(f"    d={d:.2f} | inner deferred={actual_inner*100:.1f}% | "
              f"f1_thresh={results[d]['f1_thresh']:.4f} | "
              f"youden_thresh={results[d]['youden_thresh']:.4f}")

    return results


# =============================================================================
# ONE OUTER FOLD
# =============================================================================

def deferral_cv_fold(
    fold: int,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    eeg_cols: List[str],
    human_col: str,
    ai_prob_col: str,
    ai_pred_col: str,
    groundtruth_col: str,
    ml_model_type: str,
    deferral_fractions: List[float],
    n_inner_folds: int,
    use_ai_uncertainty: bool,
    use_ai_probability: bool,
    use_eeg_complexity: bool,
    use_novice_prediction: bool,
    use_ai_overconfidence: bool,
    use_eeg_ai_mismatch: bool,
    use_raw_eeg_features: bool,
    use_ai_uncertainty_zone: bool,
    use_sweet_spot_bonus: bool,
    use_fold_col: bool = False,
    fold_col: str = 'fold'
) -> pd.DataFrame:
    """Train L2A on ``df_train`` and evaluate on the held-out ``df_test``.

    For each deferral fraction ``d`` the returned frame gains columns
    ``gated_prob_d<dd>``, ``f1_pred_d<dd>``, ``youden_pred_d<dd>`` and
    ``deferred_d<dd>``. The gating threshold is derived from the final model's
    weights on the outer-training set (no test information is used), and exact
    top-k deferral is applied throughout.
    """
    print(f"\n{'='*60}")
    print(f"OUTER FOLD {fold}")
    print(f"{'='*60}")
    print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows")

    # Impute missing feature values with outer-training means only.
    if eeg_cols and df_train[eeg_cols].isna().any().any():
        train_means = df_train[eeg_cols].mean()
        df_train = df_train.copy()
        df_test = df_test.copy()
        df_train[eeg_cols] = df_train[eeg_cols].fillna(train_means)
        df_test[eeg_cols] = df_test[eeg_cols].fillna(train_means)

    X_train = df_train[eeg_cols].values if eeg_cols else None
    X_test = df_test[eeg_cols].values if eeg_cols else None

    ai_p_tr = df_train[ai_prob_col].values
    ai_d_tr = df_train[ai_pred_col].values.astype(int)
    hu_tr = df_train[human_col].values.astype(int)
    y_tr = df_train[groundtruth_col].values

    ai_p_te = df_test[ai_prob_col].values
    hu_te = df_test[human_col].values.astype(int)
    y_te = df_test[groundtruth_col].values

    # ---- Step 1: inner CV for the F1/Youden classification thresholds -------
    print("\n  [Step 1] Inner CV — learning F1/Youden classification thresholds...")
    class_thresh_map = find_classification_thresholds_inner_cv(
        df_train, eeg_cols, human_col, ai_prob_col, ai_pred_col,
        groundtruth_col, n_inner_folds, ml_model_type, deferral_fractions,
        use_ai_uncertainty, use_ai_probability, use_eeg_complexity,
        use_novice_prediction, use_ai_overconfidence, use_eeg_ai_mismatch,
        use_raw_eeg_features, use_ai_uncertainty_zone, use_sweet_spot_bonus,
        use_fold_col=use_fold_col, fold_col=fold_col
    )

    # ---- Step 2: train the final model on the full outer-training set -------
    print("\n  [Step 2] Training final model on full outer training set...")
    model = L2AWeightModel(
        ml_model=ml_model_type,
        use_ai_uncertainty=use_ai_uncertainty,
        use_ai_probability=use_ai_probability,
        use_eeg_complexity=use_eeg_complexity,
        use_novice_prediction=use_novice_prediction,
        use_ai_overconfidence=use_ai_overconfidence,
        use_eeg_ai_mismatch=use_eeg_ai_mismatch,
        use_raw_eeg_features=use_raw_eeg_features,
        use_ai_uncertainty_zone=use_ai_uncertainty_zone,
        use_sweet_spot_bonus=use_sweet_spot_bonus
    )
    model.fit(X_train, ai_p_tr, ai_d_tr, hu_tr, y_tr, verbose=True)

    # ---- Step 3: predict weights for train and test with the same model -----
    w_train = model.predict_weights(X_train, ai_p_tr, hu_tr)
    w_test = model.predict_weights(X_test, ai_p_te, hu_te)

    # ---- Step 4: apply thresholds for each deferral fraction ----------------
    # Collect all new columns first and assign them in one pass to avoid
    # repeated single-column insertions (and any index misalignment).
    new_cols_dict = {'deferral_weight': w_test}

    ai_auprc = average_precision_score(y_te, ai_p_te)
    print(f"\n  [Step 3] Applying thresholds on test fold {fold}...")
    for d in deferral_fractions:
        thresh = class_thresh_map[d]
        f1_thresh = thresh['f1_thresh']
        youden_thresh = thresh['youden_thresh']

        suffix = f"{d:.2f}".replace('0.', '').replace('.', '')
        col_prob = f'gated_prob_d{suffix}'
        col_f1 = f'f1_pred_d{suffix}'
        col_youden = f'youden_pred_d{suffix}'
        col_defer = f'deferred_d{suffix}'

        # Derive the gating threshold from the outer-training weights, then
        # apply that same threshold to the test fold (the achieved test
        # fraction may differ slightly from d).
        k_train = max(1, int(np.ceil(d * len(w_train))))
        w_thresh = float(np.sort(w_train)[-k_train])

        test_mask = (w_test >= w_thresh)
        actual_deferred = float(np.mean(test_mask))

        gated = gate_predictions(w_test, ai_p_te, hu_te.astype(float), test_mask)

        new_cols_dict[col_prob] = gated
        new_cols_dict[col_f1] = (gated >= f1_thresh).astype(int)
        new_cols_dict[col_youden] = (gated >= youden_thresh).astype(int)
        new_cols_dict[col_defer] = test_mask.astype(int)

        sys_auprc = average_precision_score(y_te, gated)
        print(f"    d={d:.2f} | w_thresh={w_thresh:.4f} | "
              f"deferred={actual_deferred*100:.1f}% | "
              f"AI AUPRC={ai_auprc:.4f} | Gated AUPRC={sys_auprc:.4f} | "
              f"\u0394={sys_auprc - ai_auprc:+.4f}")

    return df_test.assign(**new_cols_dict)


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def evaluate_deferral_system(
    csv_path: str,
    human: str = 'novice_1',
    deferral_fractions: Optional[List[float]] = None,
    output_dir: str = './output',
    n_outer_folds: int = 10,
    n_inner_folds: int = 9,
    ml_model: str = 'rf',
    ai_prob_col: str = 'RF_prob',
    ai_pred_col: str = 'RF_pred',
    groundtruth_col: str = 'Event',
    fold_col: str = 'fold',
    data_frac: float = 1.0,
    use_ai_uncertainty: bool = True,
    use_ai_probability: bool = True,
    use_eeg_complexity: bool = True,
    use_novice_prediction: bool = True,
    use_ai_overconfidence: bool = False,
    use_eeg_ai_mismatch: bool = False,
    use_raw_eeg_features: bool = False,
    use_ai_uncertainty_zone: bool = False,
    use_sweet_spot_bonus: bool = False
):
    """Run the full outer cross-validation and write out-of-fold predictions.

    Thresholds:
      * the gating (deferral) threshold is derived from the final outer-trained
        model's weights on the outer-training set;
      * the F1 and Youden classification thresholds are learned by inner CV
        (so no test information leaks into either).
    """
    os.makedirs(output_dir, exist_ok=True)

    if deferral_fractions is None:
        deferral_fractions = list(DEFAULT_DEFERRAL_FRACTIONS)
    deferral_fractions = sorted(set(deferral_fractions))
    for d in deferral_fractions:
        if not (0.0 < d <= 1.0):
            raise ValueError(f"Deferral fraction must be in (0, 1]. Got: {d}")

    df = pd.read_csv(csv_path)

    # Require unique (Case ID, Interval) keys so results merge back cleanly.
    if df.duplicated(['Case ID', 'Interval']).any():
        raise ValueError(
            "Input dataframe has duplicate (Case ID, Interval) keys — "
            "cannot safely merge results back. Fix your input data first."
        )

    all_cols = df.columns.tolist()
    eeg_cols = [c for c in all_cols if any(b in c for b in
                ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma',
                 'ADR', 'BDR', 'ABDTR', 'SEF90', 'aEEG', 'Asymmetry'])]

    use_fold_col = (fold_col in df.columns)

    # Each case must belong to exactly one fold (guards against leakage).
    if use_fold_col:
        folds_per_case = df.groupby('Case ID')[fold_col].nunique()
        inconsistent = folds_per_case[folds_per_case > 1]
        if len(inconsistent) > 0:
            raise ValueError(
                f"The following cases are assigned to multiple folds, "
                f"which would cause data leakage: "
                f"{inconsistent.index.tolist()[:10]}"
            )

    print(f"\n{'='*60}")
    print("DEFERRAL SYSTEM CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Annotator:          {human}")
    print(f"  ML model:           {ml_model.upper()}")
    print(f"  Outer folds:        {n_outer_folds}")
    print(f"  Inner folds:        {n_inner_folds}")
    print(f"  Deferral fractions: {deferral_fractions}")
    print(f"  EEG features found: {len(eeg_cols)}")
    print(f"  Use fold col:       {use_fold_col} ('{fold_col}')")
    print("  Features active:")
    print(f"    AI uncertainty:      {use_ai_uncertainty}")
    print(f"    AI probability:      {use_ai_probability}")
    print(f"    EEG complexity:      {use_eeg_complexity}")
    print(f"    Novice prediction:   {use_novice_prediction}")
    print(f"    AI overconfidence:   {use_ai_overconfidence}")
    print(f"    EEG-AI mismatch:     {use_eeg_ai_mismatch}")
    print(f"    Raw EEG features:    {use_raw_eeg_features}")
    print(f"    AI uncertainty zone: {use_ai_uncertainty_zone}")
    print(f"    Sweet spot bonus:    {use_sweet_spot_bonus}")
    print(f"{'='*60}\n")

    # Build outer splits from the fold column when present, else stratified.
    if use_fold_col:
        print(f"Using existing '{fold_col}' column for outer splits.")
        case_fold_map = df.groupby('Case ID')[fold_col].first().to_dict()
        unique_folds = sorted(df[fold_col].unique())
        case_splits = [
            [cid for cid, f in case_fold_map.items() if f == fv]
            for fv in unique_folds
        ]
        n_outer_folds = len(case_splits)
        verify_case_splits(case_splits, df, groundtruth_col,
                           split_name=f"OUTER CV ('{fold_col}' col)",
                           case_id_col='Case ID')
    else:
        case_splits = create_stratified_case_splits(
            df, n_outer_folds, groundtruth_col
        )

    all_fold_results = []

    for fold_idx in tqdm(range(n_outer_folds), desc=f"Outer CV [{human}]"):
        test_cases = case_splits[fold_idx]
        train_cases = [c for i, fold in enumerate(case_splits)
                       if i != fold_idx for c in fold]

        df_train = df[df['Case ID'].isin(train_cases)].copy()
        df_test = df[df['Case ID'].isin(test_cases)].copy()

        if df_train.empty or df_test.empty:
            print(f"  Fold {fold_idx}: empty split, skipping.")
            continue

        # Require both classes present in the outer train and test splits.
        if len(df_train[groundtruth_col].unique()) < 2:
            print(f"  Fold {fold_idx}: train missing a class, skipping.")
            continue
        if len(df_test[groundtruth_col].unique()) < 2:
            print(f"  Fold {fold_idx}: test missing a class, skipping.")
            continue

        # Optionally subsample training cases (data-efficiency experiments).
        if data_frac < 1.0:
            train_cases_sub = stratified_case_sampling(
                df_train, 'Case ID', groundtruth_col, data_frac,
                random_state=RANDOM_SEED + fold_idx
            )
            df_train = df_train[df_train['Case ID'].isin(train_cases_sub)].copy()

        fold_result = deferral_cv_fold(
            fold=fold_idx,
            df_train=df_train,
            df_test=df_test,
            eeg_cols=eeg_cols,
            human_col=human,
            ai_prob_col=ai_prob_col,
            ai_pred_col=ai_pred_col,
            groundtruth_col=groundtruth_col,
            ml_model_type=ml_model,
            deferral_fractions=deferral_fractions,
            n_inner_folds=n_inner_folds,
            use_ai_uncertainty=use_ai_uncertainty,
            use_ai_probability=use_ai_probability,
            use_eeg_complexity=use_eeg_complexity,
            use_novice_prediction=use_novice_prediction,
            use_ai_overconfidence=use_ai_overconfidence,
            use_eeg_ai_mismatch=use_eeg_ai_mismatch,
            use_raw_eeg_features=use_raw_eeg_features,
            use_ai_uncertainty_zone=use_ai_uncertainty_zone,
            use_sweet_spot_bonus=use_sweet_spot_bonus,
            use_fold_col=use_fold_col,
            fold_col=fold_col
        )
        all_fold_results.append(fold_result)

    if not all_fold_results:
        print("ERROR: No folds produced results.")
        return

    all_results = pd.concat(all_fold_results, ignore_index=True)

    # Save out-of-fold predictions.
    results_file = os.path.join(output_dir, f'deferral_{human}.csv')
    all_results.to_csv(results_file, index=False)
    print(f"\n[ok] Out-of-fold results saved to: {results_file}")

    # Merge results back into the original table.
    print("\nMerging results back into original dataframe...")

    new_cols = ['Case ID', 'Interval', 'deferral_weight']
    for d in deferral_fractions:
        suffix = f"{d:.2f}".replace('0.', '').replace('.', '')
        new_cols += [
            f'gated_prob_d{suffix}',
            f'f1_pred_d{suffix}',
            f'youden_pred_d{suffix}',
            f'deferred_d{suffix}',
        ]

    # OOF results must also have unique keys before merging.
    if all_results.duplicated(['Case ID', 'Interval']).any():
        raise ValueError(
            "OOF results have duplicate (Case ID, Interval) keys. "
            "This likely means a row appeared in multiple test folds — "
            "check your case splits."
        )

    merged = df.merge(all_results[new_cols], on=['Case ID', 'Interval'], how='left')
    if len(merged) != len(df):
        raise ValueError(
            f"Merge row count mismatch: {len(df)} before -> {len(merged)} after. "
            f"Check for duplicate (Case ID, Interval) keys."
        )

    full_file = os.path.join(output_dir, f'deferral_{human}_full.csv')
    merged.to_csv(full_file, index=False)
    print(f"[ok] Full dataframe saved to: {full_file}")

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY — {human} — {ml_model.upper()}")
    print(f"{'='*60}")
    print(f"{'Deferral':>10} | {'Deferred%':>10} | {'AI AUPRC':>10} | "
          f"{'Sys AUPRC':>10} | {'Delta':>8}")
    print(f"{'-'*60}")

    ai_auprc_global = average_precision_score(
        all_results[groundtruth_col], all_results[ai_prob_col]
    )

    for d in deferral_fractions:
        suffix = f"{d:.2f}".replace('0.', '').replace('.', '')
        col_prob = f'gated_prob_d{suffix}'
        col_def = f'deferred_d{suffix}'

        if col_prob not in all_results.columns:
            continue

        sys_auprc = average_precision_score(
            all_results[groundtruth_col], all_results[col_prob]
        )

        # Use the stored deferred indicator rather than recomputing a quantile.
        deferred_frac = float(all_results[col_def].mean()) \
            if col_def in all_results.columns else float('nan')

        print(f"{d:>10.2f} | {deferred_frac*100:>9.1f}% | "
              f"{ai_auprc_global:>10.4f} | {sys_auprc:>10.4f} | "
              f"{sys_auprc - ai_auprc_global:>+8.4f}")

    print(f"{'='*60}")


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the L2A soft human-AI deferral system with nested "
                    "patient-level cross-validation."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to the assembled interval-level table (e.g. data_mlhc.csv).",
    )
    parser.add_argument(
        "--output_dir", default="./output",
        help="Directory for the output CSVs (created if absent).",
    )
    parser.add_argument(
        "--annotator", default="novice_1",
        help="Column name of the novice's binary predictions in the CSV.",
    )
    parser.add_argument(
        "--ai_name", default="RF",
        help="Prefix of the AI columns, e.g. 'RF' -> RF_prob, RF_pred.",
    )
    parser.add_argument(
        "--ml_model", default="rf",
        choices=["rf", "logistic", "ridge", "lasso", "elasticnet", "xgb", "lgb"],
        help="Model family for the L2A weight estimator.",
    )
    parser.add_argument("--groundtruth_col", default="Event",
                        help="Ground-truth label column.")
    parser.add_argument("--fold_col", default="fold",
                        help="Patient-level CV fold column.")
    parser.add_argument("--n_outer_folds", type=int, default=10,
                        help="Outer CV folds (ignored if --fold_col is present).")
    parser.add_argument("--n_inner_folds", type=int, default=9,
                        help="Inner CV folds for threshold selection.")
    parser.add_argument("--data_frac", type=float, default=1.0,
                        help="Fraction of training cases to use "
                             "(for data-efficiency experiments).")
    return parser.parse_args()


def main():
    args = parse_args()
    ai_prob_col = f"{args.ai_name}_prob"
    ai_pred_col = f"{args.ai_name}_pred"

    evaluate_deferral_system(
        csv_path=args.csv,
        human=args.annotator,
        deferral_fractions=DEFAULT_DEFERRAL_FRACTIONS,
        output_dir=args.output_dir,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
        ml_model=args.ml_model,
        ai_prob_col=ai_prob_col,
        ai_pred_col=ai_pred_col,
        groundtruth_col=args.groundtruth_col,
        fold_col=args.fold_col,
        data_frac=args.data_frac,
        # Canonical L2A feature set z = (p, u, v); the novice prediction is not
        # itself an input feature, and the remaining flags are experimental.
        use_ai_uncertainty=True,
        use_ai_probability=True,
        use_eeg_complexity=True,
        use_novice_prediction=False,
        use_ai_overconfidence=False,
        use_eeg_ai_mismatch=False,
        use_raw_eeg_features=False,
        use_ai_uncertainty_zone=False,
        use_sweet_spot_bonus=False,
    )


if __name__ == "__main__":
    main()
