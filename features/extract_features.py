"""
cEEG feature extraction for intraoperative ischemia detection.

Converts raw continuous-EEG recordings from carotid endarterectomy (CEA) cases
into the 111-dimensional per-interval feature vectors used by the AI and
human-AI deferral models (see the paper's Appendices A and B).

For each case, the pipeline:
  1. locates the carotid-clamp time from the annotation log,
  2. extracts a fixed post-clamp window and an artifact-free pre-clamp baseline,
  3. splits the post-clamp signal into fixed-length intervals,
  4. band-pass / notch filters each interval and estimates its power spectrum,
  5. computes band powers, band-power ratios (ADR, BDR, ABDTR), spectral edge
     frequency (SEF90), amplitude-integrated EEG (aEEG), and an interhemispheric
     asymmetry index (pdBSI), aggregated per channel, per hemisphere, and
     globally to give 111 features per interval, and
  6. writes raw, baseline-normalized (z-score), and percent-change feature
     tables per case.

Expected input layout
----------------------
    <input_parsed_path>/
        <case_id>/
            WAVEFORM/*.csv          # column 0 = time (s), columns 1..8 = bipolar
                                    # EEG channels sampled at 500 Hz
            ANNOTATION/*.annotation # CSV with columns [Time, Comment];
                                    # surgical/tech comments including clamp events

Metadata table (`input_meta_path`, CSV) must contain at least:
    Case ID, Procedure notes (L/R/Both, mapped to operative side),
    EEG change start time, EEG change end time  (HH:MM:SS or 12-hour clock)

Output layout (under --output_dir)
----------------------------------
    original/     raw feature values per interval
    normalized/   z-scored to the per-case pre-clamp baseline
    percent/      percent change from the per-case pre-clamp baseline
    baseline/     the computed baseline feature vector per case
    raw_baseline/ the raw baseline signal segment per case
Each per-interval row is keyed by [Case ID, Interval, Start Time, End Time,
Event, Location] followed by the 111 feature columns.

Usage
-----
    python extract_features.py \
        --config configs/feature_extraction.yaml \
        --output_dir ./output/features

    # Optional: shard the case list across parallel array jobs
    python extract_features.py \
        --config configs/feature_extraction.yaml \
        --output_dir ./output/features \
        --splits 8 --split_n 0

For large datasets, increase joblib parallelism with --n_jobs and/or shard the
work with --splits / --split_n. Note that the spectrogram uses a 0.5 Hz
frequency resolution (nperseg = 2 * fs); any downstream code that assumes this
spacing must be updated if the spectrogram settings are changed.
"""

__author__ = "Amir Mina"
# Adapted for public release (documentation and path handling) as part of the
# MLHC 2026 paper "Human-AI Deferral under Limited Expert Availability".

import argparse
import multiprocessing
import os
import pickle as pkl
import traceback
from datetime import datetime, timedelta
from itertools import product

import bottleneck as bn
import numpy as np
import pandas as pd
import scipy.signal as signal
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm
from tsai.data.preparation import SlidingWindow


# ---------------------------------------------------------------------------
# Acquisition / processing constants
# ---------------------------------------------------------------------------

SAMPLING_RATE_HZ = 500      # cEEG acquisition rate for this dataset
FILTER_ORDER = 50           # Butterworth band-pass order
NOTCH_FREQ_HZ = 60.0        # power-line notch frequency
NOTCH_Q = 1.0               # power-line notch quality factor
SEF_THRESHOLD = 0.9         # cumulative-power fraction for spectral edge frequency

# Bipolar montage: four channels per hemisphere (10-20 system).
CHANNEL_LIST = ["T3-O1", "P3-O1", "F3-T3", "F3-P3",
                "F4-P4", "F4-T4", "P4-O2", "T4-O2"]

# Standard EEG frequency bands (Hz).
BAND_KEYS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
BAND_RANGES = [(0, 4), (4, 8), (8, 12), (12, 30), (30, 45)]
eeg_bands = dict(zip(BAND_KEYS, BAND_RANGES))


# ---------------------------------------------------------------------------
# Case file discovery
# ---------------------------------------------------------------------------

def get_signals_path(path):
    """Return the waveform CSV inside a case directory's WAVEFORM/ folder."""
    signals_path = os.path.join(path, "WAVEFORM")
    signals = [f for f in os.listdir(signals_path) if f.endswith(".csv")][0]
    return os.path.join(signals_path, signals)


def get_comments_path(path):
    """Return the annotation file inside a case directory's ANNOTATION/ folder."""
    comments_path = os.path.join(path, "ANNOTATION")
    comments = [f for f in os.listdir(comments_path) if f.endswith(".annotation")][0]
    return os.path.join(comments_path, comments)


# ---------------------------------------------------------------------------
# Clamp / event timing
# ---------------------------------------------------------------------------

def cea_start(comments):
    """Estimate the carotid-clamp time (seconds) from surgical/tech comments.

    Matches comment text against clamp-on phrasings while excluding unclamp,
    shunt, and other non-clamp mentions, then returns the timestamp of the
    first remaining match.

    Parameters
    ----------
    comments : pandas.DataFrame
        Two-column table [Time, Comment] parsed from the annotation file.
    """
    comments["Comment"] = comments["Comment"].astype(str).str.lower()
    include = (
        "clamp on|clamped|clamnp on|clamp is on|clamp R ICA|Clamp of R ICA|"
        "clamping common again|clamping common|clamping ICA|clamp ica on left|"
        "clamping ICA|ICA clamp|clamping L ICA|clamp ICA|xclamp R ica|"
        "placing clamp|cross clamp|clamp in"
    )
    clamp = comments[comments["Comment"].str.contains(include)]
    exclude = (
        "unclamp|un-clamp|un clamp|off|temp|remov|change|not|no|min|while|time|"
        "remain|shunt|cuff|bleeding|ready|eeg|request|prep|adjust"
    )
    clamp = clamp[~clamp["Comment"].str.contains(exclude)]
    clamp = clamp.drop(columns=["Comment"])
    return clamp.iloc[0, 0]


def procedure_start(comments, procedure="CEA"):
    """Return the procedure-start (clamp) time for the given procedure type.

    Only CEA is supported in this release.
    """
    if procedure == "CEA":
        return cea_start(comments)
    raise ValueError(
        f"Only the 'CEA' procedure is supported in this release; got {procedure!r}."
    )


def clean_meta_data(meta_data):
    """Normalize operative-side labels and rename the side column."""
    side_map = {
        "L": "Left", "Left": "Left", "left": "Left",
        "R": "Right", "Right": "Right", "right": "Right",
        "Both": "Both", "both": "Both",
    }
    meta_data["Procedure notes"] = meta_data["Procedure notes"].map(side_map).astype(str)
    return meta_data.rename(columns={"Procedure notes": "side"})


def time_to_sec(x):
    """Convert an HH:MM:SS (or 12-hour) time string to seconds; NaN if not a string."""
    if isinstance(x, str):
        fmt = "%I:%M:%S %p" if " " in x else "%H:%M:%S"
        y = datetime.strptime(x, fmt)
        return timedelta(hours=y.hour, minutes=y.minute, seconds=y.second).total_seconds()
    return np.nan


# ---------------------------------------------------------------------------
# Signal loading / windowing
# ---------------------------------------------------------------------------

def loadts(input_path):
    """Load a waveform CSV as a float array (dropping the trailing row) plus columns."""
    ts_df = pd.read_csv(input_path, delimiter=",", dtype=np.float64)
    return ts_df.values[:-1, :], ts_df.columns


def windowts(ts, length, stride, y_func=None):
    """Slide a fixed-length window over the time series.

    Returns the windowed channel data, the per-window time stamps, and the
    per-window labels produced by `y_func`. Column 0 (time) is separated out;
    channel columns are retained for feature computation.
    """
    X, y = SlidingWindow(length, stride, pad_remainder=True, get_y=[0], y_func=y_func)(ts)
    t = X[:, 0, :]
    X = X[:, 1:X.shape[1] - 1, :]
    return X, t, y


def fftspec(X, fs):
    """Estimate per-window spectrograms.

    Uses a 2-second Hann window with 1-second hop, giving 0.5 Hz frequency
    resolution.
    """
    nfft = fs * 2
    freqs, times, Sx = signal.spectrogram(
        X, fs=fs, window="hann", nperseg=nfft, noverlap=nfft - fs,
        detrend=False, scaling="spectrum",
    )
    return freqs, times, Sx


def splitts(ts, main_start, forwl=None, backwl=None):
    """Split the time series at `main_start` into forward and backward segments.

    `forwl` / `backwl` bound the number of samples kept after / before the split
    (None keeps everything on that side).
    """
    split_idx = bn.nanargmax(ts[:, 0] >= main_start)
    back_idx = 0 if backwl is None else split_idx - backwl
    forw_idx = -1 if forwl is None else split_idx + forwl
    ts_back = ts[back_idx:split_idx, :]
    ts_forw = ts[split_idx:forw_idx, :]
    return ts_forw, ts_back


def baseline_finder(ts_back, bll):
    """Locate the quietest baseline window in the pre-clamp segment.

    Returns (start, stop) sample indices of the length-`bll` window with the
    lowest average moving variance across channels.
    """
    window_vars = bn.move_var(ts_back[:, 1:].T, window=int(bll))
    window_var_means = bn.nanmean(window_vars, axis=0)
    baseline_stop = bn.nanargmin(window_var_means)
    baseline_start = baseline_stop - bll
    return (baseline_start, baseline_stop)


# ---------------------------------------------------------------------------
# Interval labeling
# ---------------------------------------------------------------------------

def overlap_percent(a, b):
    """Fraction of interval `a` that overlaps interval `b`."""
    length = a[1] - a[0]
    if (a[1] <= b[1]) and (a[0] >= b[0]):
        o = length
    elif (a[1] >= b[0]) and (a[1] <= b[1]) and (a[0] <= b[0]):
        o = a[1] - b[0]
    elif (a[1] >= b[1]) and (a[0] >= b[0]) and (a[0] <= b[1]):
        o = b[1] - a[0]
    elif (a[1] >= b[1]) and (a[0] <= b[0]):
        o = b[1] - b[0]
    else:
        o = 0
    return o / length


def y_func_indiv(a, b_list, threshold):
    """Label interval `a` as ischemic (1) if it overlaps any annotated ischemic
    period in `b_list` by more than `threshold`; otherwise 0."""
    if np.all(np.isnan(np.array(b_list[0]))):
        return 0
    maxp = np.max([overlap_percent(a, b) for b in b_list])
    return 1 if maxp > threshold else 0


# ---------------------------------------------------------------------------
# Spectral / amplitude features
# ---------------------------------------------------------------------------

def db_pwr(Sx):
    """Convert a power spectrogram to decibels (with a small-value floor)."""
    Sx[Sx == 0] = np.quantile(Sx.flatten(), 0.01)
    return 10 * np.log10(Sx * 1e6)


def sef(Sx, freqs, threshold=SEF_THRESHOLD):
    """Spectral edge frequency: the frequency below which `threshold` of the
    cumulative spectral power lies."""
    pwr_cs = np.cumsum(Sx, axis=2)
    sef_value = np.apply_along_axis(
        lambda x: bn.nanargmax(x > bn.nanmax(x) * threshold), 2, pwr_cs
    )
    return freqs[sef_value]


def asymmetry(left, right):
    """Normalized left/right power difference, summed over frequency."""
    value = np.abs((right - left) / (right + left))
    return np.apply_along_axis(np.sum, 1, value)


def pdbsi(Sx):
    """Pairwise-derived brain symmetry index across the four homologous pairs."""
    value = 0
    value += asymmetry(Sx[:, 0, :, :], Sx[:, -1, :, :])
    value += asymmetry(Sx[:, 1, :, :], Sx[:, -2, :, :])
    value += asymmetry(Sx[:, 2, :, :], Sx[:, -3, :, :])
    value += asymmetry(Sx[:, 3, :, :], Sx[:, -4, :, :])
    return np.expand_dims(value, axis=1)


def featurize(X, Sx, freqs, fs, channels):
    """Compute the per-interval feature matrix.

    Produces, for each interval: band powers, band-power ratios (ADR, BDR,
    ABDTR), SEF90, and aEEG for every channel; the same quantities on the
    left-hemisphere, right-hemisphere, and all-channel averages; and one pdBSI
    asymmetry feature. Returns the per-interval mean and standard deviation of
    each quantity over the within-interval time axis.
    """
    # Odd trailing channel index -> left hemisphere; even -> right.
    channel_side = np.array([int(channel[-1]) % 2 for channel in channels])
    left_idx = np.where(channel_side == 1)[0]
    right_idx = np.where(channel_side == 0)[0]

    delta_freq_ix = np.where((freqs >= eeg_bands["Delta"][0]) & (freqs <= eeg_bands["Delta"][1]))[0]
    theta_freq_ix = np.where((freqs >= eeg_bands["Theta"][0]) & (freqs <= eeg_bands["Theta"][1]))[0]
    alpha_freq_ix = np.where((freqs >= eeg_bands["Alpha"][0]) & (freqs <= eeg_bands["Alpha"][1]))[0]
    beta_freq_ix = np.where((freqs >= eeg_bands["Beta"][0]) & (freqs <= eeg_bands["Beta"][1]))[0]
    gamma_freq_ix = np.where((freqs >= eeg_bands["Gamma"][0]) & (freqs <= eeg_bands["Gamma"][1]))[0]

    def pwr_features(Sx):
        pwr = db_pwr(Sx)
        delta_pwr = bn.nanmean(pwr[:, :, delta_freq_ix, :], axis=2)
        theta_pwr = bn.nanmean(pwr[:, :, theta_freq_ix, :], axis=2)
        alpha_pwr = bn.nanmean(pwr[:, :, alpha_freq_ix, :], axis=2)
        beta_pwr = bn.nanmean(pwr[:, :, beta_freq_ix, :], axis=2)
        gamma_pwr = bn.nanmean(pwr[:, :, gamma_freq_ix, :], axis=2)
        adr = alpha_pwr / delta_pwr
        bdr = beta_pwr / delta_pwr
        abdtr = (alpha_pwr + beta_pwr) / (delta_pwr + theta_pwr)
        sef90 = sef(Sx, freqs, SEF_THRESHOLD)
        return np.concatenate(
            [delta_pwr, theta_pwr, alpha_pwr, beta_pwr, gamma_pwr,
             adr, bdr, abdtr, sef90], axis=1
        )

    # Per-channel features.
    channel_feats = pwr_features(Sx)
    aeeg = bn.move_max(X, window=fs) - bn.move_min(X, window=fs)

    # Left-hemisphere average.
    Sx_left = np.expand_dims(bn.nanmean(Sx[:, left_idx, :, :], axis=1), 1)
    X_left = np.expand_dims(bn.nanmean(X[:, left_idx, :], axis=1), 1)
    left_feats = pwr_features(Sx_left)
    left_aeeg = bn.move_max(X_left, window=fs) - bn.move_min(X_left, window=fs)

    # Right-hemisphere average.
    Sx_right = np.expand_dims(bn.nanmean(Sx[:, right_idx, :, :], axis=1), 1)
    X_right = np.expand_dims(bn.nanmean(X[:, right_idx, :], axis=1), 1)
    right_feats = pwr_features(Sx_right)
    right_aeeg = bn.move_max(X_right, window=fs) - bn.move_min(X_right, window=fs)

    # All-channel (bilateral) average.
    Sx_bilat = np.expand_dims(bn.nanmean(Sx, axis=1), 1)
    X_bilat = np.expand_dims(bn.nanmean(X, axis=1), 1)
    bilat_feats = pwr_features(Sx_bilat)
    bilat_aeeg = bn.move_max(X_bilat, window=fs) - bn.move_min(X_bilat, window=fs)

    bsi = pdbsi(Sx)

    listx = [channel_feats, aeeg, left_feats, left_aeeg,
             right_feats, right_aeeg, bilat_feats, bilat_aeeg, bsi]

    xmean = np.concatenate([bn.nanmean(x, -1) for x in listx], -1)
    xstd = np.concatenate([bn.nanstd(x, -1) for x in listx], -1)
    return xmean, xstd


# ---------------------------------------------------------------------------
# Work sharding
# ---------------------------------------------------------------------------

def chunk_paths(paths, n_splits, split_index):
    """Return the `split_index`-th chunk of `paths` split into `n_splits` groups.

    Used to shard the case list across parallel array jobs.
    """
    return list(np.array_split(np.array(paths, dtype=object), n_splits)[split_index])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    config = yaml.safe_load(open(args.config_path, "rb"))

    parsed_path = config["input_parsed_path"]
    meta_path = config["input_meta_path"]

    bpf = [config["l_freq"], config["h_freq"]]
    window_length = config["epoch_size"]
    window_stride = config["epoch_stride"]
    forward_duration = config["duration"]
    baseline_length = config["baseline_length"]

    procedure = config.get("procedure", "CEA")
    if procedure != "CEA":
        raise ValueError(
            f"Only the 'CEA' procedure is supported in this release; got {procedure!r}."
        )

    population_baselines = config.get("population_baselines")
    if population_baselines is not None and not os.path.isfile(population_baselines):
        raise ValueError(f"population_baselines file not found: {population_baselines}")

    n_splits = args.splits
    split_index = args.split_n
    n_jobs = args.n_jobs if args.n_jobs != -1 else multiprocessing.cpu_count()

    # Resolve the list of case directories to process.
    case_list_path = config.get("case_list_path")
    if case_list_path:
        if case_list_path.endswith(".pkl"):
            with open(case_list_path, "rb") as fh:
                paths = pkl.load(fh)
        else:
            with open(case_list_path) as fh:
                paths = [line.strip() for line in fh if line.strip()]
    else:
        paths = [
            os.path.join(parsed_path, d)
            for d in sorted(os.listdir(parsed_path))
            if os.path.isdir(os.path.join(parsed_path, d))
        ]

    # Prepare output directories.
    output_dir = args.output_dir
    output_path_original = os.path.join(output_dir, "original")
    output_path_baseline = os.path.join(output_dir, "baseline")
    output_path_normalized = os.path.join(output_dir, "normalized")
    output_path_percent = os.path.join(output_dir, "percent")
    output_path_raw_baseline = os.path.join(output_dir, "raw_baseline")
    for d in (output_dir, output_path_original, output_path_baseline,
              output_path_normalized, output_path_percent, output_path_raw_baseline):
        os.makedirs(d, exist_ok=True)

    # Load and normalize metadata.
    meta_data = clean_meta_data(pd.read_csv(meta_path, dtype=object))
    meta_data["EEG change start time"] = meta_data["EEG change start time"].apply(time_to_sec)
    meta_data["EEG change end time"] = meta_data["EEG change end time"].apply(time_to_sec)

    def make_spec_array(path):
        # Case id is the leading token of the directory / file name.
        r = os.path.basename(path).split(".")[0].split("-")[0]
        record_info = meta_data[meta_data["Case ID"] == r]

        def y_func(x, window_length=window_length, record_info=record_info, threshold=0.5):
            a_list = list(zip(x - window_length, x))
            b_list = list(zip(record_info["EEG change start time"],
                              record_info["EEG change end time"]))
            return np.array([y_func_indiv(a, b_list, threshold) for a in a_list])

        try:
            # Skip cases with no matching clinical record.
            if record_info.shape[0] == 0:
                print(f"[skip] {r}: missing from clinical/metadata table")
                return

            side = record_info["side"].values[0]

            # Skip cases whose operative side is undetermined.
            if procedure == "CEA" and side not in ["Left", "Right", "Both"]:
                print(f"[skip] {r}: operative side not determined")
                return

            ts_path, comments_path = get_signals_path(path), get_comments_path(path)
            comments = pd.read_csv(comments_path, header=None,
                                   names=["Time", "Comment"], skiprows=1,
                                   usecols=range(2))
            main_start = procedure_start(comments, procedure=procedure)

            # Skip cases with no detectable clamp time.
            if main_start is None:
                print(f"[skip] {r}: procedure start time not found")
                return

            ts, columns = loadts(ts_path)
            channels = columns[1:]

            # Skip cases with an unexpected number of channels.
            if len(channels) != len(CHANNEL_LIST):
                print(f"[skip] {r}: unexpected channel count ({len(channels)})")
                return

            fs = SAMPLING_RATE_HZ

            # Split at the clamp time; keep `forward_duration` seconds afterward.
            forwl = fs * forward_duration if forward_duration is not None else None
            ts_forw, ts_back = splitts(ts, main_start, forwl)

            # If there is too little pre-clamp signal for a baseline, fall back to
            # a window taken later in the recording.
            if main_start - ts[0, 0] < baseline_length:
                fake_start = ts[0, 0] + 600
                _, ts_back = splitts(ts, fake_start, forwl)

            # Locate the baseline window.
            bll = fs * baseline_length
            bl_times = baseline_finder(ts_back, bll)

            # Window the post-clamp signal.
            wl = fs * window_length
            ws = int(fs * window_stride)
            X, t, y = windowts(ts_forw, wl, ws, y_func)

            # Drop the trailing (possibly partial) window when the duration is open-ended.
            if forward_duration is None:
                X = X[:-1]
                y = y[:-1]
                t = t[:-1]

            # Band-pass and power-line notch filter each window.
            sos = signal.butter(FILTER_ORDER, bpf, "bp", fs=fs, output="sos")
            X = signal.sosfiltfilt(sos, X)
            b_notch, a_notch = signal.iirnotch(NOTCH_FREQ_HZ, NOTCH_Q, fs)
            X = signal.filtfilt(b_notch, a_notch, X)

            # Spectrogram per window.
            freqs, times, Sx = fftspec(X, fs)

            # Assemble the per-interval index columns.
            case_id = np.expand_dims(np.repeat(r, y.shape[0]), 1)
            interval = np.expand_dims(np.arange(y.shape[0]), 1)
            interval_times = t[:, [0, -1]]
            event = np.expand_dims(y, 1)
            side_bin = 1 if side == "Left" else 2
            location = np.expand_dims(np.repeat(side_bin, y.shape[0]), 1)
            spec_array_info = np.concatenate(
                [case_id, interval, interval_times, event, location], axis=1
            )

            # Build feature column names (per-channel, then hemisphere/global, then pdBSI).
            info_names = ["Case ID", "Interval", "Start Time", "End Time", "Event", "Location"]
            feature_names = BAND_KEYS + ["ADR", "BDR", "ABDTR", "SEF90", "aEEG"]
            feature_names_channel = list(product(feature_names, channels))
            feature_names_channel = [name[0] + "_" + name[1] for name in feature_names_channel]
            feature_names_sides = (
                [name + "_1" for name in feature_names]
                + [name + "_2" for name in feature_names]
                + [name + "_0" for name in feature_names]
            )
            feature_names = feature_names_channel + feature_names_sides + ["Asymmetry"]
            cols = info_names + feature_names

            # Baseline features: either a precomputed population baseline or a
            # per-case baseline estimated from the pre-clamp window.
            if population_baselines is not None:
                # Experimental path: expects a CSV whose first two rows are the
                # baseline mean and std feature vectors. Verify the schema for
                # your data before relying on this.
                population_bl = pd.read_csv(population_baselines)
                features_bl_mean = population_bl.iloc[0, :].to_numpy(dtype=float)
                features_bl_std = population_bl.iloc[1, :].to_numpy(dtype=float)
            else:
                # Filter and featurize the raw baseline segment, and save it.
                X_bl = ts_back[bl_times[0]:bl_times[1], 1:].T
                bldf = pd.DataFrame(X_bl.T, columns=CHANNEL_LIST)
                bl_path = os.path.join(output_path_raw_baseline, r + "-raw-baseline.csv")
                bldf.to_csv(bl_path)

                X_bl = np.expand_dims(X_bl, 0)
                X_bl = signal.sosfiltfilt(sos, X_bl)
                X_bl = signal.filtfilt(b_notch, a_notch, X_bl)
                n_bl = X_bl.shape[-1]
                freqs_bl, times_bl, Sx_bl = fftspec(X_bl, fs)
                features_bl_mean, features_bl_std = featurize(X_bl, Sx_bl, freqs_bl, fs, channels)

                # Record the baseline feature vector (and its sample count) per case.
                base_array = np.concatenate([features_bl_mean, features_bl_std], 0)
                base_info = spec_array_info[0:2, :].copy()
                base_info[:, 1] = np.array([-1, -1])
                base_info[:, 2:4] = np.repeat(np.array(np.expand_dims(bl_times, 1)), 2, axis=-1).T
                base_info[:, -2] = np.array([-1, -1])
                base_arr = np.concatenate([base_info, base_array], axis=-1)
                base_df = pd.DataFrame(base_arr, columns=cols)
                new_row = pd.DataFrame([[np.nan] * len(cols)], columns=cols)
                new_row.iloc[0, 0] = "n"
                new_row.iloc[0, 1] = n_bl
                base_df = pd.concat([base_df, new_row], ignore_index=True)
                spec_array_b_path = os.path.join(output_path_baseline, r + "-baseline_spec_array.csv")
                base_df.to_csv(spec_array_b_path, index=False)

            # Interval features, plus baseline-normalized and percent-change versions.
            features, _ = featurize(X, Sx, freqs, fs, channels)
            norm_features = (features - features_bl_mean) / features_bl_std
            perc_features = (features - features_bl_mean) / features_bl_mean * 100

            orig_df = pd.DataFrame(np.concatenate([spec_array_info, features], axis=-1), columns=cols)
            orig_df.to_csv(os.path.join(output_path_original, r + "-spec_array.csv"), index=False)

            norm_df = pd.DataFrame(np.concatenate([spec_array_info, norm_features], axis=-1), columns=cols)
            norm_df.to_csv(os.path.join(output_path_normalized, r + "-normalized_spec_array.csv"), index=False)

            perc_df = pd.DataFrame(np.concatenate([spec_array_info, perc_features], axis=-1), columns=cols)
            perc_df.to_csv(os.path.join(output_path_percent, r + "-percent_spec_array.csv"), index=False)

            print(f"[ok]   {r}: {y.shape[0]} intervals written")

        except Exception as exc:
            print(f"[error] {r}: {exc}")
            traceback.print_exc()

    if n_splits > 0 and split_index > -1:
        selected = chunk_paths(paths, n_splits, split_index)
        Parallel(n_jobs=n_jobs, backend="loky")(delayed(make_spec_array)(p) for p in selected)
    else:
        Parallel(n_jobs=n_jobs, backend="loky")(delayed(make_spec_array)(p) for p in tqdm(paths))


def main():
    parser = argparse.ArgumentParser(
        description="Extract 111 cEEG features per interval for CEA cases."
    )
    parser.add_argument(
        "--output_dir", required=True, type=str,
        help="Output directory (created if absent), e.g. ./output/features",
    )
    parser.add_argument(
        "--config", dest="config_path", type=str,
        default="configs/feature_extraction.yaml",
        help="Path to the feature-extraction config YAML.",
    )
    parser.add_argument(
        "--splits", dest="splits", type=int, default=0,
        help="Split the case list into this many shards (0 disables sharding).",
    )
    parser.add_argument(
        "--split_n", dest="split_n", type=int, default=-1,
        help="Which shard to process (0-indexed; requires --splits).",
    )
    parser.add_argument(
        "--n_jobs", dest="n_jobs", type=int, default=-1,
        help="Number of parallel workers (-1 uses all available CPUs).",
    )
    parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
