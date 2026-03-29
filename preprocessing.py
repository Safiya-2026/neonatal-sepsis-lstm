"""
Data Preprocessing Utilities
==============================
Handles loading, cleaning, normalizing, and windowing
neonatal physiological time-series data for the LSTM Autoencoder.

Expected raw features (configurable):
  - Heart Rate (HR)           — bpm
  - Respiratory Rate (RR)     — breaths/min
  - SpO2                      — %
  - Temperature               — °C
  - Systolic BP               — mmHg
  - Diastolic BP              — mmHg
  - Mean Arterial Pressure    — mmHg
  - Blood Glucose             — mmol/L
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


FEATURE_COLUMNS = [
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "temperature",
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "blood_glucose"
]


# ─────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────
PHYSIOLOGICAL_BOUNDS = {
    "heart_rate":              (60,  220),
    "respiratory_rate":        (20,  100),
    "spo2":                    (60,  100),
    "temperature":             (34,   42),
    "systolic_bp":             (20,  120),
    "diastolic_bp":            (10,   90),
    "mean_arterial_pressure":  (15,  100),
    "blood_glucose":           (1.5,  25),
}


def remove_physiological_outliers(df, bounds=None):
    """
    Replace values outside physiological plausibility bounds with NaN.

    Args:
        df     : DataFrame with feature columns
        bounds : Dict {col: (min, max)}; defaults to PHYSIOLOGICAL_BOUNDS

    Returns:
        Cleaned DataFrame
    """
    if bounds is None:
        bounds = PHYSIOLOGICAL_BOUNDS
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def impute_missing(df, method="linear"):
    """
    Interpolate then forward/backward fill remaining NaNs.

    Args:
        df     : DataFrame
        method : Interpolation method ("linear", "time", "nearest")

    Returns:
        Imputed DataFrame
    """
    df = df.interpolate(method=method, limit_direction="both")
    df = df.ffill().bfill()
    return df


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────
def fit_scaler(data_np, save_path=None):
    """
    Fit a MinMaxScaler on training data.

    Args:
        data_np   : Array of shape (n_samples, seq_len, n_features)
        save_path : If provided, saves scaler as .pkl

    Returns:
        Fitted scaler
    """
    n_samples, seq_len, n_features = data_np.shape
    flat = data_np.reshape(-1, n_features)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(flat)
    if save_path:
        joblib.dump(scaler, save_path)
        print(f"Scaler saved to: {save_path}")
    return scaler


def apply_scaler(data_np, scaler):
    """
    Apply a fitted scaler to windowed data.

    Args:
        data_np : (n_samples, seq_len, n_features)
        scaler  : Fitted MinMaxScaler

    Returns:
        Scaled array of same shape
    """
    n_samples, seq_len, n_features = data_np.shape
    flat = data_np.reshape(-1, n_features)
    scaled = scaler.transform(flat)
    return scaled.reshape(n_samples, seq_len, n_features)


# ─────────────────────────────────────────────
# Sliding Window
# ─────────────────────────────────────────────
def create_windows(df, seq_len=24, step=1, feature_cols=None):
    """
    Convert a time-series DataFrame into overlapping windows.

    Args:
        df           : DataFrame, one row = one time step (e.g., 1 hour)
        seq_len      : Window length (number of time steps)
        step         : Stride between consecutive windows
        feature_cols : List of columns to include; defaults to FEATURE_COLUMNS

    Returns:
        windows : np.ndarray of shape (n_windows, seq_len, n_features)
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    values = df[feature_cols].values.astype(np.float32)
    windows = []
    for start in range(0, len(values) - seq_len + 1, step):
        windows.append(values[start:start + seq_len])
    return np.array(windows)


# ─────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────
def preprocess_patient(df, seq_len=24, step=1,
                       scaler=None, scaler_save_path=None,
                       feature_cols=None):
    """
    Full preprocessing pipeline for a single patient's record.

    Steps:
      1. Remove physiological outliers
      2. Interpolate missing values
      3. Slide windows
      4. Normalize (fit scaler if not provided)

    Args:
        df                : Raw patient DataFrame
        seq_len           : Window length
        step              : Stride
        scaler            : Pre-fitted scaler (None to fit a new one)
        scaler_save_path  : Path to save fitted scaler
        feature_cols      : Feature columns to use

    Returns:
        windows_scaled : (n_windows, seq_len, n_features)
        scaler         : The scaler used (fitted if newly created)
    """
    df = remove_physiological_outliers(df)
    df = impute_missing(df)
    windows = create_windows(df, seq_len=seq_len, step=step,
                             feature_cols=feature_cols)

    if scaler is None:
        scaler = fit_scaler(windows, save_path=scaler_save_path)

    windows_scaled = apply_scaler(windows, scaler)
    return windows_scaled, scaler


# ─────────────────────────────────────────────
# Synthetic Demo Data Generator
# ─────────────────────────────────────────────
def generate_synthetic_data(n_normal=800, n_sepsis=200,
                             seq_len=24, n_features=8,
                             seed=42, save_dir="data/"):
    """
    Generates synthetic neonatal physiological data for testing.

    Normal signals: low-amplitude, stationary noise.
    Sepsis signals: higher amplitude, sudden spikes, trend shifts.

    Args:
        n_normal   : Number of normal windows
        n_sepsis   : Number of sepsis windows
        seq_len    : Sequence length
        n_features : Number of features
        seed       : Random seed
        save_dir   : Where to save .npy files

    Returns:
        normal_data, sepsis_data : np.ndarrays
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Normal: stationary around 0.5 ± 0.1
    normal_data = (rng.standard_normal((n_normal, seq_len, n_features))
                   * 0.1 + 0.5).astype(np.float32)

    # Sepsis: larger variance + abrupt spike in last quarter
    sepsis_data = (rng.standard_normal((n_sepsis, seq_len, n_features))
                   * 0.3 + 0.5).astype(np.float32)
    spike_start = int(seq_len * 0.75)
    sepsis_data[:, spike_start:, :] += rng.uniform(
        0.3, 0.8, size=(n_sepsis, seq_len - spike_start, n_features)
    ).astype(np.float32)

    # Train/test split
    train_normal = normal_data[:int(n_normal * 0.8)]
    test_normal  = normal_data[int(n_normal * 0.8):]
    test_sepsis  = sepsis_data

    np.save(os.path.join(save_dir, "normal_signals.npy"), train_normal)
    np.save(os.path.join(save_dir, "test_normal.npy"),    test_normal)
    np.save(os.path.join(save_dir, "test_sepsis.npy"),    test_sepsis)

    print(f"Synthetic data saved to '{save_dir}':")
    print(f"  normal_signals.npy : {train_normal.shape}")
    print(f"  test_normal.npy    : {test_normal.shape}")
    print(f"  test_sepsis.npy    : {test_sepsis.shape}")

    return train_normal, test_normal, test_sepsis


if __name__ == "__main__":
    generate_synthetic_data()
