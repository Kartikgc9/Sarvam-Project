"""
Spectral-domain audio quality metrics.

Metrics
-------
spectral_flatness  : Wiener entropy — ratio of geometric mean to arithmetic mean
                     of the power spectrum.  Near 1 → noise-like (flat spectrum).
                     Near 0 → tonal / speech-like (peaky spectrum).
                     High flatness in a labelled speech file → likely noise or music.

zcr                : Zero-Crossing Rate — mean rate (per second) at which the
                     waveform changes sign.  Speech typically 0.02–0.10 crossings/sample;
                     fricatives and noise are higher.

spectral_centroid  : Weighted mean frequency (Hz).  Low centroid → mumbling / low-freq
                     noise.  Very high centroid → hiss / high-freq noise.

spectral_rolloff   : Frequency below which 85% of spectral energy is concentrated.
                     Useful for distinguishing speech from broadband noise.
"""

import logging
from typing import Dict

import numpy as np
import librosa

logger = logging.getLogger(__name__)


def compute_spectral_flatness(audio: np.ndarray, sr: int) -> float:
    """
    Spectral flatness (Wiener entropy) averaged across frames.
    Returns value in [0, 1].
    """
    n_fft = min(512, len(audio))
    if len(audio) < n_fft:
        return 1.0   # Too short — treat as noise

    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft)
    return float(np.mean(flatness))


def compute_zcr(audio: np.ndarray, sr: int) -> float:
    """
    Zero-crossing rate (mean crossings per sample, then normalised to per-second).
    """
    frame_length = int(0.025 * sr)
    hop_length   = int(0.010 * sr)

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    # zcr is in [0, 1] (crossings per sample); scale to crossings per second
    return float(np.mean(zcr) * sr)


def compute_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Spectral centroid in Hz."""
    n_fft = min(2048, len(audio))
    if len(audio) < n_fft:
        return 0.0

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft)
    return float(np.mean(centroid))


def compute_spectral_rolloff(audio: np.ndarray, sr: int,
                              roll_percent: float = 0.85) -> float:
    """
    Frequency (Hz) below which `roll_percent` of total spectral energy falls.
    """
    n_fft = min(2048, len(audio))
    if len(audio) < n_fft:
        return 0.0

    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=n_fft, roll_percent=roll_percent
    )
    return float(np.mean(rolloff))


def compute_all_spectral_metrics(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute all spectral metrics in a single call.

    Parameters
    ----------
    audio : float32 numpy array (pre-processed)
    sr    : sample rate

    Returns
    -------
    dict with keys: spectral_flatness, zcr, spectral_centroid, spectral_rolloff
    """
    return {
        "spectral_flatness":  compute_spectral_flatness(audio, sr),
        "zcr":                compute_zcr(audio, sr),
        "spectral_centroid":  compute_spectral_centroid(audio, sr),
        "spectral_rolloff":   compute_spectral_rolloff(audio, sr),
    }
