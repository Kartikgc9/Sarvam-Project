"""
Signal-level audio quality metrics.

Metrics
-------
duration        : Length of recording in seconds.
rms_energy      : Root-mean-square energy — measures average loudness.
clipping_ratio  : Fraction of samples at or near ±1 (overdriven mic / ADC saturation).
silence_ratio   : Fraction of short-time frames below a dB threshold.
snr_db          : Estimated signal-to-noise ratio in decibels.
                  Uses a noise-floor estimation approach:
                    - Frame the signal into 25 ms windows.
                    - Sort frames by energy.
                    - Bottom `noise_percentile`% → noise floor.
                    - Top (100 - noise_percentile)% avg → signal level.
                    - SNR = 10 * log10(E_signal / E_noise).
"""

import logging
from typing import Dict

import numpy as np
import librosa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_duration(audio: np.ndarray, sr: int) -> float:
    """Duration in seconds."""
    return float(len(audio) / sr)


def compute_rms_energy(audio: np.ndarray) -> float:
    """Root Mean Square energy of the full waveform."""
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def compute_clipping_ratio(audio: np.ndarray, threshold: float = 0.99) -> float:
    """
    Fraction of samples with |amplitude| >= threshold.
    Values close to 1.0 indicate saturation/clipping distortion.
    """
    return float(np.sum(np.abs(audio) >= threshold) / max(len(audio), 1))


def compute_silence_ratio(
    audio: np.ndarray, sr: int, threshold_db: float = -40.0
) -> float:
    """
    Fraction of short-time frames whose energy is below threshold_db.

    A high silence ratio means the recording is mostly padding or dead air,
    not actual speech content.
    """
    frame_length = int(0.025 * sr)   # 25 ms
    hop_length   = int(0.010 * sr)   # 10 ms

    if len(audio) < frame_length:
        # Audio shorter than one frame — treat as silent
        return 1.0

    frames = librosa.util.frame(
        audio, frame_length=frame_length, hop_length=hop_length, axis=0
    )  # shape: (n_frames, frame_length)

    frame_power = np.mean(frames.astype(np.float64) ** 2, axis=1)
    frame_db    = 10.0 * np.log10(frame_power + 1e-10)

    n_silent = int(np.sum(frame_db < threshold_db))
    return float(n_silent / max(len(frame_db), 1))


def compute_snr(
    audio: np.ndarray, sr: int, noise_percentile: int = 10
) -> float:
    """
    Estimate SNR (dB) via noise-floor percentile method.

    Algorithm:
      1. Frame signal into 25 ms non-overlapping windows.
      2. Compute per-frame energy.
      3. Bottom `noise_percentile`% of frames by energy → noise floor.
      4. Top `noise_percentile`% of frames by energy → signal level.
      5. SNR = 10 * log10(E_signal / E_noise).

    Returns values clipped to [-20, 60] dB.
    """
    frame_length = int(0.025 * sr)
    hop_length   = int(0.010 * sr)

    if len(audio) < frame_length:
        return 0.0

    frames = librosa.util.frame(
        audio, frame_length=frame_length, hop_length=hop_length, axis=0
    )  # (n_frames, frame_length)

    frame_energies = np.mean(frames.astype(np.float64) ** 2, axis=1)

    if len(frame_energies) < 4:
        return 0.0

    noise_thresh  = np.percentile(frame_energies, noise_percentile)
    signal_thresh = np.percentile(frame_energies, 100 - noise_percentile)

    noise_frames  = frame_energies[frame_energies <= noise_thresh]
    signal_frames = frame_energies[frame_energies >= signal_thresh]

    noise_energy  = float(np.mean(noise_frames))
    signal_energy = float(np.mean(signal_frames))

    if noise_energy <= 1e-12:
        return 40.0   # Essentially silent noise floor → very clean

    snr = 10.0 * np.log10(signal_energy / noise_energy)
    return float(np.clip(snr, -20.0, 60.0))


# ---------------------------------------------------------------------------
# Aggregate function called by the pipeline
# ---------------------------------------------------------------------------

def compute_all_signal_metrics(
    audio: np.ndarray, sr: int, config
) -> Dict[str, float]:
    """
    Compute all signal-level metrics in a single call.

    Parameters
    ----------
    audio  : float32 numpy array, pre-processed (mono, 16 kHz, peak-normalised)
    sr     : sample rate
    config : OmegaConf config object (or any object with .metrics.*)

    Returns
    -------
    dict with keys: duration, rms_energy, clipping_ratio, silence_ratio, snr_db
    """
    return {
        "duration":       compute_duration(audio, sr),
        "rms_energy":     compute_rms_energy(audio),
        "clipping_ratio": compute_clipping_ratio(
            audio, config.metrics.clipping.threshold
        ),
        "silence_ratio":  compute_silence_ratio(
            audio, sr, config.metrics.silence.threshold_db
        ),
        "snr_db":         compute_snr(
            audio, sr, config.metrics.snr.noise_percentile
        ),
    }
