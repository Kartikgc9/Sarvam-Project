"""
Audio preprocessing: resample → mono → peak-normalise.

All downstream metrics assume:
  - sample_rate = 16 000 Hz  (standard for speech models)
  - mono channel
  - float32 values in [-1.0, 1.0]
"""

import logging
from typing import Dict, Any

import numpy as np
import librosa

logger = logging.getLogger(__name__)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if audio.ndim == 1:
        return audio
    # Shape could be (samples, channels) or (channels, samples)
    if audio.shape[0] < audio.shape[-1]:
        # Likely (channels, samples) — channels axis is smaller
        return audio.mean(axis=0)
    return audio.mean(axis=-1)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate using librosa."""
    if orig_sr == target_sr:
        return audio
    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return resampled.astype(np.float32)


def peak_normalise(audio: np.ndarray) -> np.ndarray:
    """
    Peak-normalise audio to [-1, 1].
    If the signal is all-zero (pure silence), returns it unchanged.
    """
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        return (audio / peak).astype(np.float32)
    return audio.astype(np.float32)


def preprocess_audio(sample: Dict[str, Any], target_sr: int = 16000) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for a single audio sample.

    Steps:
      1. Convert to mono
      2. Resample to target_sr
      3. Peak-normalise

    Modifies sample in-place and adds 'duration' key.
    Returns the modified sample dict.
    """
    audio: np.ndarray = sample["audio_array"]
    sr: int = sample["sample_rate"]

    if audio is None or len(audio) == 0:
        raise ValueError(f"Empty audio for sample {sample.get('id', '?')}")

    # 1. Mono
    audio = to_mono(audio)

    # 2. Resample
    audio = resample(audio, orig_sr=sr, target_sr=target_sr)

    # 3. Normalise
    audio = peak_normalise(audio)

    sample["audio_array"] = audio
    sample["sample_rate"] = target_sr
    sample["duration"] = len(audio) / target_sr

    return sample
