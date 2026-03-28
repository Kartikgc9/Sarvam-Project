"""
Voice Activity Detection (VAD) metric using Silero-VAD.

Silero-VAD is a lightweight (~1 MB) LSTM-based model that predicts
per-frame speech probabilities.  It runs entirely on CPU in milliseconds.

Why VAD matters
---------------
SNR alone cannot distinguish between:
  - A noisy but speech-rich file (high SNR, should KEEP)
  - A clean but content-free file (high SNR from music, should DISCARD)

VAD ratio = fraction of audio duration containing speech frames.
A low VAD ratio (<0.3) flags recordings that are mostly silence, music,
room tone, or non-speech noise.

Model loading
-------------
The model is loaded once per process (singleton pattern).
In multiprocessing scenarios, pass the pre-loaded model explicitly
to avoid reloading it in every worker call.
"""

import logging
from typing import Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton — loaded on first call
_vad_model: Optional[Any] = None
_vad_utils: Optional[Any] = None


def load_vad_model(cache_dir: str = "models/") -> Tuple[Any, Any]:
    """
    Load Silero-VAD model via torch.hub (cached after first call).

    Returns (model, utils) tuple where utils[0] = get_speech_timestamps.
    Returns (None, None) if torch is not available or download fails.
    """
    global _vad_model, _vad_utils

    if _vad_model is not None:
        return _vad_model, _vad_utils

    try:
        import torch

        # Point torch.hub cache to our models/ directory
        import os
        os.makedirs(cache_dir, exist_ok=True)
        torch.hub.set_dir(cache_dir)

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        _vad_model = model
        _vad_utils = utils
        logger.info("Silero-VAD loaded successfully.")
    except TypeError:
        # Older torch versions don't support trust_repo
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                verbose=False,
            )
            _vad_model = model
            _vad_utils = utils
            logger.info("Silero-VAD loaded (legacy API).")
        except Exception as e:
            logger.warning(f"Silero-VAD load failed: {e}")
    except Exception as e:
        logger.warning(f"Silero-VAD load failed: {e}")

    return _vad_model, _vad_utils


def compute_vad_ratio(
    audio: np.ndarray,
    sr: int,
    config,
    vad_model: Optional[Any] = None,
    vad_utils: Optional[Any] = None,
) -> float:
    """
    Compute the fraction of the audio duration classified as speech by Silero-VAD.

    Parameters
    ----------
    audio     : float32 numpy array at 16 kHz (after preprocessing)
    sr        : sample rate (must be 16 000)
    config    : OmegaConf config
    vad_model : pre-loaded Silero-VAD model (optional — loads if None)
    vad_utils : pre-loaded utils tuple (optional — loads if None)

    Returns
    -------
    float in [0.0, 1.0]  — 0 = no speech, 1 = all speech
    """
    if vad_model is None or vad_utils is None:
        vad_model, vad_utils = load_vad_model(config.models.vad_cache_dir)

    # Graceful fallback if model could not be loaded
    if vad_model is None:
        logger.debug("VAD model unavailable — returning neutral fallback 0.5")
        return 0.5

    try:
        import torch

        get_speech_timestamps = vad_utils[0]
        audio_tensor = torch.FloatTensor(audio)

        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=sr,
            threshold=float(config.metrics.vad.threshold),
            min_speech_duration_ms=int(config.metrics.vad.min_speech_duration_ms),
            min_silence_duration_ms=int(config.metrics.vad.min_silence_duration_ms),
            return_seconds=False,
        )

        if not speech_timestamps:
            return 0.0

        total_speech = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
        return float(np.clip(total_speech / max(len(audio), 1), 0.0, 1.0))

    except Exception as e:
        logger.warning(f"VAD inference error: {e}")
        return 0.5   # neutral fallback — don't penalise for model errors
