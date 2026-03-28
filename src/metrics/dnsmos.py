"""
DNSMOS P.835 — Deep Noise Suppression Mean Opinion Score.

Reference
---------
Reddy et al., "DNSMOS P.835: A non-intrusive perceptual objective
speech quality metric to evaluate noise suppressors", ICASSP 2022.
Microsoft DNS Challenge: https://github.com/microsoft/DNS-Challenge

Why DNSMOS?
-----------
Traditional signal metrics (SNR, RMS) measure physical properties but do not
capture perceptual quality — i.e., whether a human would rate the audio as
natural and intelligible.  DNSMOS predicts three MOS subscores:
  - SIG : Speech signal quality (1–5)
  - BAK : Background noise intrusiveness (1–5)
  - OVR : Overall quality (1–5)

The model is a tiny ONNX network (~8 MB) that runs on CPU in ~5 ms per clip,
making it practical for large-scale filtering pipelines.

Setup
-----
Download the model once with:
    python setup_models.py
The ONNX file is placed at models/sig_bak_ovr.onnx.
"""

import logging
import os
from typing import Dict, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# DNSMOS constants (must match the model's training configuration)
_SAMPLE_RATE    = 16_000
_INPUT_LENGTH   = 9.01          # seconds — model fixed context window
_N_MELS         = 120
_FRAME_SIZE     = 320           # n_fft = FRAME_SIZE + 1
_HOP_LENGTH     = 160

# Polynomial calibration coefficients from the paper (non-personalised MOS)
_COEFF_OVR = [-0.06766283, 1.11546468,  0.04602535]
_COEFF_SIG = [-0.08397278, 1.22083953,  0.00524390]
_COEFF_BAK = [-0.13166888, 1.60915514, -0.39604546]

# Module-level singleton
_dnsmos_session: Optional[Any] = None


def load_dnsmos_model(model_path: str = "models/sig_bak_ovr.onnx") -> Optional[Any]:
    """
    Load the DNSMOS ONNX session (cached after first call).

    Returns the onnxruntime.InferenceSession, or None if the model file
    is missing or onnxruntime is unavailable.
    """
    global _dnsmos_session

    if _dnsmos_session is not None:
        return _dnsmos_session

    if not os.path.exists(model_path):
        logger.warning(
            f"DNSMOS model not found at '{model_path}'. "
            "Run `python setup_models.py` to download it."
        )
        return None

    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.log_severity_level   = 3   # suppress verbose ONNX logs

        _dnsmos_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info("DNSMOS model loaded successfully.")
    except Exception as e:
        logger.warning(f"Failed to load DNSMOS model: {e}")

    return _dnsmos_session


def _audio_melspec(audio: np.ndarray) -> np.ndarray:
    """
    Compute the normalised log-mel spectrogram used as DNSMOS input.
    Output shape: (time_frames, n_mels)
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=_SAMPLE_RATE,
        n_fft=_FRAME_SIZE + 1,
        hop_length=_HOP_LENGTH,
        n_mels=_N_MELS,
    )
    mel_db = (librosa.power_to_db(mel, ref=np.max) + 40.0) / 40.0
    return mel_db.T   # (time, n_mels)


def _prepare_audio(audio: np.ndarray) -> np.ndarray:
    """
    Pad or crop audio to exactly _INPUT_LENGTH seconds at _SAMPLE_RATE.
    Short clips are tiled (repeated) rather than zero-padded to avoid
    creating artificial silence boundaries.
    """
    target = int(_INPUT_LENGTH * _SAMPLE_RATE)

    if len(audio) == 0:
        return np.zeros(target, dtype=np.float32)

    if len(audio) < target:
        repeats = int(np.ceil(target / len(audio)))
        audio = np.tile(audio, repeats)

    return audio[:target].astype(np.float32)


def _polyfit_calibrate(sig: float, bak: float, ovr: float):
    """Apply polynomial calibration from the DNSMOS paper."""
    p_sig = float(np.poly1d(_COEFF_SIG)(sig))
    p_bak = float(np.poly1d(_COEFF_BAK)(bak))
    p_ovr = float(np.poly1d(_COEFF_OVR)(ovr))
    return p_sig, p_bak, p_ovr


def compute_dnsmos(
    audio: np.ndarray,
    sr: int = 16_000,
    session: Optional[Any] = None,
    model_path: str = "models/sig_bak_ovr.onnx",
) -> Dict[str, float]:
    """
    Compute DNSMOS P.835 perceptual quality scores.

    Parameters
    ----------
    audio      : float32 numpy array (pre-processed at 16 kHz)
    sr         : sample rate (must be 16 000)
    session    : pre-loaded onnxruntime.InferenceSession (optional)
    model_path : path to sig_bak_ovr.onnx (used only if session is None)

    Returns
    -------
    dict with keys: dnsmos_sig, dnsmos_bak, dnsmos_ovr  (each in [1.0, 5.0])
    """
    _FALLBACK = {"dnsmos_sig": 3.0, "dnsmos_bak": 3.0, "dnsmos_ovr": 3.0}

    if session is None:
        session = load_dnsmos_model(model_path)

    if session is None:
        logger.debug("DNSMOS model unavailable — returning neutral score 3.0")
        return _FALLBACK

    try:
        audio_in   = _prepare_audio(audio)
        mel        = _audio_melspec(audio_in)               # (T, 120)
        mel_input  = mel[np.newaxis, :, :].astype(np.float32)  # (1, T, 120)

        input_name = session.get_inputs()[0].name
        raw_out    = session.run(None, {input_name: mel_input})[0][0]
        # raw_out shape: (3,) → [sig_raw, bak_raw, ovr_raw]

        sig_raw, bak_raw, ovr_raw = float(raw_out[0]), float(raw_out[1]), float(raw_out[2])
        p_sig, p_bak, p_ovr = _polyfit_calibrate(sig_raw, bak_raw, ovr_raw)

        return {
            "dnsmos_sig": float(np.clip(p_sig, 1.0, 5.0)),
            "dnsmos_bak": float(np.clip(p_bak, 1.0, 5.0)),
            "dnsmos_ovr": float(np.clip(p_ovr, 1.0, 5.0)),
        }

    except Exception as e:
        logger.warning(f"DNSMOS inference error: {e}")
        return _FALLBACK
