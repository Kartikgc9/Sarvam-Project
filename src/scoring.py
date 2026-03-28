"""
Quality scoring and filtering decision logic.

Two-stage decision process
--------------------------
Stage 1 — Hard rules (binary, applied first):
  Any sample failing a hard rule is immediately discarded with a named reason.
  Hard rules catch extreme cases where scoring would be meaningless:
    • duration_too_short  : < 0.5 s  — no meaningful utterance
    • duration_too_long   : > 30 s   — likely a recording artefact
    • excessive_clipping  : > 5 %    — severe distortion
    • mostly_silent       : > 80 %   — not a real speech sample

Stage 2 — Weighted quality score:
  score = Σ weight_i × normalised_metric_i

  Weights (and justification):
    0.30 — SNR         : Noise is the biggest enemy of ASR training.
    0.25 — VAD ratio   : Actual speech must be present; silence or music hurt models.
    0.20 — DNSMOS OVR  : Perceptual MOS aligns best with downstream model quality.
    0.15 — Active ratio: Penalise recordings that are mostly silence but pass Stage 1.
    0.10 — No clipping : Signal integrity; even mild clipping degrades spectral features.

  Total weights sum to 1.0.
  score ∈ [0, 1].  Threshold = 0.60 → KEEP.
"""

from typing import Dict, Any, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_snr(snr_db: float, lo: float = -5.0, hi: float = 40.0) -> float:
    """Map SNR (dB) linearly from [lo, hi] → [0, 1]."""
    return float(np.clip((snr_db - lo) / (hi - lo), 0.0, 1.0))


def _norm_dnsmos(ovr: float) -> float:
    """Map DNSMOS OVR from [1, 5] → [0, 1]."""
    return float(np.clip((ovr - 1.0) / 4.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Stage 1 — Hard rules
# ---------------------------------------------------------------------------

def apply_hard_rules(
    metrics: Dict[str, float], config
) -> Tuple[bool, str]:
    """
    Check hard-discard rules.

    Returns
    -------
    (True, "ok")                        if all rules pass
    (False, "<reason>")                 if a rule is violated
    """
    r = config.hard_rules

    if metrics["duration"] < float(r.min_duration):
        return False, f"duration_too_short ({metrics['duration']:.2f}s)"

    if metrics["duration"] > float(r.max_duration):
        return False, f"duration_too_long ({metrics['duration']:.2f}s)"

    if metrics["clipping_ratio"] > float(r.max_clipping_ratio):
        return False, f"excessive_clipping ({metrics['clipping_ratio']:.4f})"

    if metrics["silence_ratio"] > float(r.max_silence_ratio):
        return False, f"mostly_silent ({metrics['silence_ratio']:.4f})"

    return True, "ok"


# ---------------------------------------------------------------------------
# Stage 2 — Weighted score
# ---------------------------------------------------------------------------

def compute_quality_score(
    metrics: Dict[str, float], config
) -> float:
    """
    Compute weighted quality score in [0, 1].

    Each component is independently normalised to [0, 1] before weighting
    so that metrics with different units are treated fairly.
    """
    w = config.scoring.weights

    snr_score     = _norm_snr(
        metrics.get("snr_db", 0.0),
        float(config.scoring.snr_min),
        float(config.scoring.snr_max),
    )
    vad_score     = float(np.clip(metrics.get("vad_ratio", 0.5), 0.0, 1.0))
    dnsmos_score  = _norm_dnsmos(metrics.get("dnsmos_ovr", 3.0))
    active_score  = float(np.clip(1.0 - metrics.get("silence_ratio", 0.0), 0.0, 1.0))
    # Clipping penalty: map clipping_ratio ∈ [0, max_clip] → [1, 0]
    max_clip      = float(config.hard_rules.max_clipping_ratio)
    clip_score    = float(np.clip(
        1.0 - metrics.get("clipping_ratio", 0.0) / max_clip, 0.0, 1.0
    ))

    score = (
        float(w.snr)         * snr_score
        + float(w.vad_ratio) * vad_score
        + float(w.dnsmos)    * dnsmos_score
        + float(w.active_ratio) * active_score
        + float(w.clipping)  * clip_score
    )

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Combined decision
# ---------------------------------------------------------------------------

def make_decision(
    metrics: Dict[str, float], config
) -> Dict[str, Any]:
    """
    Full two-stage filtering decision.

    Parameters
    ----------
    metrics : dict containing all computed metrics for one sample
    config  : OmegaConf config

    Returns
    -------
    dict with keys: quality_score (float), decision ("KEEP"/"DISCARD"), reason (str)
    """
    # Stage 1
    passed, reason = apply_hard_rules(metrics, config)
    if not passed:
        return {
            "quality_score": 0.0,
            "decision": "DISCARD",
            "reason": reason,
        }

    # Stage 2
    score = compute_quality_score(metrics, config)
    threshold = float(config.scoring.threshold)

    if score >= threshold:
        return {
            "quality_score": round(score, 4),
            "decision": "KEEP",
            "reason": "quality_score_pass",
        }
    else:
        return {
            "quality_score": round(score, 4),
            "decision": "DISCARD",
            "reason": f"low_score_{score:.3f}",
        }
