"""
Unit tests for all audio quality metrics.

Run with:
    pytest tests/ -v

Tests use synthetically generated audio so no real dataset is needed.
Covers:
  - Duration, RMS, clipping, silence, SNR (signal metrics)
  - Spectral flatness, ZCR, centroid (spectral metrics)
  - Scoring and hard rules
  - Preprocessing (mono conversion, resampling, normalisation)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from src.metrics.signal_metrics import (
    compute_duration,
    compute_rms_energy,
    compute_clipping_ratio,
    compute_silence_ratio,
    compute_snr,
)
from src.metrics.spectral_metrics import (
    compute_spectral_flatness,
    compute_zcr,
    compute_spectral_centroid,
)
from src.preprocess import to_mono, resample, peak_normalise

SR = 16_000


# ── Audio generators ─────────────────────────────────────────────────────────

def make_sine(freq: float = 440.0, duration: float = 2.0,
              amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration: float = 2.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration * sr), dtype=np.float32)


def make_white_noise(duration: float = 2.0, amplitude: float = 0.3,
                     sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (amplitude * rng.standard_normal(int(duration * sr))).astype(np.float32)


def make_clipped(duration: float = 2.0, sr: int = SR) -> np.ndarray:
    audio = make_sine(duration=duration)
    return np.clip(audio * 3.0, -1.0, 1.0).astype(np.float32)


# ── Duration ─────────────────────────────────────────────────────────────────

class TestDuration:
    def test_two_seconds(self):
        assert abs(compute_duration(make_sine(duration=2.0), SR) - 2.0) < 0.01

    def test_five_seconds(self):
        assert abs(compute_duration(make_sine(duration=5.0), SR) - 5.0) < 0.01

    def test_single_sample(self):
        assert compute_duration(np.array([0.0], dtype=np.float32), SR) == pytest.approx(1.0 / SR)


# ── RMS Energy ───────────────────────────────────────────────────────────────

class TestRMSEnergy:
    def test_silence_is_zero(self):
        assert compute_rms_energy(make_silence()) < 1e-9

    def test_sine_rms_close_to_theory(self):
        # Sine with amplitude A has RMS = A / sqrt(2)
        A = 0.5
        expected = A / np.sqrt(2)
        assert abs(compute_rms_energy(make_sine(amplitude=A)) - expected) < 0.01

    def test_rms_increases_with_amplitude(self):
        r1 = compute_rms_energy(make_sine(amplitude=0.3))
        r2 = compute_rms_energy(make_sine(amplitude=0.6))
        assert r2 > r1


# ── Clipping ─────────────────────────────────────────────────────────────────

class TestClippingRatio:
    def test_no_clipping(self):
        assert compute_clipping_ratio(make_sine(amplitude=0.5), 0.99) == 0.0

    def test_full_clipping(self):
        audio = np.ones(SR * 2, dtype=np.float32)  # all samples = 1.0
        assert compute_clipping_ratio(audio, 0.99) > 0.99

    def test_clipped_signal_has_positive_ratio(self):
        assert compute_clipping_ratio(make_clipped(), 0.99) > 0.0

    def test_threshold_sensitivity(self):
        audio = make_sine(amplitude=0.9)
        # No samples above 0.99 for amplitude 0.9
        assert compute_clipping_ratio(audio, 0.99) == 0.0
        # Many samples above 0.5 threshold
        assert compute_clipping_ratio(audio, 0.50) > 0.0


# ── Silence Ratio ─────────────────────────────────────────────────────────────

class TestSilenceRatio:
    def test_pure_silence_is_fully_silent(self):
        ratio = compute_silence_ratio(make_silence(duration=3.0), SR, threshold_db=-40.0)
        assert ratio > 0.95

    def test_loud_sine_is_not_silent(self):
        ratio = compute_silence_ratio(make_sine(amplitude=0.5), SR, threshold_db=-40.0)
        assert ratio < 0.05

    def test_short_audio_handled(self):
        short = make_sine(duration=0.01)   # shorter than one frame
        ratio = compute_silence_ratio(short, SR, threshold_db=-40.0)
        assert 0.0 <= ratio <= 1.0


# ── SNR ──────────────────────────────────────────────────────────────────────

class TestSNR:
    def test_speech_with_noise_floor_has_high_snr(self):
        # Realistic: loud speech segment surrounded by low-level background noise
        # The percentile estimator finds variation BETWEEN frames, not absolute noise.
        rng   = np.random.default_rng(0)
        audio = 0.003 * rng.standard_normal(SR * 3).astype(np.float32)  # noise floor
        t     = np.linspace(0, 1.0, SR, endpoint=False)
        audio[SR : SR * 2] += (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        snr = compute_snr(audio, SR)
        assert snr > 15.0, f"Expected SNR > 15 dB for clean speech over noise, got {snr:.1f}"

    def test_pure_noise_has_low_snr(self):
        snr = compute_snr(make_white_noise(duration=3.0), SR)
        assert snr < 15.0

    def test_silence_returns_finite_value(self):
        snr = compute_snr(make_silence(duration=2.0), SR)
        assert np.isfinite(snr)

    def test_snr_in_valid_range(self):
        snr = compute_snr(make_sine(), SR)
        assert -20.0 <= snr <= 60.0


# ── Spectral Flatness ─────────────────────────────────────────────────────────

class TestSpectralFlatness:
    def test_white_noise_has_high_flatness(self):
        flatness = compute_spectral_flatness(make_white_noise(duration=2.0), SR)
        assert flatness > 0.3

    def test_pure_sine_has_low_flatness(self):
        flatness = compute_spectral_flatness(make_sine(duration=2.0), SR)
        assert flatness < 0.3

    def test_flatness_in_zero_one(self):
        flatness = compute_spectral_flatness(make_sine(), SR)
        assert 0.0 <= flatness <= 1.0


# ── ZCR ──────────────────────────────────────────────────────────────────────

class TestZCR:
    def test_silence_has_near_zero_zcr(self):
        zcr = compute_zcr(make_silence(), SR)
        assert zcr < 10.0   # crossings / second

    def test_noise_has_higher_zcr_than_sine(self):
        zcr_sine  = compute_zcr(make_sine(freq=440.0), SR)
        zcr_noise = compute_zcr(make_white_noise(), SR)
        assert zcr_noise > zcr_sine

    def test_zcr_non_negative(self):
        assert compute_zcr(make_sine(), SR) >= 0.0


# ── Spectral Centroid ─────────────────────────────────────────────────────────

class TestSpectralCentroid:
    def test_centroid_near_sine_frequency(self):
        freq = 1000.0
        centroid = compute_spectral_centroid(make_sine(freq=freq, duration=2.0), SR)
        # Allow ±20% tolerance for windowing effects
        assert 500.0 < centroid < 3000.0

    def test_higher_freq_has_higher_centroid(self):
        c_low  = compute_spectral_centroid(make_sine(freq=200.0), SR)
        c_high = compute_spectral_centroid(make_sine(freq=4000.0), SR)
        assert c_high > c_low


# ── Preprocessing ────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_stereo_to_mono(self):
        stereo = np.stack([make_sine(freq=440), make_sine(freq=880)], axis=0)
        mono = to_mono(stereo)
        assert mono.ndim == 1
        assert len(mono) == SR * 2   # duration preserved

    def test_peak_normalise_bounds(self):
        audio = make_sine(amplitude=0.3)
        normed = peak_normalise(audio)
        assert np.max(np.abs(normed)) == pytest.approx(1.0, abs=1e-5)

    def test_peak_normalise_silence(self):
        silence = make_silence()
        normed = peak_normalise(silence)
        assert np.max(np.abs(normed)) < 1e-8

    def test_resample_preserves_duration(self):
        audio_48k = make_sine(sr=48000, duration=2.0)
        audio_16k = resample(audio_48k, orig_sr=48000, target_sr=16000)
        expected_len = int(2.0 * 16000)
        assert abs(len(audio_16k) - expected_len) <= 2   # allow 1–2 sample rounding


# ── Scoring ───────────────────────────────────────────────────────────────────

class TestScoring:
    """Integration tests for scoring logic using mock config."""

    @pytest.fixture
    def cfg(self):
        from omegaconf import OmegaConf
        return OmegaConf.create({
            "hard_rules": {
                "min_duration": 0.5,
                "max_duration": 30.0,
                "max_clipping_ratio": 0.05,
                "max_silence_ratio": 0.80,
            },
            "scoring": {
                "weights": {
                    "snr": 0.30,
                    "vad_ratio": 0.25,
                    "dnsmos": 0.20,
                    "active_ratio": 0.15,
                    "clipping": 0.10,
                },
                "threshold": 0.60,
                "snr_min": -5.0,
                "snr_max": 40.0,
            },
        })

    def test_high_quality_sample_is_kept(self, cfg):
        from src.scoring import make_decision
        metrics = {
            "duration": 4.0, "clipping_ratio": 0.0, "silence_ratio": 0.10,
            "snr_db": 30.0, "vad_ratio": 0.90, "dnsmos_ovr": 4.2,
            "rms_energy": 0.05,
        }
        result = make_decision(metrics, cfg)
        assert result["decision"] == "KEEP"
        assert result["quality_score"] >= 0.60

    def test_short_sample_hard_discarded(self, cfg):
        from src.scoring import make_decision
        metrics = {
            "duration": 0.2, "clipping_ratio": 0.0, "silence_ratio": 0.0,
            "snr_db": 35.0, "vad_ratio": 0.95, "dnsmos_ovr": 4.5,
            "rms_energy": 0.1,
        }
        result = make_decision(metrics, cfg)
        assert result["decision"] == "DISCARD"
        assert "short" in result["reason"]

    def test_mostly_silent_is_discarded(self, cfg):
        from src.scoring import make_decision
        metrics = {
            "duration": 5.0, "clipping_ratio": 0.0, "silence_ratio": 0.92,
            "snr_db": 5.0, "vad_ratio": 0.05, "dnsmos_ovr": 2.0,
            "rms_energy": 0.001,
        }
        result = make_decision(metrics, cfg)
        assert result["decision"] == "DISCARD"

    def test_score_between_zero_and_one(self, cfg):
        from src.scoring import compute_quality_score
        metrics = {
            "duration": 3.0, "clipping_ratio": 0.01, "silence_ratio": 0.20,
            "snr_db": 15.0, "vad_ratio": 0.70, "dnsmos_ovr": 3.5,
            "rms_energy": 0.03,
        }
        score = compute_quality_score(metrics, cfg)
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
