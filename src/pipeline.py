"""
Audio Filtering Pipeline — main orchestrator.

Architecture
------------
1.  IndicVoicesLoader streams samples one language at a time from HuggingFace.
    No full dataset is ever held in RAM.

2.  Samples are collected into batches (default 32).

3.  For each batch:
    a. Preprocessing (resample, mono, normalise) — sequential, fast.
    b. CPU metrics (SNR, RMS, clipping, silence, spectral) — parallel via
       ProcessPoolExecutor.  Each worker is independent (pure numpy/librosa),
       so no pickling issues.
    c. Neural metrics (Silero-VAD, DNSMOS) — sequential in main process using
       pre-loaded models.  Avoids re-loading large models in every worker.
    d. Scoring + hard-rules decision.
    e. Incremental CSV write + checkpoint update.

4.  Crash recovery: the checkpoint file records processed sample IDs so that
    a re-run resumes from where it left off.

Parallelism note
----------------
ProcessPoolExecutor on Windows uses "spawn" (not "fork"), so worker code must
be importable at module level.  `_compute_cpu_metrics` is a top-level function
satisfying this requirement.
"""

import csv
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from .loader import IndicVoicesLoader
from .preprocess import preprocess_audio
from .metrics.signal_metrics import compute_all_signal_metrics
from .metrics.spectral_metrics import compute_all_spectral_metrics
from .metrics.vad_metrics import load_vad_model, compute_vad_ratio
from .metrics.dnsmos import load_dnsmos_model, compute_dnsmos
from .scoring import make_decision

logger = logging.getLogger(__name__)

# CSV column order (matches README output spec)
CSV_COLUMNS = [
    "id", "language", "duration_s", "rms_energy",
    "snr_db", "silence_ratio", "clipping_ratio",
    "spectral_flatness", "zcr", "spectral_centroid", "spectral_rolloff",
    "vad_ratio",
    "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr",
    "quality_score", "decision", "reason",
]


# ---------------------------------------------------------------------------
# Worker function — must be at module level for ProcessPoolExecutor (Windows)
# ---------------------------------------------------------------------------

def _compute_cpu_metrics(
    sample: Dict[str, Any], config_dict: dict
) -> Dict[str, Any]:
    """
    Compute all CPU-bound metrics for one sample.
    Runs inside a worker process — receives and returns plain dicts.
    audio_array is NOT returned (stripped before pickling back) to keep
    inter-process data transfer minimal.
    """
    from omegaconf import OmegaConf
    from src.metrics.signal_metrics import compute_all_signal_metrics
    from src.metrics.spectral_metrics import compute_all_spectral_metrics

    cfg = OmegaConf.create(config_dict)
    audio = np.array(sample["audio_array"], dtype=np.float32)
    sr = int(sample["sample_rate"])

    signal   = compute_all_signal_metrics(audio, sr, cfg)
    spectral = compute_all_spectral_metrics(audio, sr)

    # Build result without audio_array to reduce pickle size
    result = {k: v for k, v in sample.items() if k != "audio_array"}
    result.update(signal)
    result.update(spectral)
    return result


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class AudioFilteringPipeline:

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = OmegaConf.load(config_path)
        self._config_dict: dict = OmegaConf.to_container(
            self.config, resolve=True
        )   # plain dict — safe to pickle to workers

        self._setup_output_dirs()
        self._init_csv()

        logger.info("Loading VAD model …")
        self.vad_model, self.vad_utils = load_vad_model(
            str(self.config.models.vad_cache_dir)
        )

        logger.info("Loading DNSMOS model …")
        self.dnsmos_session = load_dnsmos_model(
            str(self.config.models.dnsmos_path)
        )

        self.loader = IndicVoicesLoader(self.config)
        self.processed_ids = self._load_checkpoint()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_output_dirs(self):
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("models",  exist_ok=True)

    def _init_csv(self):
        """Write CSV header if file doesn't already exist."""
        path = str(self.config.pipeline.output_csv)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def _load_checkpoint(self) -> set:
        path = str(self.config.pipeline.checkpoint_file)
        if os.path.exists(path):
            with open(path, "r") as f:
                ids = {line.strip() for line in f if line.strip()}
            logger.info(f"Checkpoint: {len(ids)} samples already processed — resuming.")
            return ids
        return set()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _write_result(self, result: Dict[str, Any]):
        """Append one result row to the CSV."""
        row = {
            "id":                result.get("id", ""),
            "language":          result.get("language", ""),
            "duration_s":        round(float(result.get("duration", 0)), 3),
            "rms_energy":        round(float(result.get("rms_energy", 0)), 6),
            "snr_db":            round(float(result.get("snr_db", 0)), 2),
            "silence_ratio":     round(float(result.get("silence_ratio", 0)), 4),
            "clipping_ratio":    round(float(result.get("clipping_ratio", 0)), 6),
            "spectral_flatness": round(float(result.get("spectral_flatness", 0)), 4),
            "zcr":               round(float(result.get("zcr", 0)), 2),
            "spectral_centroid": round(float(result.get("spectral_centroid", 0)), 2),
            "spectral_rolloff":  round(float(result.get("spectral_rolloff", 0)), 2),
            "vad_ratio":         round(float(result.get("vad_ratio", 0)), 4),
            "dnsmos_sig":        round(float(result.get("dnsmos_sig", 0)), 3),
            "dnsmos_bak":        round(float(result.get("dnsmos_bak", 0)), 3),
            "dnsmos_ovr":        round(float(result.get("dnsmos_ovr", 0)), 3),
            "quality_score":     round(float(result.get("quality_score", 0)), 4),
            "decision":          result.get("decision", ""),
            "reason":            result.get("reason", ""),
        }
        with open(
            str(self.config.pipeline.output_csv), "a", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

    def _save_checkpoint(self, sample_id: str):
        with open(str(self.config.pipeline.checkpoint_file), "a") as f:
            f.write(f"{sample_id}\n")

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _preprocess_batch(
        self, batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Preprocess all samples; skip any that raise errors."""
        processed = []
        for s in batch:
            try:
                processed.append(
                    preprocess_audio(s, int(self.config.audio.sample_rate))
                )
            except Exception as e:
                logger.warning(f"Preprocessing failed [{s.get('id')}]: {e}")
        return processed

    def _parallel_cpu_metrics(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run CPU metrics in parallel.  Returns dict keyed by sample ID.
        """
        n_workers = int(self.config.pipeline.num_workers)
        results: Dict[str, Dict[str, Any]] = {}

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_id = {
                executor.submit(_compute_cpu_metrics, s, self._config_dict): s["id"]
                for s in samples
            }
            for future in as_completed(future_to_id):
                sid = future_to_id[future]
                try:
                    results[sid] = future.result()
                except Exception as e:
                    logger.warning(f"CPU metrics failed [{sid}]: {e}")

        return results

    def _neural_metrics(
        self,
        cpu_result: Dict[str, Any],
        audio: np.ndarray,
        sr: int,
    ) -> Dict[str, Any]:
        """Add VAD and DNSMOS scores to a cpu_result dict."""
        cpu_result["vad_ratio"] = compute_vad_ratio(
            audio, sr, self.config, self.vad_model, self.vad_utils
        )
        dnsmos = compute_dnsmos(
            audio, sr, self.dnsmos_session,
            str(self.config.models.dnsmos_path),
        )
        cpu_result.update(dnsmos)
        return cpu_result

    def process_batch(
        self, batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Full processing pipeline for one batch of samples.
        Returns list of result dicts (each includes metrics + decision).
        """
        # 1. Preprocess
        preprocessed = self._preprocess_batch(batch)
        if not preprocessed:
            return []

        # Build lookup by ID for later re-joining with audio arrays
        audio_by_id = {s["id"]: (s["audio_array"], s["sample_rate"])
                       for s in preprocessed}

        # 2. Parallel CPU metrics (audio_array sent to workers, not returned)
        cpu_results = self._parallel_cpu_metrics(preprocessed)

        # 3. Neural metrics + scoring (sequential, main process)
        final_results = []
        for sid, cpu_result in cpu_results.items():
            audio, sr = audio_by_id[sid]
            try:
                result = self._neural_metrics(cpu_result, audio, sr)
                decision = make_decision(result, self.config)
                result.update(decision)
                final_results.append(result)
            except Exception as e:
                logger.warning(f"Scoring failed [{sid}]: {e}")

        return final_results

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Run the full pipeline over all configured languages and samples.

        Returns a summary dict with counts and elapsed time.
        """
        logger.info("=" * 60)
        logger.info("  Audio Filtering Pipeline — Starting")
        logger.info("=" * 60)

        t0 = time.time()
        total = kept = discarded = 0

        batch_size = int(self.config.pipeline.batch_size)

        with tqdm(desc="Filtering audio samples", unit="sample") as pbar:
            for batch in self.loader.iter_batches(
                batch_size, skip_ids=self.processed_ids
            ):
                results = self.process_batch(batch)

                for r in results:
                    self._write_result(r)
                    self._save_checkpoint(r["id"])
                    self.processed_ids.add(r["id"])

                    if r["decision"] == "KEEP":
                        kept += 1
                    else:
                        discarded += 1
                    total += 1

                pbar.update(len(results))
                keep_rate = kept / total * 100 if total else 0
                pbar.set_postfix(
                    kept=kept,
                    discarded=discarded,
                    keep_pct=f"{keep_rate:.1f}%",
                )

        elapsed = time.time() - t0
        summary = {
            "total": total,
            "kept": kept,
            "discarded": discarded,
            "keep_rate_pct": round(kept / max(total, 1) * 100, 1),
            "elapsed_seconds": round(elapsed, 1),
            "output_csv": str(self.config.pipeline.output_csv),
        }

        logger.info("=" * 60)
        logger.info(f"  Done in {elapsed:.1f}s")
        logger.info(f"  Total: {total}  |  Kept: {kept}  |  Discarded: {discarded}")
        logger.info(f"  Keep rate: {summary['keep_rate_pct']}%")
        logger.info(f"  Results: {summary['output_csv']}")
        logger.info("=" * 60)

        return summary
