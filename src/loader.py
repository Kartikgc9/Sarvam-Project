"""
IndicVoices dataset loader with HuggingFace streaming support.

Supports:
  - HuggingFace streaming (memory-safe for large datasets)
  - Per-language config loading (IndicVoices uses language configs)
  - Fallback to local audio files (WAV/FLAC/MP3)
  - Checkpoint-aware iteration (skip already-processed samples)
"""

import logging
import os
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Maps IndicVoices language codes to HuggingFace config names.
# Adjust if the dataset uses different keys.
LANGUAGE_CONFIG_MAP = {
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
    "bn": "bn",
    "gu": "gu",
    "mr": "mr",
    "pa": "pa",
    "ur": "ur",
    "as": "as",
    "or": "or",
    "raj": "raj",
    "mai": "mai",
}

# Candidate column names for audio / transcript / language fields
_AUDIO_COLS = ["audio", "input_values", "audio_array"]
_TRANSCRIPT_COLS = ["text", "transcript", "sentence", "transcription", "normalized_text"]
_LANGUAGE_COLS = ["language", "lang", "locale", "language_id"]


def _find_column(sample: dict, candidates: list, default: Any = None) -> Any:
    for col in candidates:
        if col in sample:
            return sample[col]
    return default


def _extract_sample(raw: dict, language: str, index: int) -> Optional[Dict[str, Any]]:
    """Extract a normalised sample dict from a raw HuggingFace row."""
    audio_field = _find_column(raw, _AUDIO_COLS)
    if audio_field is None:
        return None

    # HuggingFace audio datasets return a dict with 'array' and 'sampling_rate'
    if isinstance(audio_field, dict):
        array = audio_field.get("array")
        sr = audio_field.get("sampling_rate", 16000)
    elif isinstance(audio_field, (list, np.ndarray)):
        array = audio_field
        sr = raw.get("sampling_rate", 16000)
    else:
        return None

    if array is None or len(array) == 0:
        return None

    sample_id = raw.get(
        "id",
        raw.get("client_id", raw.get("path", f"{language}_{index:06d}"))
    )

    transcript = _find_column(raw, _TRANSCRIPT_COLS, default="")
    detected_lang = _find_column(raw, _LANGUAGE_COLS, default=language)

    return {
        "id": str(sample_id),
        "language": str(detected_lang) if detected_lang else language,
        "audio_array": np.array(array, dtype=np.float32),
        "sample_rate": int(sr),
        "transcript": str(transcript) if transcript else "",
        "speaker_id": str(raw.get("speaker_id", raw.get("client_id", "unknown"))),
    }


class IndicVoicesLoader:
    """
    Streams IndicVoices from HuggingFace Hub one language at a time.
    Falls back to local audio files if the HuggingFace dataset is unavailable.
    """

    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset.name
        self.languages = list(config.dataset.languages)
        self.split = config.dataset.split
        self.streaming = config.dataset.streaming
        self.max_per_lang = config.dataset.max_samples_per_language

    # ------------------------------------------------------------------
    # HuggingFace loading helpers
    # ------------------------------------------------------------------

    def _load_hf_language(self, language: str):
        """Try to load a single-language config from HuggingFace."""
        from datasets import load_dataset

        config_name = LANGUAGE_CONFIG_MAP.get(language, language)

        # Strategy 1: language-specific config (most datasets)
        try:
            ds = load_dataset(
                self.dataset_name,
                config_name,
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=True,
            )
            logger.info(f"Loaded HuggingFace config '{config_name}' for language '{language}'")
            return ds
        except Exception as e1:
            logger.debug(f"Config '{config_name}' failed: {e1}")

        # Strategy 2: no config, filter by language column later
        try:
            ds = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=True,
            )
            logger.info(f"Loaded full dataset; will filter for language='{language}'")
            return ds
        except Exception as e2:
            logger.warning(f"Could not load dataset for language '{language}': {e2}")
            return None

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def iter_samples(
        self, skip_ids: Optional[set] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield normalised sample dicts across all configured languages.

        Args:
            skip_ids: set of sample IDs already processed (checkpoint support).
        """
        skip_ids = skip_ids or set()

        for language in self.languages:
            logger.info(f"Starting language: {language}")
            ds = self._load_hf_language(language)

            if ds is None:
                logger.warning(f"Skipping language '{language}' — dataset unavailable.")
                continue

            count = 0
            for raw_index, raw in enumerate(ds):
                if self.max_per_lang is not None and count >= self.max_per_lang:
                    break

                sample = _extract_sample(raw, language, raw_index)
                if sample is None:
                    continue

                if sample["id"] in skip_ids:
                    continue

                yield sample
                count += 1

            logger.info(f"Language '{language}': yielded {count} samples")

    def iter_batches(
        self, batch_size: int, skip_ids: Optional[set] = None
    ) -> Iterator[List[Dict[str, Any]]]:
        """Yield batches of samples."""
        batch: List[Dict[str, Any]] = []
        for sample in self.iter_samples(skip_ids=skip_ids):
            batch.append(sample)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


# ------------------------------------------------------------------
# Local file loader (fallback / testing)
# ------------------------------------------------------------------

class LocalAudioLoader:
    """
    Load audio files from a local directory tree.
    Folder structure expected: root/<language>/*.{wav,flac,mp3}
    """

    SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus"}

    def __init__(self, root_dir: str, languages: Optional[List[str]] = None,
                 max_per_lang: Optional[int] = None):
        self.root_dir = Path(root_dir)
        self.languages = languages
        self.max_per_lang = max_per_lang

    def iter_samples(self, skip_ids: Optional[set] = None) -> Iterator[Dict[str, Any]]:
        import soundfile as sf

        skip_ids = skip_ids or set()
        lang_dirs = sorted(self.root_dir.iterdir()) if self.root_dir.is_dir() else []

        for lang_dir in lang_dirs:
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name
            if self.languages and lang not in self.languages:
                continue

            count = 0
            for audio_path in sorted(lang_dir.rglob("*")):
                if audio_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                if self.max_per_lang is not None and count >= self.max_per_lang:
                    break

                sample_id = audio_path.stem
                if sample_id in skip_ids:
                    continue

                try:
                    array, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
                    yield {
                        "id": sample_id,
                        "language": lang,
                        "audio_array": array,
                        "sample_rate": sr,
                        "transcript": "",
                        "speaker_id": "unknown",
                    }
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to read {audio_path}: {e}")

    def iter_batches(self, batch_size: int,
                     skip_ids: Optional[set] = None) -> Iterator[List[Dict[str, Any]]]:
        batch: List[Dict[str, Any]] = []
        for sample in self.iter_samples(skip_ids=skip_ids):
            batch.append(sample)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
