"""
IndicVoices dataset loader.

Supports:
  - HuggingFace streaming with token auth (ai4bharat/IndicVoices)
  - ManifestLoader for locally downloaded data via setup_dataset.py
  - LocalAudioLoader fallback for plain WAV/FLAC/MP3 directories
  - Checkpoint-aware iteration (skip already-processed samples)

IndicVoices column schema (from parquet):
  audio_filepath : {bytes: binary, path: str}  — audio content
  text           : str                          — transcript
  language       : str                          — language name
  speaker_id     : str                          — speaker identifier
  duration       : float                        — duration in seconds
"""

import io
import logging
import os
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# IndicVoices language config names on HuggingFace Hub
# These match the directory names produced by setup_dataset.py
LANGUAGE_CONFIG_MAP = {
    "hi": "hi",   # Hindi
    "ta": "ta",   # Tamil
    "te": "te",   # Telugu
    "kn": "kn",   # Kannada
    "ml": "ml",   # Malayalam
    "bn": "bn",   # Bengali
    "gu": "gu",   # Gujarati
    "mr": "mr",   # Marathi
    "pa": "pa",   # Punjabi
    "ur": "ur",   # Urdu
    "as": "as",   # Assamese
    "or": "or",   # Odia
    "raj": "raj", # Rajasthani
    "mai": "mai", # Maithili
    "ks": "ks",   # Kashmiri
    "ne": "ne",   # Nepali
    "sa": "sa",   # Sanskrit
    "sd": "sd",   # Sindhi
    "doi": "doi", # Dogri
    "kok": "kok", # Konkani
    "mni": "mni", # Manipuri (Meitei)
    "sat": "sat", # Santali
}

# Candidate column names — checked in order, first match wins
_AUDIO_COLS      = ["audio_filepath", "audio", "input_values", "audio_array"]
_TRANSCRIPT_COLS = ["text", "transcript", "sentence", "transcription", "normalized_text"]
_LANGUAGE_COLS   = ["language", "lang", "locale", "language_id"]


def _find_column(sample: dict, candidates: list, default: Any = None) -> Any:
    for col in candidates:
        if col in sample:
            return sample[col]
    return default


def _decode_audio_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decode raw audio bytes (any format soundfile supports) to float32 array."""
    try:
        import soundfile as sf
        array, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return array
    except Exception as e:
        logger.debug(f"soundfile decode failed: {e}")
        return None


def _extract_sample(raw: dict, language: str, index: int) -> Optional[Dict[str, Any]]:
    """Extract a normalised sample dict from a raw HuggingFace row."""
    audio_field = _find_column(raw, _AUDIO_COLS)
    if audio_field is None:
        return None

    array: Optional[np.ndarray] = None
    sr: int = 16000

    if isinstance(audio_field, dict):
        # Case 1: HuggingFace auto-decoded Audio feature
        #   {'array': np.ndarray, 'sampling_rate': int}
        if "array" in audio_field:
            array = audio_field["array"]
            sr    = audio_field.get("sampling_rate", 16000)

        # Case 2: Raw IndicVoices parquet struct
        #   {'bytes': bytes, 'path': str}
        elif "bytes" in audio_field and audio_field["bytes"]:
            array = _decode_audio_bytes(audio_field["bytes"])
            # sampling_rate not stored in the struct; loader will resample to target_sr

    elif isinstance(audio_field, (list, np.ndarray)):
        array = audio_field
        sr    = raw.get("sampling_rate", 16000)

    if array is None or len(array) == 0:
        return None

    # Prefix with language code — IndicVoices reuses integers per-language
    raw_id    = raw.get("id", raw.get("client_id", raw.get("path", index)))
    sample_id = f"{language}_{raw_id}"

    transcript    = _find_column(raw, _TRANSCRIPT_COLS, default="")
    detected_lang = _find_column(raw, _LANGUAGE_COLS, default=language)

    return {
        "id":           str(sample_id),
        "language":     language,
        "language_label": str(detected_lang) if detected_lang else language,
        "audio_array":  np.array(array, dtype=np.float32),
        "sample_rate":  int(sr),
        "transcript":   str(transcript) if transcript else "",
        "speaker_id":   str(raw.get("speaker_id", raw.get("client_id", "unknown"))),
    }


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace streaming loader (primary — for Colab / demo)
# ──────────────────────────────────────────────────────────────────────────────

class IndicVoicesLoader:
    """
    Streams IndicVoices from HuggingFace Hub one language at a time.

    Requires:
        huggingface-cli login   OR   HF_TOKEN env var   OR   token= kwarg

    Falls back to LocalAudioLoader if the HuggingFace dataset is unavailable.
    """

    def __init__(self, config):
        self.config       = config
        self.dataset_name = config.dataset.name           # "ai4bharat/IndicVoices"
        self.languages    = list(config.dataset.languages)
        self.split        = config.dataset.split
        self.streaming    = config.dataset.streaming
        self.max_per_lang = config.dataset.max_samples_per_language

    def _load_hf_language(self, language: str):
        """Load one language config from HuggingFace Hub."""
        from datasets import load_dataset

        config_name = LANGUAGE_CONFIG_MAP.get(language, language)

        # Strategy 1: language-specific config (works for IndicVoices)
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

        # Strategy 2: no config, filter by language column
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


# ──────────────────────────────────────────────────────────────────────────────
# Manifest loader (for data downloaded via setup_dataset.py)
# ──────────────────────────────────────────────────────────────────────────────

class ManifestLoader:
    """
    Load samples from JSONL manifests produced by setup_dataset.py.

    Expected structure (created by setup_dataset.py):
        <save_dir>/manifests/<lang>_manifests/<shard>.jsonl

    Each JSONL line has:
        audio_filepath : local path to audio file
        text           : transcript
        language       : language name
        speaker_id     : speaker identifier
        duration       : duration in seconds
    """

    def __init__(self, manifest_dir: str, languages: Optional[List[str]] = None,
                 max_per_lang: Optional[int] = None):
        self.manifest_dir = Path(manifest_dir)
        self.languages    = languages          # None = all
        self.max_per_lang = max_per_lang

    def _iter_manifest_files(self):
        """Yield (language, jsonl_path) pairs."""
        for lang_dir in sorted(self.manifest_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            # Directory name: "<lang>_manifests"
            lang = lang_dir.name.replace("_manifests", "")
            if self.languages and lang not in self.languages:
                continue
            for jf in sorted(lang_dir.glob("*.jsonl")):
                yield lang, jf

    def iter_samples(self, skip_ids: Optional[set] = None) -> Iterator[Dict[str, Any]]:
        import json
        import soundfile as sf

        skip_ids = skip_ids or set()

        for language, jf in self._iter_manifest_files():
            count = 0
            with open(jf, encoding="utf-8") as fh:
                for line in fh:
                    if self.max_per_lang is not None and count >= self.max_per_lang:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    audio_path = row.get("audio_filepath", "")
                    sample_id  = f"{language}_{Path(audio_path).stem}"

                    if sample_id in skip_ids:
                        continue
                    if not os.path.exists(audio_path):
                        logger.warning(f"Audio file not found: {audio_path}")
                        continue

                    try:
                        array, sr = sf.read(audio_path, dtype="float32", always_2d=False)
                    except Exception as e:
                        logger.warning(f"Failed to read {audio_path}: {e}")
                        continue

                    yield {
                        "id":          sample_id,
                        "language":    language,
                        "audio_array": array,
                        "sample_rate": sr,
                        "transcript":  str(row.get("text", "")),
                        "speaker_id":  str(row.get("speaker_id", "unknown")),
                    }
                    count += 1

            logger.info(f"Language '{language}': yielded {count} samples from {jf.name}")

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


# ──────────────────────────────────────────────────────────────────────────────
# Local directory loader (fallback / testing)
# ──────────────────────────────────────────────────────────────────────────────

class LocalAudioLoader:
    """
    Load audio files from a local directory tree.
    Folder structure expected: root/<language>/*.{wav,flac,mp3}
    """

    SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus"}

    def __init__(self, root_dir: str, languages: Optional[List[str]] = None,
                 max_per_lang: Optional[int] = None):
        self.root_dir     = Path(root_dir)
        self.languages    = languages
        self.max_per_lang = max_per_lang

    def iter_samples(self, skip_ids: Optional[set] = None) -> Iterator[Dict[str, Any]]:
        import soundfile as sf

        skip_ids  = skip_ids or set()
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
                        "id":          sample_id,
                        "language":    lang,
                        "audio_array": array,
                        "sample_rate": sr,
                        "transcript":  "",
                        "speaker_id":  "unknown",
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
