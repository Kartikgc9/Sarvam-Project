"""
Audio Filtering Pipeline — CLI Entry Point

Usage
-----
# Full pipeline run (uses config/config.yaml by default)
    python main.py

# Custom config
    python main.py --config config/config.yaml

# Run only visualization (if results.csv already exists)
    python main.py --visualize-only

# Override max samples per language without editing config
    python main.py --max-samples 200

# Use a local audio directory instead of HuggingFace
    python main.py --local-dir /path/to/audio/

The `if __name__ == '__main__'` guard is REQUIRED on Windows
to prevent ProcessPoolExecutor workers from spawning sub-processes
indefinitely (Windows uses 'spawn', not 'fork').
"""

import argparse
import logging
import os
import sys

# Make `src` importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))


def _setup_logging(log_file: str = "outputs/pipeline.log"):
    os.makedirs("outputs", exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio Filtering Pipeline for Indic Speech",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to OmegaConf YAML config file.",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Skip pipeline run; only generate plots from existing results.csv.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max_samples_per_language in config (useful for quick tests).",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Load audio from a local directory tree (root/<lang>/*.wav) "
             "instead of HuggingFace.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Override output CSV path.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    _setup_logging()
    logger = logging.getLogger(__name__)

    # ── Visualize-only mode ───────────────────────────────────────────────
    if args.visualize_only:
        from src.visualize import generate_all_plots
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        csv_path = args.output_csv or str(cfg.pipeline.output_csv)
        generate_all_plots(csv_path=csv_path)
        return

    # ── Full pipeline run ─────────────────────────────────────────────────
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)

    # Apply CLI overrides
    if args.max_samples is not None:
        cfg.dataset.max_samples_per_language = args.max_samples
        logger.info(f"max_samples_per_language overridden to {args.max_samples}")

    if args.output_csv is not None:
        cfg.pipeline.output_csv = args.output_csv

    if args.local_dir is not None:
        # Swap loader to LocalAudioLoader
        from src.pipeline import AudioFilteringPipeline
        from src.loader import LocalAudioLoader

        pipeline = AudioFilteringPipeline.__new__(AudioFilteringPipeline)
        pipeline.config = cfg
        pipeline._config_dict = OmegaConf.to_container(cfg, resolve=True)
        pipeline._setup_output_dirs()
        pipeline._init_csv()

        from src.metrics.vad_metrics import load_vad_model
        from src.metrics.dnsmos import load_dnsmos_model
        logger.info("Loading VAD model …")
        pipeline.vad_model, pipeline.vad_utils = load_vad_model(
            str(cfg.models.vad_cache_dir)
        )
        logger.info("Loading DNSMOS model …")
        pipeline.dnsmos_session = load_dnsmos_model(str(cfg.models.dnsmos_path))

        pipeline.loader = LocalAudioLoader(
            root_dir=args.local_dir,
            languages=list(cfg.dataset.languages) if cfg.dataset.languages else None,
            max_per_lang=cfg.dataset.max_samples_per_language,
        )
        pipeline.processed_ids = pipeline._load_checkpoint()
    else:
        from src.pipeline import AudioFilteringPipeline
        pipeline = AudioFilteringPipeline(config_path=args.config)
        # Apply potential overrides post-init
        if args.max_samples is not None:
            pipeline.config.dataset.max_samples_per_language = args.max_samples

    # Run
    stats = pipeline.run()

    # Generate visualizations
    logger.info("Generating visualizations …")
    from src.visualize import generate_all_plots
    generate_all_plots(
        csv_path=str(cfg.pipeline.output_csv),
        threshold=float(cfg.scoring.threshold),
    )

    logger.info("Pipeline complete.  Check outputs/ for results and plots.")
    return stats


if __name__ == "__main__":
    # Required guard for Windows multiprocessing (spawn method)
    main()
