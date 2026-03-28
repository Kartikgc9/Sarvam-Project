"""
One-time model setup script.

Downloads:
  1. DNSMOS P.835 ONNX model  (Microsoft DNS-Challenge)
  2. Silero-VAD               (via torch.hub cache)

Run once before the pipeline:
    python setup_models.py
"""

import logging
import os
import sys

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DNSMOS model
# ---------------------------------------------------------------------------

DNSMOS_URLS = [
    # Primary: Microsoft DNS Challenge GitHub raw file
    "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
    # Mirror / fallback
    "https://github.com/microsoft/DNS-Challenge/raw/refs/heads/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
]
DNSMOS_PATH = "models/sig_bak_ovr.onnx"


def download_dnsmos() -> bool:
    """Download the DNSMOS ONNX model file."""
    os.makedirs("models", exist_ok=True)

    if os.path.exists(DNSMOS_PATH):
        size_kb = os.path.getsize(DNSMOS_PATH) / 1024
        logger.info(f"DNSMOS model already present ({size_kb:.0f} KB): {DNSMOS_PATH}")
        return True

    for url in DNSMOS_URLS:
        logger.info(f"Downloading DNSMOS from {url} …")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()

            with open(DNSMOS_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            size_kb = os.path.getsize(DNSMOS_PATH) / 1024
            logger.info(f"DNSMOS model saved ({size_kb:.0f} KB): {DNSMOS_PATH}")
            return True

        except Exception as e:
            logger.warning(f"  Failed ({url}): {e}")
            if os.path.exists(DNSMOS_PATH):
                os.remove(DNSMOS_PATH)   # clean up partial download

    logger.error(
        "Could not download DNSMOS model.\n"
        "Manual download steps:\n"
        "  1. Visit https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS/DNSMOS\n"
        "  2. Download 'sig_bak_ovr.onnx'\n"
        "  3. Place it at: models/sig_bak_ovr.onnx"
    )
    return False


# ---------------------------------------------------------------------------
# Silero-VAD
# ---------------------------------------------------------------------------

def setup_silero_vad() -> bool:
    """Pre-download Silero-VAD via torch.hub."""
    os.makedirs("models", exist_ok=True)
    try:
        import torch
        torch.hub.set_dir("models/")
        logger.info("Downloading / caching Silero-VAD via torch.hub …")

        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
        except TypeError:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                verbose=False,
            )

        logger.info("Silero-VAD cached successfully.")
        return True

    except Exception as e:
        logger.error(f"Silero-VAD setup failed: {e}")
        logger.error(
            "Make sure torch is installed: pip install torch torchaudio"
        )
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Audio Filtering Pipeline — Model Setup")
    print("=" * 55)

    dnsmos_ok = download_dnsmos()
    vad_ok    = setup_silero_vad()

    print("\n" + "=" * 55)
    print(f"  DNSMOS P.835  : {'✓ OK' if dnsmos_ok else '✗ FAILED'}")
    print(f"  Silero-VAD    : {'✓ OK' if vad_ok    else '✗ FAILED'}")
    print("=" * 55)

    if dnsmos_ok and vad_ok:
        print("\n  All models ready.  Run the pipeline with:")
        print("    python main.py")
    else:
        print("\n  Some models failed.  Check the logs above.")
        sys.exit(1)
