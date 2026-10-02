"""
Model Manager for Compressor Reflex.
Handles locating, downloading, and caching model weights from Hugging Face Hub:
https://huggingface.co/aialchemist-dev/compressor-reflex
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict

HF_REPO_ID = "aialchemist-dev/compressor-reflex"
REQUIRED_FILES = [
    "model_int8.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
]

def get_default_cache_dir() -> Path:
    """Returns the default cache directory for compressor-reflex weights."""
    if os.environ.get("COMPRESSOR_MODEL_DIR"):
        return Path(os.environ["COMPRESSOR_MODEL_DIR"]).resolve()

    # User home cache: ~/.cache/compressor-reflex or %LOCALAPPDATA%/compressor-reflex on Windows
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            base = Path(local_appdata)
        else:
            base = Path.home() / ".cache"
    else:
        base = Path.home() / ".cache"

    return (base / "compressor-reflex").resolve()

def check_local_model_dir(dir_path: Path) -> bool:
    """Checks whether all required model files exist in the specified directory."""
    if not dir_path.exists() or not dir_path.is_dir():
        return False
    return all((dir_path / f).exists() for f in REQUIRED_FILES)

def ensure_model_files(
    target_dir: Optional[Path] = None,
    force_download: bool = False,
    quiet: bool = False
) -> Path:
    """
    Ensures that model_int8.onnx and tokenizer files exist locally.
    If missing, automatically downloads from Hugging Face: aialchemist-dev/compressor-reflex.
    """
    model_dir = target_dir or get_default_cache_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    if not force_download and check_local_model_dir(model_dir):
        return model_dir

    if not quiet:
        print(f"[CompressorReflex] Model weights not found in {model_dir}", file=sys.stderr)
        print(f"[CompressorReflex] Downloading from Hugging Face ({HF_REPO_ID})...", file=sys.stderr)

    try:
        from huggingface_hub import hf_hub_download

        for fname in REQUIRED_FILES:
            if not quiet:
                print(f"[CompressorReflex] Downloading {fname}...", file=sys.stderr)
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=fname,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )

        if not quiet:
            print(f"[CompressorReflex] Successfully downloaded and cached model to: {model_dir}", file=sys.stderr)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to download Compressor Reflex model weights from Hugging Face Hub ({HF_REPO_ID}). "
            f"Please check your internet connection or manually place {REQUIRED_FILES} into {model_dir}. "
            f"Error details: {exc}"
        ) from exc

    return model_dir

def get_model_info() -> Dict[str, str]:
    """Returns metadata about the model repository and local cache."""
    cache_dir = get_default_cache_dir()
    cached = check_local_model_dir(cache_dir)
    return {
        "repo_id": HF_REPO_ID,
        "repo_url": f"https://huggingface.co/{HF_REPO_ID}",
        "cache_dir": str(cache_dir),
        "is_cached": str(cached),
        "required_files": ", ".join(REQUIRED_FILES)
    }
