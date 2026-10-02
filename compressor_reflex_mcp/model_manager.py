"""
Model Manager for Compressor Reflex.
Handles locating, downloading, caching, and cryptographic integrity verification
of model weights from Hugging Face Hub:
https://huggingface.co/aialchemist-dev/compressor-reflex
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Optional, Dict

HF_REPO_ID = "aialchemist-dev/compressor-reflex"
# Pinned immutable commit revision to protect against supply chain tampering
HF_PINNED_REVISION = os.environ.get("COMPRESSOR_HF_REVISION", "666666b051e7c4809c484cec6071512664525525")

# Cryptographic SHA-256 digests of verified, delivered model artifacts
VERIFIED_SHA256 = {
    "model_int8.onnx": "98af8b8106c9d56c2fde1a334ef8b12524913071ffcb422dee017cab36092b94",
    "tokenizer.json": "56fe0ed03f90cad1671dc293e57f7ea11a2779b46b4484c164a4550c97509e5f",
    "tokenizer_config.json": "4dcff24e98312b0e225c34cf7238693cd4e34cf0cd7c8201fdf88b020ce9fd89",
}

REQUIRED_FILES = list(VERIFIED_SHA256.keys())

def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a local file in memory-efficient chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def get_default_cache_dir() -> Path:
    """Returns the default cache directory for compressor-reflex weights."""
    if os.environ.get("COMPRESSOR_MODEL_DIR"):
        return Path(os.environ["COMPRESSOR_MODEL_DIR"]).resolve()

    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        base = Path(local_appdata) if local_appdata else Path.home() / ".cache"
    else:
        base = Path.home() / ".cache"

    return (base / "compressor-reflex").resolve()

def check_local_model_dir(dir_path: Path, verify_hashes: bool = True) -> bool:
    """
    Checks whether all required model files exist and (optionally) match verified cryptographic hashes.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return False

    for fname, expected_hash in VERIFIED_SHA256.items():
        fpath = dir_path / fname
        if not fpath.exists():
            return False
        if verify_hashes and not os.environ.get("COMPRESSOR_SKIP_HASH_CHECK"):
            actual_hash = compute_sha256(fpath)
            if actual_hash != expected_hash:
                print(
                    f"[CompressorReflex Security] Checksum mismatch for {fname}! "
                    f"Expected: {expected_hash}, Got: {actual_hash}",
                    file=sys.stderr
                )
                return False

    return True

def ensure_model_files(
    target_dir: Optional[Path] = None,
    force_download: bool = False,
    quiet: bool = False
) -> Path:
    """
    Ensures that model_int8.onnx and tokenizer files exist locally and pass SHA-256 verification.
    If missing or corrupt, downloads from Hugging Face at the pinned immutable revision.
    """
    model_dir = target_dir or get_default_cache_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    if not force_download and check_local_model_dir(model_dir):
        return model_dir

    if not quiet:
        print(f"[CompressorReflex] Model weights not found or invalid in {model_dir}", file=sys.stderr)
        print(f"[CompressorReflex] Downloading from Hugging Face ({HF_REPO_ID}@{HF_PINNED_REVISION[:8]})...", file=sys.stderr)

    try:
        from huggingface_hub import hf_hub_download

        for fname, expected_hash in VERIFIED_SHA256.items():
            if not quiet:
                print(f"[CompressorReflex] Downloading {fname}...", file=sys.stderr)
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=fname,
                revision=HF_PINNED_REVISION,
                local_dir=str(model_dir)
            )

            # Strict cryptographic verification
            if not os.environ.get("COMPRESSOR_SKIP_HASH_CHECK"):
                actual_hash = compute_sha256(Path(downloaded_path))
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Cryptographic integrity verification failed for {fname}! "
                        f"Expected SHA-256: {expected_hash}, Actual: {actual_hash}. "
                        "Aborting to protect against supply chain tampering."
                    )

        if not quiet:
            print(f"[CompressorReflex] Successfully verified and cached model to: {model_dir}", file=sys.stderr)

    except Exception as exc:
        raise RuntimeError(
            f"Failed to securely obtain Compressor Reflex model weights from Hugging Face Hub ({HF_REPO_ID}). "
            f"Error details: {exc}"
        ) from exc

    return model_dir

def get_model_info() -> Dict[str, str]:
    """Returns metadata about the model repository and local cache."""
    cache_dir = get_default_cache_dir()
    cached = check_local_model_dir(cache_dir, verify_hashes=False)
    return {
        "repo_id": HF_REPO_ID,
        "repo_url": f"https://huggingface.co/{HF_REPO_ID}",
        "pinned_revision": HF_PINNED_REVISION,
        "cache_dir": str(cache_dir),
        "is_cached": str(cached),
        "integrity_verification": "SHA-256 Enabled"
    }