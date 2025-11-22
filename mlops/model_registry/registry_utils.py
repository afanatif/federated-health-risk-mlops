"""
Simple Model Registry Utilities
Location: mlops/model_registry/registry_utils.py

Handles model versioning, registration, promotion, and metadata management.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import hashlib

from config.settings import get_path

# ---------------------------------------------------------------------------
# Directory Setup
# ---------------------------------------------------------------------------

ROOT = get_path("models", "root")
INCOMING_DIR = get_path("models", "incoming")
BLESSED_DIR = get_path("models", "blessed")
REJECTED_DIR = get_path("models", "rejected")
ARCHIVE_DIR = get_path("models", "archive")
METADATA_DIR = get_path("models", "metadata")


for d in [INCOMING_DIR, BLESSED_DIR, REJECTED_DIR, ARCHIVE_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _generate_hash(filepath: Path, chunk_size: int = 65536):
    """
    Compute an MD5 checksum for the given model file.

    Useful for:
    - Detecting duplicate models (same file → same hash)
    - Tracking exact artifacts used for training/evaluation
    - Detecting tampering or file corruption
    - Supporting reproducibility by attaching a stable fingerprint

    Parameters
    ----------
    filepath : Path
        Path to the model artifact (.pt, .pth, .onnx, etc.)
    chunk_size : int, optional
        Size of chunks for streaming the file to avoid RAM spikes.

    Returns
    -------
    str
        Hex digest MD5 hash of the file contents.
    """
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()



def _load_metadata_file(meta_file: Path):
    if not meta_file.exists():
        return []
    with open(meta_file, "r") as f:
        return json.load(f)


def _save_metadata_file(meta_file: Path, data):
    with open(meta_file, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# 1. Register Model
# ---------------------------------------------------------------------------

def register_model(model_path: Path, metrics: dict) -> dict:
    """
    Register a new model coming from training/validation.

    Steps:
    1. Move to archive with versioning.
    2. Save metadata entry.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_path} not found")

    # Create version tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"{model_path.stem}_v{timestamp}"
    archived_model = ARCHIVE_DIR / f"{version}{model_path.suffix}"

    # Move model
    shutil.copy(model_path, archived_model)

    # Compute hash
    file_hash = _generate_hash(archived_model)

    # Metadata
    metadata_entry = {
        "version": version,
        "file_name": archived_model.name,
        "path": str(archived_model),
        "registered_at": timestamp,
        "metrics": metrics,
        "hash_md5": file_hash,
        "status": "archived"
    }

    meta_file = METADATA_DIR / "registry.json"
    metadata_list = _load_metadata_file(meta_file)
    metadata_list.append(metadata_entry)
    _save_metadata_file(meta_file, metadata_list)

    return metadata_entry


# ---------------------------------------------------------------------------
# 2. Promote Model → blessed/
# ---------------------------------------------------------------------------

def promote_model(version: str):
    """
    Promote a versioned model from archive → blessed.
    """
    meta_file = METADATA_DIR / "registry.json"
    metadata = _load_metadata_file(meta_file)

    # Find entry
    entry = next((m for m in metadata if m["version"] == version), None)
    if not entry:
        raise ValueError(f"Version {version} not found in registry")

    archived_path = Path(entry["path"])

    if not archived_path.exists():
        raise FileNotFoundError(f"Archived file missing: {archived_path}")

    # Copy to blessed
    blessed_path = BLESSED_DIR / archived_path.name
    shutil.copy(archived_path, blessed_path)

    # Update metadata
    entry["status"] = "blessed"
    entry["blessed_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    _save_metadata_file(meta_file, metadata)

    return str(blessed_path)


# ---------------------------------------------------------------------------
# 3. Reject Model → rejected/
# ---------------------------------------------------------------------------

def reject_model(model_path: Path, reason: str = "Failed validation"):
    """
    Move raw incoming model to rejected/ with metadata entry.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    rejected_path = REJECTED_DIR / model_path.name
    shutil.move(model_path, rejected_path)

    # Log rejection metadata
    meta_file = METADATA_DIR / "rejected.json"
    entry = {
        "file": model_path.name,
        "rejected_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "reason": reason
    }

    meta_list = _load_metadata_file(meta_file)
    meta_list.append(entry)
    _save_metadata_file(meta_file, meta_list)

    return str(rejected_path)


# ---------------------------------------------------------------------------
# 4. List registry entries
# ---------------------------------------------------------------------------

def list_registry(status: str = None):
    """
    Returns all registry entries, optionally filtered by status.
    """
    meta_file = METADATA_DIR / "registry.json"
    metadata = _load_metadata_file(meta_file)

    if status:
        return [m for m in metadata if m.get("status") == status]

    return metadata


# ---------------------------------------------------------------------------
# 5. Get best model according to metric (e.g. mAP50)
# ---------------------------------------------------------------------------

def get_best_model(metric: str = "mAP50"):
    """
    Returns the version with the highest metric value.
    """
    registry = list_registry(status="archived")

    if not registry:
        return None

    best = max(registry, key=lambda m: m["metrics"].get(metric, 0.0))
    return best

