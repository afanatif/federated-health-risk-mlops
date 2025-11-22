"""
Model Registration Script
-------------------------
This script performs the full workflow for adding a new model to the registry:

1. Validate the model using mlops/evaluate/validate_model.py
2. Apply a policy decision (pass/fail)
3. If passed → register model using registry_utils
4. Automatically assign version, compute hash, and store metadata

This script is intentionally small — the business logic belongs in:
- validate_model.py
- metrics.py
- registry_utils.py
"""

import json
from pathlib import Path
from datetime import datetime

from mlops.evaluate.validate_model import validate_model
from mlops.evaluate.metrics import model_passes_policy
from mlops.model_registry.registry_utils import register_model


def register_new_model(model_path: str, task: str = "detection"):
    """
    Full workflow for validating + registering a model.

    Parameters
    ----------
    model_path : str
        Path to .pt/.pth model inside models/incoming/
    task : str
        Task type for the validator ("detection", etc.)

    Returns
    -------
    dict
        Information about registration result.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.suffix not in [".pt", ".pth"]:
        raise ValueError("Only .pt or .pth YOLO models are supported.")

    print(f"🔍 Validating model: {model_path}")

    # -------------------------
    # STEP 1 — VALIDATE MODEL
    # -------------------------
    metrics = validate_model(model_path=str(model_path), task=task)

    print("Validation metrics:")
    print(json.dumps(metrics, indent=4))

    # -------------------------
    # STEP 2 — DECISION POLICY
    # -------------------------
    if not model_passes_policy(metrics):
        print("❌ Model FAILED validation policy. Not registering.")
        return {
            "status": "rejected",
            "reason": "validation_failed",
            "metrics": metrics
        }

    print("✅ Model PASSED validation policy. Proceeding to registry...")

    # -------------------------
    # STEP 3 — REGISTER MODEL
    # -------------------------
    metadata = {
        "task": task,
        "validated_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "source": "incoming",
        "status": "validated",
    }

    registry_info = register_model(model_path, metadata)

    # -------------------------
    # STEP 4 — OUTPUT
    # -------------------------
    print("📦 Model successfully registered!")
    print(f"➡ Registry path: {registry_info['model_path']}")
    print(f"➡ Version: {registry_info['version']}")

    return {
        "status": "registered",
        "version": registry_info["version"],
        "registry_path": registry_info["model_path"],
        "metadata_path": registry_info["metadata_path"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register a new trained model.")
    parser.add_argument("--model", required=True, help="Path to model in models/incoming/")
    parser.add_argument("--task", default="detection", help="YOLO task type")

    args = parser.parse_args()

    result = register_new_model(args.model, args.task)
    print("\nFinal Output:")
    print(json.dumps(result, indent=4))
