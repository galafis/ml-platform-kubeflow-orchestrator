"""Model deployment component for serving pipeline outputs."""

import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DeploymentResult:
    """Result from a model deployment operation."""

    model_name: str
    model_version: str
    deployment_target: str
    endpoint_url: Optional[str]
    status: str
    deployed_at: str
    artifact_path: str
    metadata: Dict[str, Any]


class ModelDeployer:
    """Handles model artifact packaging and deployment.

    Supports local artifact serialization with metadata tracking.
    In production, integrates with Kubernetes or cloud endpoints.
    """

    def __init__(
        self,
        artifact_dir: str = "./artifacts",
        deployment_target: str = "local",
        base_endpoint_url: Optional[str] = None,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.deployment_target = deployment_target
        self.base_endpoint_url = base_endpoint_url or "http://localhost:8000"

    def deploy(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> DeploymentResult:
        """Deploy a trained model by serializing and registering it.

        Args:
            model: Trained scikit-learn compatible model.
            model_name: Human-readable model identifier.
            model_version: Semantic version string.
            metrics: Optional evaluation metrics.
            hyperparameters: Optional training hyperparameters.

        Returns:
            DeploymentResult with deployment details.
        """
        logger.info(
            f"Deploying model '{model_name}' v{model_version} "
            f"to {self.deployment_target}"
        )

        version_dir = self.artifact_dir / model_name / model_version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Serialize the model
        model_path = version_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        model_size_mb = model_path.stat().st_size / (1024 * 1024)

        # Save metadata alongside the model
        metadata = {
            "model_name": model_name,
            "model_version": model_version,
            "deployment_target": self.deployment_target,
            "model_size_mb": round(model_size_mb, 2),
            "deployed_at": datetime.utcnow().isoformat(),
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "artifact_path": str(model_path),
            "model_class": type(model).__name__,
        }

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        endpoint_url = (
            f"{self.base_endpoint_url}/v1/models/{model_name}"
            f"/versions/{model_version}/predict"
        )

        result = DeploymentResult(
            model_name=model_name,
            model_version=model_version,
            deployment_target=self.deployment_target,
            endpoint_url=endpoint_url,
            status="deployed",
            deployed_at=metadata["deployed_at"],
            artifact_path=str(model_path),
            metadata=metadata,
        )

        logger.info(
            f"Model deployed: {model_name} v{model_version} "
            f"({model_size_mb:.2f} MB) -> {endpoint_url}"
        )

        return result

    def load_model(
        self, model_name: str, model_version: str
    ) -> Any:
        """Load a previously deployed model from artifacts."""
        model_path = (
            self.artifact_dir / model_name / model_version / "model.pkl"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}"
            )

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Loaded model: {model_name} v{model_version}")
        return model

    def list_deployments(
        self, model_name: Optional[str] = None
    ) -> list:
        """List all deployed model versions."""
        deployments = []

        search_path = (
            self.artifact_dir / model_name
            if model_name
            else self.artifact_dir
        )

        if not search_path.exists():
            return deployments

        for metadata_file in search_path.rglob("metadata.json"):
            with open(metadata_file) as f:
                metadata = json.load(f)
            deployments.append(metadata)

        return sorted(
            deployments,
            key=lambda x: x.get("deployed_at", ""),
            reverse=True,
        )

    def rollback(
        self, model_name: str, target_version: str
    ) -> DeploymentResult:
        """Rollback to a specific model version.

        Returns:
            DeploymentResult pointing to the rollback version.
        """
        model = self.load_model(model_name, target_version)

        metadata_path = (
            self.artifact_dir
            / model_name
            / target_version
            / "metadata.json"
        )
        with open(metadata_path) as f:
            metadata = json.load(f)

        logger.info(
            f"Rolling back {model_name} to v{target_version}"
        )

        return DeploymentResult(
            model_name=model_name,
            model_version=target_version,
            deployment_target=self.deployment_target,
            endpoint_url=(
                f"{self.base_endpoint_url}/v1/models/{model_name}"
                f"/versions/{target_version}/predict"
            ),
            status="rollback",
            deployed_at=datetime.utcnow().isoformat(),
            artifact_path=str(
                self.artifact_dir / model_name / target_version / "model.pkl"
            ),
            metadata=metadata,
        )
