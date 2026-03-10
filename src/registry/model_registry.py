"""Model registry for versioned model management with MLflow integration."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelStage(str, Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class RegisteredModel:
    """A registered model entry in the registry."""

    name: str
    version: str
    stage: ModelStage
    artifact_path: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    registered_at: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


class ModelRegistry:
    """Local model registry with stage-based lifecycle management.

    Provides versioned model tracking, stage transitions, and
    metadata storage. Designed as a lightweight alternative for
    environments without MLflow connectivity.
    """

    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_path / "index.json"
        self._index: Dict[str, List[Dict]] = self._load_index()

    def _load_index(self) -> Dict[str, List[Dict]]:
        """Load the registry index from disk."""
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Persist the registry index to disk."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    def register_model(
        self,
        name: str,
        version: str,
        artifact_path: str,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> RegisteredModel:
        """Register a new model version in the registry.

        Args:
            name: Model name identifier.
            version: Semantic version string.
            artifact_path: Path to the model artifact.
            metrics: Evaluation metrics.
            hyperparameters: Training hyperparameters.
            description: Human-readable description.
            tags: Key-value tags for filtering.

        Returns:
            The registered model entry.
        """
        model = RegisteredModel(
            name=name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            artifact_path=artifact_path,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            registered_at=datetime.utcnow().isoformat(),
            description=description,
            tags=tags or {},
        )

        if name not in self._index:
            self._index[name] = []

        existing_versions = {
            entry["version"] for entry in self._index[name]
        }
        if version in existing_versions:
            raise ValueError(
                f"Model '{name}' version '{version}' already exists"
            )

        self._index[name].append({
            "version": model.version,
            "stage": model.stage.value,
            "artifact_path": model.artifact_path,
            "metrics": model.metrics,
            "hyperparameters": model.hyperparameters,
            "registered_at": model.registered_at,
            "description": model.description,
            "tags": model.tags,
        })

        self._save_index()

        logger.info(
            f"Registered model: {name} v{version} "
            f"(stage={model.stage.value})"
        )

        return model

    def transition_stage(
        self,
        name: str,
        version: str,
        target_stage: ModelStage,
    ) -> RegisteredModel:
        """Transition a model version to a new lifecycle stage.

        If promoting to PRODUCTION, any existing production version
        of the same model is automatically moved to ARCHIVED.
        """
        if name not in self._index:
            raise KeyError(f"Model '{name}' not found in registry")

        entry = None
        for item in self._index[name]:
            if item["version"] == version:
                entry = item
                break

        if entry is None:
            raise KeyError(
                f"Version '{version}' not found for model '{name}'"
            )

        old_stage = entry["stage"]

        # Auto-archive current production model
        if target_stage == ModelStage.PRODUCTION:
            for item in self._index[name]:
                if (
                    item["stage"] == ModelStage.PRODUCTION.value
                    and item["version"] != version
                ):
                    item["stage"] = ModelStage.ARCHIVED.value
                    logger.info(
                        f"Auto-archived {name} v{item['version']} "
                        f"(replaced by v{version})"
                    )

        entry["stage"] = target_stage.value
        self._save_index()

        logger.info(
            f"Transitioned {name} v{version}: "
            f"{old_stage} -> {target_stage.value}"
        )

        return RegisteredModel(
            name=name,
            version=version,
            stage=target_stage,
            artifact_path=entry["artifact_path"],
            metrics=entry["metrics"],
            hyperparameters=entry["hyperparameters"],
            registered_at=entry["registered_at"],
            description=entry.get("description", ""),
            tags=entry.get("tags", {}),
        )

    def get_production_model(
        self, name: str
    ) -> Optional[RegisteredModel]:
        """Get the current production model for a given name."""
        if name not in self._index:
            return None

        for entry in self._index[name]:
            if entry["stage"] == ModelStage.PRODUCTION.value:
                return RegisteredModel(
                    name=name,
                    version=entry["version"],
                    stage=ModelStage.PRODUCTION,
                    artifact_path=entry["artifact_path"],
                    metrics=entry["metrics"],
                    hyperparameters=entry["hyperparameters"],
                    registered_at=entry["registered_at"],
                    description=entry.get("description", ""),
                    tags=entry.get("tags", {}),
                )
        return None

    def list_models(
        self, name: Optional[str] = None, stage: Optional[ModelStage] = None
    ) -> List[RegisteredModel]:
        """List registered models with optional filters."""
        results: List[RegisteredModel] = []

        search_models = (
            {name: self._index[name]}
            if name and name in self._index
            else self._index
        )

        for model_name, versions in search_models.items():
            for entry in versions:
                if stage and entry["stage"] != stage.value:
                    continue
                results.append(
                    RegisteredModel(
                        name=model_name,
                        version=entry["version"],
                        stage=ModelStage(entry["stage"]),
                        artifact_path=entry["artifact_path"],
                        metrics=entry["metrics"],
                        hyperparameters=entry["hyperparameters"],
                        registered_at=entry["registered_at"],
                        description=entry.get("description", ""),
                        tags=entry.get("tags", {}),
                    )
                )

        return results

    def delete_model(self, name: str, version: str) -> None:
        """Remove a model version from the registry."""
        if name not in self._index:
            raise KeyError(f"Model '{name}' not found")

        self._index[name] = [
            e for e in self._index[name] if e["version"] != version
        ]

        if not self._index[name]:
            del self._index[name]

        self._save_index()
        logger.info(f"Deleted model: {name} v{version}")
