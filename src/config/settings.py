"""Application settings with Pydantic configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class PipelineConfig:
    """Configuration for Kubeflow pipeline execution."""

    name: str = "ml-training-pipeline"
    namespace: str = "kubeflow"
    experiment_name: str = "default-experiment"
    pipeline_root: str = "gs://ml-platform-artifacts/pipelines"
    service_account: str = "pipeline-runner"
    max_cache_staleness: str = "P0D"
    enable_caching: bool = True


@dataclass
class MLflowConfig:
    """MLflow tracking server configuration."""

    tracking_uri: str = "http://mlflow:5000"
    registry_uri: str = "http://mlflow:5000"
    artifact_location: str = "gs://ml-platform-artifacts/mlflow"
    default_experiment: str = "platform-experiments"


@dataclass
class DatabaseConfig:
    """PostgreSQL metadata store configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "ml_platform"
    user: str = "ml_admin"
    password: str = "changeme"
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class APIConfig:
    """FastAPI server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"


@dataclass
class MonitoringConfig:
    """Prometheus metrics configuration."""

    enabled: bool = True
    port: int = 9090
    push_gateway: str = "http://prometheus-pushgateway:9091"


@dataclass
class QualityGateConfig:
    """Model quality gate thresholds for automated promotion."""

    min_accuracy: float = 0.85
    min_precision: float = 0.80
    min_recall: float = 0.80
    min_f1_score: float = 0.82
    max_latency_ms: float = 100.0
    max_model_size_mb: float = 500.0


@dataclass
class PlatformSettings:
    """Root configuration aggregating all sub-configs."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PlatformSettings":
        """Load settings from a YAML configuration file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()

        with open(config_path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        return cls(
            pipeline=PipelineConfig(**raw.get("pipeline", {})),
            mlflow=MLflowConfig(**raw.get("mlflow", {})),
            database=DatabaseConfig(**raw.get("database", {})),
            api=APIConfig(**raw.get("api", {})),
            monitoring=MonitoringConfig(**raw.get("monitoring", {})),
            quality_gate=QualityGateConfig(**raw.get("quality_gate", {})),
        )


def load_settings(config_path: Optional[str] = None) -> PlatformSettings:
    """Load platform settings from YAML or use defaults."""
    if config_path:
        return PlatformSettings.from_yaml(config_path)
    return PlatformSettings()
