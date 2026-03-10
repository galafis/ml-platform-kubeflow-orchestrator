"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    """Request to trigger a pipeline run."""

    data_path: str = Field(..., description="Path to training data")
    target_column: str = Field(..., description="Target variable column name")
    model_name: str = Field(default="default-model", description="Model identifier")
    model_version: str = Field(default="1.0.0", description="Semantic version")
    task_type: str = Field(default="classification", description="Task type")
    algorithm: str = Field(default="gradient_boosting", description="ML algorithm")
    auto_deploy: bool = Field(default=True, description="Auto-deploy on gate pass")
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional hyperparameters"
    )


class PipelineRunResponse(BaseModel):
    """Response from a pipeline run."""

    run_id: str
    status: str
    model_name: str
    model_version: str
    metrics: Dict[str, float]
    passed_quality_gate: bool
    deployed: bool
    artifact_path: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model information from the registry."""

    name: str
    version: str
    stage: str
    metrics: Dict[str, float]
    registered_at: str
    description: str = ""
    tags: Dict[str, str] = {}


class ModelListResponse(BaseModel):
    """List of registered models."""

    models: List[ModelInfoResponse]
    total: int


class PromotionRequest(BaseModel):
    """Request to promote a model version."""

    model_name: str
    version: str
    target_stage: str = Field(
        ..., description="Target stage: staging or production"
    )


class PromotionResponse(BaseModel):
    """Response from a promotion operation."""

    model_name: str
    version: str
    new_stage: str
    previous_version: Optional[str] = None
    success: bool
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    uptime_seconds: float


class MetricsSummaryResponse(BaseModel):
    """Metrics summary response."""

    counters: Dict[str, float]
    gauges: Dict[str, float]
    histogram_counts: Dict[str, int]
    total_records: int
