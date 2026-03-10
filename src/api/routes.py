"""FastAPI route definitions for the ML platform."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    HealthResponse,
    MetricsSummaryResponse,
    ModelInfoResponse,
    ModelListResponse,
    PipelineRunRequest,
    PipelineRunResponse,
    PromotionRequest,
    PromotionResponse,
)
from src.config.settings import PlatformSettings
from src.monitoring.metrics import MetricsCollector
from src.pipelines.deployment_pipeline import DeploymentPipeline
from src.pipelines.training_pipeline import TrainingPipeline
from src.registry.model_registry import ModelRegistry, ModelStage
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

settings = PlatformSettings()
registry = ModelRegistry()
metrics_collector = MetricsCollector()
deployment_pipeline = DeploymentPipeline()

_start_time: Optional[float] = None


def set_start_time(t: float) -> None:
    global _start_time
    _start_time = t


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Platform health check endpoint."""
    import time

    uptime = time.time() - (_start_time or time.time())
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=round(uptime, 2),
    )


@router.post("/pipelines/training", response_model=PipelineRunResponse)
async def run_training_pipeline(request: PipelineRunRequest):
    """Trigger a training pipeline execution."""
    try:
        pipeline = TrainingPipeline(
            settings=settings,
            model_name=request.model_name,
            model_version=request.model_version,
            task_type=request.task_type,
            algorithm=request.algorithm,
            hyperparameters=request.hyperparameters,
        )

        result = pipeline.run(
            data_path=request.data_path,
            target_column=request.target_column,
            auto_deploy=request.auto_deploy,
        )

        return PipelineRunResponse(
            run_id=result.run_id,
            status=result.status,
            model_name=result.model_name,
            model_version=result.model_version,
            metrics=result.metrics,
            passed_quality_gate=result.passed_quality_gate,
            deployed=result.deployed,
            artifact_path=result.artifact_path,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    name: Optional[str] = Query(None, description="Filter by model name"),
    stage: Optional[str] = Query(None, description="Filter by stage"),
):
    """List all registered models with optional filters."""
    stage_filter = ModelStage(stage) if stage else None
    models = registry.list_models(name=name, stage=stage_filter)

    return ModelListResponse(
        models=[
            ModelInfoResponse(
                name=m.name,
                version=m.version,
                stage=m.stage.value,
                metrics=m.metrics,
                registered_at=m.registered_at,
                description=m.description,
                tags=m.tags,
            )
            for m in models
        ],
        total=len(models),
    )


@router.get("/models/{model_name}/production", response_model=ModelInfoResponse)
async def get_production_model(model_name: str):
    """Get the current production model."""
    model = registry.get_production_model(model_name)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"No production model found for '{model_name}'",
        )

    return ModelInfoResponse(
        name=model.name,
        version=model.version,
        stage=model.stage.value,
        metrics=model.metrics,
        registered_at=model.registered_at,
        description=model.description,
        tags=model.tags,
    )


@router.post("/models/promote", response_model=PromotionResponse)
async def promote_model(request: PromotionRequest):
    """Promote a model to a target stage."""
    if request.target_stage == "staging":
        result = deployment_pipeline.promote_to_staging(
            request.model_name, request.version
        )
    elif request.target_stage == "production":
        result = deployment_pipeline.promote_to_production(
            request.model_name, request.version
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target stage: {request.target_stage}",
        )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.message)

    return PromotionResponse(
        model_name=result.model_name,
        version=result.version,
        new_stage=result.new_stage,
        previous_version=result.previous_version,
        success=result.success,
        message=result.message,
    )


@router.post(
    "/models/{model_name}/rollback/{target_version}",
    response_model=PromotionResponse,
)
async def rollback_model(model_name: str, target_version: str):
    """Rollback a model to a previous version."""
    result = deployment_pipeline.rollback(model_name, target_version)

    if not result.success:
        raise HTTPException(status_code=400, detail=result.message)

    return PromotionResponse(
        model_name=result.model_name,
        version=result.version,
        new_stage=result.new_stage,
        previous_version=result.previous_version,
        success=result.success,
        message=result.message,
    )


@router.get("/metrics", response_model=MetricsSummaryResponse)
async def get_metrics():
    """Get platform metrics summary."""
    summary = metrics_collector.get_summary()
    return MetricsSummaryResponse(**summary)


@router.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    from fastapi.responses import PlainTextResponse

    output = metrics_collector.get_prometheus_output()
    return PlainTextResponse(content=output, media_type="text/plain")
