"""Integration tests for the training pipeline end-to-end flow."""

import numpy as np
import pandas as pd
import pytest

from src.config.settings import PlatformSettings, QualityGateConfig
from src.pipelines.training_pipeline import TrainingPipeline


class TestTrainingPipelineIntegration:
    def test_full_pipeline_run(self, sample_csv_path, tmp_path):
        """Test complete pipeline: load -> preprocess -> train -> evaluate -> deploy."""
        settings = PlatformSettings(
            quality_gate=QualityGateConfig(
                min_accuracy=0.3,
                min_precision=0.3,
                min_recall=0.3,
                min_f1_score=0.3,
            )
        )

        pipeline = TrainingPipeline(
            settings=settings,
            model_name="integration-test-model",
            model_version="1.0.0",
            task_type="classification",
            algorithm="gradient_boosting",
        )

        # Override artifact paths for testing
        pipeline.deployer.artifact_dir = tmp_path / "artifacts"
        pipeline.deployer.artifact_dir.mkdir()
        pipeline.registry.registry_path = tmp_path / "registry"
        pipeline.registry.registry_path.mkdir()
        pipeline.registry._index_path = (
            pipeline.registry.registry_path / "index.json"
        )
        pipeline.registry._index = {}

        result = pipeline.run(
            data_path=sample_csv_path,
            target_column="target",
            auto_deploy=True,
        )

        assert result.run_id is not None
        assert result.model_name == "integration-test-model"
        assert result.model_version == "1.0.0"
        assert "accuracy" in result.metrics
        assert isinstance(result.passed_quality_gate, bool)

    def test_pipeline_without_auto_deploy(self, sample_csv_path, tmp_path):
        """Test pipeline with auto_deploy=False."""
        settings = PlatformSettings()

        pipeline = TrainingPipeline(
            settings=settings,
            model_name="no-deploy-model",
            model_version="0.1.0",
        )
        pipeline.deployer.artifact_dir = tmp_path / "artifacts2"
        pipeline.deployer.artifact_dir.mkdir()
        pipeline.registry.registry_path = tmp_path / "registry2"
        pipeline.registry.registry_path.mkdir()
        pipeline.registry._index_path = (
            pipeline.registry.registry_path / "index.json"
        )
        pipeline.registry._index = {}

        result = pipeline.run(
            data_path=sample_csv_path,
            target_column="target",
            auto_deploy=False,
        )

        assert result.deployed is False
