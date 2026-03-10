"""Unit tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_metrics_endpoint(self):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "counters" in data
        assert "gauges" in data

    def test_prometheus_endpoint(self):
        response = client.get("/api/v1/metrics/prometheus")
        assert response.status_code == 200


class TestModelsEndpoint:
    def test_list_models_empty(self):
        response = client.get("/api/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "total" in data

    def test_get_production_model_not_found(self):
        response = client.get(
            "/api/v1/models/nonexistent-model/production"
        )
        assert response.status_code == 404


class TestPipelineEndpoint:
    def test_training_pipeline_missing_data(self):
        response = client.post(
            "/api/v1/pipelines/training",
            json={
                "data_path": "/nonexistent/path.csv",
                "target_column": "target",
            },
        )
        assert response.status_code in (404, 500)

    def test_training_pipeline_validation(self):
        response = client.post(
            "/api/v1/pipelines/training",
            json={
                "data_path": "test.csv",
                "target_column": "target",
                "task_type": "classification",
                "algorithm": "gradient_boosting",
            },
        )
        # Should fail because the file doesn't exist
        assert response.status_code in (404, 500)


class TestPromotionEndpoint:
    def test_invalid_stage(self):
        response = client.post(
            "/api/v1/models/promote",
            json={
                "model_name": "test",
                "version": "1.0.0",
                "target_stage": "invalid",
            },
        )
        assert response.status_code == 400
