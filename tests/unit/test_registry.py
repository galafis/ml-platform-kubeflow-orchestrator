"""Unit tests for the model registry."""

import pytest

from src.registry.model_registry import ModelRegistry, ModelStage


class TestModelRegistry:
    def test_register_model(self, temp_registry_dir):
        registry = ModelRegistry(registry_path=temp_registry_dir)

        model = registry.register_model(
            name="test-model",
            version="1.0.0",
            artifact_path="/tmp/model.pkl",
            metrics={"accuracy": 0.92},
            hyperparameters={"n_estimators": 100},
        )

        assert model.name == "test-model"
        assert model.version == "1.0.0"
        assert model.stage == ModelStage.DEVELOPMENT

    def test_duplicate_version_raises(self, temp_registry_dir):
        registry = ModelRegistry(registry_path=temp_registry_dir)

        registry.register_model(
            name="model-a",
            version="1.0.0",
            artifact_path="/tmp/model.pkl",
        )

        with pytest.raises(ValueError, match="already exists"):
            registry.register_model(
                name="model-a",
                version="1.0.0",
                artifact_path="/tmp/model2.pkl",
            )

    def test_transition_stage(self, temp_registry_dir):
        registry = ModelRegistry(registry_path=temp_registry_dir)

        registry.register_model(
            name="model-b",
            version="2.0.0",
            artifact_path="/tmp/model.pkl",
        )

        result = registry.transition_stage(
            "model-b", "2.0.0", ModelStage.STAGING
        )

        assert result.stage == ModelStage.STAGING

    def test_production_auto_archives(self, temp_registry_dir):
        registry = ModelRegistry(registry_path=temp_registry_dir)

        registry.register_model(
            name="model-c", version="1.0.0", artifact_path="/tmp/v1.pkl"
        )
        registry.register_model(
            name="model-c", version="2.0.0", artifact_path="/tmp/v2.pkl"
        )

        registry.transition_stage(
            "model-c", "1.0.0", ModelStage.PRODUCTION
        )
        registry.transition_stage(
            "model-c", "2.0.0", ModelStage.PRODUCTION
        )

        prod = registry.get_production_model("model-c")
        assert prod is not None
        assert prod.version == "2.0.0"

        all_models = registry.list_models(name="model-c")
        archived = [
            m for m in all_models if m.stage == ModelStage.ARCHIVED
        ]
        assert len(archived) == 1
        assert archived[0].version == "1.0.0"

    def test_get_production_model(self, temp_registry_dir):
        registry = ModelRegistry(registry_path=temp_registry_dir)

        assert registry.get_production_model("nonexistent") is None

        registry.register_model(
            name="model-d", version="1.0.0", artifact_path="/tmp/m.pkl"
        )
        registry.transition_stage(
            "model-d", "1.0.0", ModelStage.PRODUCTION
        )

        prod = registry.get_production_model("model-d")
        assert prod is not None
        assert prod.version == "1.0.0"

    def test_list_models_with_filter(self, temp_registry_dir):
        registry = ModelRegistry(registry_path=temp_registry_dir)

        registry.register_model(
            name="model-e", version="1.0.0", artifact_path="/tmp/m1.pkl"
        )
        registry.register_model(
            name="model-e", version="2.0.0", artifact_path="/tmp/m2.pkl"
        )
        registry.transition_stage(
            "model-e", "2.0.0", ModelStage.STAGING
        )

        staging = registry.list_models(stage=ModelStage.STAGING)
        assert len(staging) == 1
        assert staging[0].version == "2.0.0"

    def test_delete_model(self, temp_registry_dir):
        registry = ModelRegistry(registry_path=temp_registry_dir)

        registry.register_model(
            name="model-f", version="1.0.0", artifact_path="/tmp/m.pkl"
        )
        registry.delete_model("model-f", "1.0.0")

        models = registry.list_models(name="model-f")
        assert len(models) == 0
