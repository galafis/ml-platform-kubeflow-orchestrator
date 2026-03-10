"""Deployment pipeline for model promotion and rollback operations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.components.deployer import DeploymentResult, ModelDeployer
from src.registry.model_registry import ModelRegistry, ModelStage, RegisteredModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PromotionResult:
    """Result from a model promotion operation."""

    model_name: str
    version: str
    previous_version: Optional[str]
    new_stage: str
    deployment: Optional[DeploymentResult]
    success: bool
    message: str


class DeploymentPipeline:
    """Manages model promotion, deployment, and rollback workflows.

    Provides safe promotion with validation and automatic
    rollback capabilities.
    """

    def __init__(
        self,
        artifact_dir: str = "./artifacts",
        deployment_target: str = "local",
    ):
        self.registry = ModelRegistry()
        self.deployer = ModelDeployer(
            artifact_dir=artifact_dir,
            deployment_target=deployment_target,
        )

    def promote_to_staging(
        self, model_name: str, version: str
    ) -> PromotionResult:
        """Promote a model version to staging.

        Args:
            model_name: Name of the model.
            version: Version to promote.

        Returns:
            PromotionResult with promotion details.
        """
        logger.info(
            f"Promoting {model_name} v{version} to staging"
        )

        try:
            self.registry.transition_stage(
                model_name, version, ModelStage.STAGING
            )

            return PromotionResult(
                model_name=model_name,
                version=version,
                previous_version=None,
                new_stage=ModelStage.STAGING.value,
                deployment=None,
                success=True,
                message=f"Successfully promoted to staging",
            )
        except (KeyError, ValueError) as e:
            return PromotionResult(
                model_name=model_name,
                version=version,
                previous_version=None,
                new_stage="failed",
                deployment=None,
                success=False,
                message=str(e),
            )

    def promote_to_production(
        self, model_name: str, version: str
    ) -> PromotionResult:
        """Promote a model version to production.

        Records the previous production version for potential rollback.

        Args:
            model_name: Name of the model.
            version: Version to promote.

        Returns:
            PromotionResult with promotion details.
        """
        current_prod = self.registry.get_production_model(model_name)
        previous_version = (
            current_prod.version if current_prod else None
        )

        logger.info(
            f"Promoting {model_name} v{version} to production "
            f"(replacing v{previous_version})"
        )

        try:
            self.registry.transition_stage(
                model_name, version, ModelStage.PRODUCTION
            )

            return PromotionResult(
                model_name=model_name,
                version=version,
                previous_version=previous_version,
                new_stage=ModelStage.PRODUCTION.value,
                deployment=None,
                success=True,
                message=(
                    f"Promoted to production "
                    f"(previous: v{previous_version})"
                ),
            )
        except (KeyError, ValueError) as e:
            return PromotionResult(
                model_name=model_name,
                version=version,
                previous_version=previous_version,
                new_stage="failed",
                deployment=None,
                success=False,
                message=str(e),
            )

    def rollback(
        self, model_name: str, target_version: str
    ) -> PromotionResult:
        """Rollback to a previous model version.

        Args:
            model_name: Name of the model.
            target_version: Version to rollback to.

        Returns:
            PromotionResult with rollback details.
        """
        current_prod = self.registry.get_production_model(model_name)
        current_version = (
            current_prod.version if current_prod else None
        )

        logger.info(
            f"Rolling back {model_name} from v{current_version} "
            f"to v{target_version}"
        )

        try:
            rollback_result = self.deployer.rollback(
                model_name, target_version
            )

            self.registry.transition_stage(
                model_name, target_version, ModelStage.PRODUCTION
            )

            return PromotionResult(
                model_name=model_name,
                version=target_version,
                previous_version=current_version,
                new_stage=ModelStage.PRODUCTION.value,
                deployment=rollback_result,
                success=True,
                message=(
                    f"Rolled back to v{target_version} "
                    f"from v{current_version}"
                ),
            )
        except (FileNotFoundError, KeyError) as e:
            return PromotionResult(
                model_name=model_name,
                version=target_version,
                previous_version=current_version,
                new_stage="failed",
                deployment=None,
                success=False,
                message=f"Rollback failed: {e}",
            )

    def get_deployment_history(
        self, model_name: str
    ) -> List[Dict[str, Any]]:
        """Get the deployment history for a model."""
        models = self.registry.list_models(name=model_name)

        history = []
        for model in models:
            history.append({
                "version": model.version,
                "stage": model.stage.value,
                "registered_at": model.registered_at,
                "metrics": model.metrics,
            })

        return sorted(
            history,
            key=lambda x: x["registered_at"],
            reverse=True,
        )
