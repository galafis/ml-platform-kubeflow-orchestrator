"""Standalone evaluation pipeline for model comparison and promotion."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.components.deployer import ModelDeployer
from src.components.evaluator import EvaluationMetrics, ModelEvaluator
from src.config.settings import PlatformSettings
from src.registry.model_registry import ModelRegistry, ModelStage
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Result from champion vs challenger comparison."""

    champion_version: str
    challenger_version: str
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    primary_metric: str
    improvement: float
    recommend_promotion: bool


class EvaluationPipeline:
    """Pipeline for evaluating and comparing model versions.

    Supports champion-challenger comparisons with automated
    promotion based on performance improvement.
    """

    def __init__(
        self,
        model_name: str,
        task_type: str = "classification",
        settings: Optional[PlatformSettings] = None,
    ):
        self.model_name = model_name
        self.task_type = task_type
        self.settings = settings or PlatformSettings()

        self.evaluator = ModelEvaluator(
            task_type=task_type,
            quality_gate=self.settings.quality_gate,
        )
        self.registry = ModelRegistry()
        self.deployer = ModelDeployer()

    def evaluate_model(
        self,
        model_version: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> EvaluationMetrics:
        """Evaluate a specific model version on test data.

        Args:
            model_version: Version string of the model to evaluate.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            EvaluationMetrics with computed metrics.
        """
        logger.info(
            f"Evaluating {self.model_name} v{model_version}"
        )

        model = self.deployer.load_model(self.model_name, model_version)
        predictions = model.predict(X_test)

        return self.evaluator.evaluate(y_test, predictions)

    def compare_champion_challenger(
        self,
        challenger_version: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        primary_metric: Optional[str] = None,
        auto_promote: bool = False,
    ) -> ComparisonResult:
        """Compare current production model against a challenger.

        Args:
            challenger_version: Version of the challenger model.
            X_test: Test features.
            y_test: Test labels.
            primary_metric: Metric to use for comparison.
            auto_promote: Auto-promote if challenger wins.

        Returns:
            ComparisonResult with comparison details.
        """
        if primary_metric is None:
            primary_metric = (
                "f1_score" if self.task_type == "classification" else "r2"
            )

        # Get current champion
        champion = self.registry.get_production_model(self.model_name)
        if champion is None:
            logger.info("No production model found, promoting challenger")
            challenger_eval = self.evaluate_model(
                challenger_version, X_test, y_test
            )

            if auto_promote and challenger_eval.passed_quality_gate:
                self.registry.transition_stage(
                    self.model_name,
                    challenger_version,
                    ModelStage.PRODUCTION,
                )

            return ComparisonResult(
                champion_version="none",
                challenger_version=challenger_version,
                champion_metrics={},
                challenger_metrics=challenger_eval.metrics,
                primary_metric=primary_metric,
                improvement=challenger_eval.metrics.get(primary_metric, 0),
                recommend_promotion=challenger_eval.passed_quality_gate,
            )

        # Evaluate both models
        champion_eval = self.evaluate_model(
            champion.version, X_test, y_test
        )
        challenger_eval = self.evaluate_model(
            challenger_version, X_test, y_test
        )

        comparison = self.evaluator.compare_models(
            champion_eval, challenger_eval, primary_metric
        )

        if auto_promote and comparison["recommend_promotion"]:
            self.registry.transition_stage(
                self.model_name,
                challenger_version,
                ModelStage.PRODUCTION,
            )
            logger.info(
                f"Auto-promoted {self.model_name} v{challenger_version} "
                f"to production"
            )

        return ComparisonResult(
            champion_version=champion.version,
            challenger_version=challenger_version,
            champion_metrics=champion_eval.metrics,
            challenger_metrics=challenger_eval.metrics,
            primary_metric=primary_metric,
            improvement=comparison["improvement"],
            recommend_promotion=comparison["recommend_promotion"],
        )

    def batch_evaluate(
        self,
        versions: List[str],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> List[Dict[str, Any]]:
        """Evaluate multiple model versions and rank them.

        Args:
            versions: List of model versions to evaluate.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Ranked list of evaluation results.
        """
        results = []

        for version in versions:
            try:
                eval_result = self.evaluate_model(version, X_test, y_test)
                results.append({
                    "version": version,
                    "metrics": eval_result.metrics,
                    "passed_gate": eval_result.passed_quality_gate,
                    "gate_failures": eval_result.gate_failures,
                })
            except FileNotFoundError:
                logger.warning(
                    f"Model {self.model_name} v{version} not found, skipping"
                )

        ranking_metric = (
            "f1_score" if self.task_type == "classification" else "r2"
        )
        results.sort(
            key=lambda x: x["metrics"].get(ranking_metric, 0),
            reverse=True,
        )

        return results
