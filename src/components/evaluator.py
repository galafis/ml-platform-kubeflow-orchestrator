"""Model evaluation component with quality gate enforcement."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from src.config.settings import QualityGateConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Computed model evaluation metrics."""

    metrics: Dict[str, float]
    task_type: str
    num_samples: int
    passed_quality_gate: bool
    gate_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "task_type": self.task_type,
            "num_samples": self.num_samples,
            "passed_quality_gate": self.passed_quality_gate,
            "gate_failures": self.gate_failures,
        }


class ModelEvaluator:
    """Evaluates model performance and enforces quality gates.

    Computes classification or regression metrics and checks
    against configurable thresholds for automated promotion.
    """

    def __init__(
        self,
        task_type: str = "classification",
        quality_gate: Optional[QualityGateConfig] = None,
    ):
        self.task_type = task_type
        self.quality_gate = quality_gate or QualityGateConfig()

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_size_mb: Optional[float] = None,
        inference_latency_ms: Optional[float] = None,
    ) -> EvaluationMetrics:
        """Evaluate model predictions against ground truth.

        Args:
            y_true: Ground truth labels.
            y_pred: Model predictions.
            model_size_mb: Optional model artifact size in MB.
            inference_latency_ms: Optional inference latency in ms.

        Returns:
            EvaluationMetrics with computed metrics and gate status.
        """
        if self.task_type == "classification":
            metrics = self._classification_metrics(y_true, y_pred)
        else:
            metrics = self._regression_metrics(y_true, y_pred)

        if model_size_mb is not None:
            metrics["model_size_mb"] = model_size_mb
        if inference_latency_ms is not None:
            metrics["inference_latency_ms"] = inference_latency_ms

        gate_failures = self._check_quality_gate(metrics)
        passed = len(gate_failures) == 0

        result = EvaluationMetrics(
            metrics=metrics,
            task_type=self.task_type,
            num_samples=len(y_true),
            passed_quality_gate=passed,
            gate_failures=gate_failures,
        )

        status = "PASSED" if passed else "FAILED"
        logger.info(f"Model evaluation {status}: {metrics}")
        if gate_failures:
            logger.warning(f"Quality gate failures: {gate_failures}")

        return result

    def _classification_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        average = "binary"
        n_classes = len(np.unique(y_true))
        if n_classes > 2:
            average = "weighted"

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average=average, zero_division=0)
            ),
        }

    def _regression_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mse": float(mean_squared_error(y_true, y_pred)),
        }

    def _check_quality_gate(
        self, metrics: Dict[str, float]
    ) -> List[str]:
        """Check metrics against quality gate thresholds."""
        failures: List[str] = []
        gate = self.quality_gate

        if self.task_type == "classification":
            checks = {
                "accuracy": gate.min_accuracy,
                "precision": gate.min_precision,
                "recall": gate.min_recall,
                "f1_score": gate.min_f1_score,
            }
            for metric_name, threshold in checks.items():
                value = metrics.get(metric_name, 0.0)
                if value < threshold:
                    failures.append(
                        f"{metric_name}={value:.4f} < {threshold:.4f}"
                    )

        if "inference_latency_ms" in metrics:
            if metrics["inference_latency_ms"] > gate.max_latency_ms:
                failures.append(
                    f"latency={metrics['inference_latency_ms']:.1f}ms "
                    f"> {gate.max_latency_ms:.1f}ms"
                )

        if "model_size_mb" in metrics:
            if metrics["model_size_mb"] > gate.max_model_size_mb:
                failures.append(
                    f"model_size={metrics['model_size_mb']:.1f}MB "
                    f"> {gate.max_model_size_mb:.1f}MB"
                )

        return failures

    def compare_models(
        self,
        champion_metrics: EvaluationMetrics,
        challenger_metrics: EvaluationMetrics,
        primary_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare champion and challenger model metrics.

        Returns:
            Comparison results with recommendation.
        """
        if primary_metric is None:
            primary_metric = (
                "f1_score"
                if self.task_type == "classification"
                else "r2"
            )

        champion_val = champion_metrics.metrics.get(primary_metric, 0.0)
        challenger_val = challenger_metrics.metrics.get(primary_metric, 0.0)
        improvement = challenger_val - champion_val

        promote = (
            challenger_metrics.passed_quality_gate and improvement > 0
        )

        comparison = {
            "primary_metric": primary_metric,
            "champion_value": champion_val,
            "challenger_value": challenger_val,
            "improvement": improvement,
            "improvement_pct": (
                (improvement / champion_val * 100)
                if champion_val != 0
                else 0.0
            ),
            "challenger_passed_gate": challenger_metrics.passed_quality_gate,
            "recommend_promotion": promote,
        }

        if promote:
            logger.info(
                f"Recommending promotion: {primary_metric} "
                f"improved by {improvement:.4f} "
                f"({comparison['improvement_pct']:.1f}%)"
            )
        else:
            logger.info(
                f"Keeping champion: challenger {primary_metric}="
                f"{challenger_val:.4f} vs champion={champion_val:.4f}"
            )

        return comparison
