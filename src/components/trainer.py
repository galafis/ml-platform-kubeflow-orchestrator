"""Model training component with experiment tracking integration."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASSIFIER_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}

REGRESSOR_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "ridge": Ridge,
}


@dataclass
class TrainingResult:
    """Result from model training."""

    model: Any
    model_type: str
    task_type: str
    hyperparameters: Dict[str, Any]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time_seconds: float
    feature_importances: Optional[Dict[str, float]] = None


class ModelTrainer:
    """Trains ML models with cross-validation and experiment tracking.

    Supports classification and regression tasks with multiple
    algorithm options and automated hyperparameter handling.
    """

    def __init__(
        self,
        task_type: str = "classification",
        algorithm: str = "gradient_boosting",
        hyperparameters: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        random_state: int = 42,
    ):
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got '{task_type}'"
            )

        self.task_type = task_type
        self.algorithm = algorithm
        self.cv_folds = cv_folds
        self.random_state = random_state

        if scoring is None:
            self.scoring = (
                "accuracy" if task_type == "classification" else "r2"
            )
        else:
            self.scoring = scoring

        registry = (
            CLASSIFIER_REGISTRY
            if task_type == "classification"
            else REGRESSOR_REGISTRY
        )

        if algorithm not in registry:
            raise ValueError(
                f"Algorithm '{algorithm}' not found. "
                f"Available: {list(registry.keys())}"
            )

        default_params = {"random_state": random_state}
        if hyperparameters:
            default_params.update(hyperparameters)

        self._model_class = registry[algorithm]
        self._hyperparameters = default_params
        self._model = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> TrainingResult:
        """Train the model with cross-validation.

        Args:
            X_train: Training features.
            y_train: Training target.

        Returns:
            TrainingResult with model and performance metrics.
        """
        logger.info(
            f"Training {self.algorithm} ({self.task_type}) "
            f"on {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features"
        )

        start_time = time.time()

        self._model = self._model_class(**self._hyperparameters)

        cv_scores = cross_val_score(
            self._model,
            X_train,
            y_train,
            cv=self.cv_folds,
            scoring=self.scoring,
        )

        self._model.fit(X_train, y_train)

        training_time = time.time() - start_time

        feature_importances = self._extract_feature_importances(
            X_train.columns.tolist()
        )

        result = TrainingResult(
            model=self._model,
            model_type=self.algorithm,
            task_type=self.task_type,
            hyperparameters=self._hyperparameters,
            cv_scores=cv_scores.tolist(),
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            training_time_seconds=round(training_time, 2),
            feature_importances=feature_importances,
        )

        logger.info(
            f"Training complete: {self.scoring}="
            f"{result.cv_mean:.4f} (+/- {result.cv_std:.4f}), "
            f"time={result.training_time_seconds}s"
        )

        return result

    def predict(
        self, X: pd.DataFrame
    ) -> np.ndarray:
        """Generate predictions using the trained model."""
        if self._model is None:
            raise RuntimeError("Model must be trained before prediction")
        return self._model.predict(X)

    def predict_proba(
        self, X: pd.DataFrame
    ) -> Optional[np.ndarray]:
        """Generate probability predictions (classification only)."""
        if self._model is None:
            raise RuntimeError("Model must be trained before prediction")
        if self.task_type != "classification":
            return None
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def _extract_feature_importances(
        self, feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """Extract feature importances if available."""
        if self._model is None:
            return None
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
            return dict(
                sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        if hasattr(self._model, "coef_"):
            coefs = np.abs(self._model.coef_).flatten()
            if len(coefs) == len(feature_names):
                importance_dict = dict(zip(feature_names, coefs))
                return dict(
                    sorted(
                        importance_dict.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )
        return None
