"""Unit tests for pipeline components."""

import numpy as np
import pandas as pd
import pytest

from src.components.data_loader import DataLoader
from src.components.evaluator import ModelEvaluator
from src.components.preprocessor import Preprocessor
from src.components.trainer import ModelTrainer
from src.config.settings import QualityGateConfig


class TestDataLoader:
    def test_load_csv(self, sample_csv_path):
        loader = DataLoader()
        result = loader.load(sample_csv_path)

        assert result.num_rows == 200
        assert result.num_columns == 6
        assert "target" in result.column_types

    def test_load_parquet(self, sample_parquet_path):
        loader = DataLoader()
        result = loader.load(sample_parquet_path)

        assert result.num_rows == 200

    def test_load_unsupported_format(self, tmp_path):
        bad_path = tmp_path / "data.xlsx"
        bad_path.touch()

        loader = DataLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(str(bad_path))

    def test_load_splits(self, sample_csv_path):
        loader = DataLoader()
        splits = loader.load_splits(
            sample_csv_path,
            target_column="target",
            test_size=0.2,
            validation_size=0.1,
        )

        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits

        X_train, y_train = splits["train"]
        assert len(X_train) == len(y_train)
        assert "target" not in X_train.columns

    def test_load_with_sampling(self, sample_csv_path):
        loader = DataLoader(sample_size=50)
        result = loader.load(sample_csv_path)

        assert result.num_rows == 50

    def test_summary(self, sample_csv_path):
        loader = DataLoader()
        result = loader.load(sample_csv_path)
        summary = result.summary()

        assert "num_rows" in summary
        assert "num_columns" in summary


class TestPreprocessor:
    def test_fit_transform(self, sample_classification_data):
        df = sample_classification_data.drop(columns=["target"])
        preprocessor = Preprocessor()

        result = preprocessor.fit_transform(df)

        assert result.dataframe.shape[0] == df.shape[0]
        assert len(result.transformations_applied) > 0
        assert result.original_shape == df.shape

    def test_transform_requires_fit(self, sample_classification_data):
        df = sample_classification_data.drop(columns=["target"])
        preprocessor = Preprocessor()

        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(df)

    def test_drops_high_missing_columns(self, tmp_path):
        df = pd.DataFrame({
            "good_col": [1, 2, 3, 4, 5],
            "bad_col": [None, None, None, None, 1],
            "numeric": [10, 20, 30, 40, 50],
        })

        preprocessor = Preprocessor(drop_threshold=0.5)
        result = preprocessor.fit_transform(df)

        assert "bad_col" in result.columns_dropped

    def test_scaling(self, sample_classification_data):
        df = sample_classification_data[["feature_1", "feature_2"]]
        preprocessor = Preprocessor(scale_features=True)

        result = preprocessor.fit_transform(df)
        scaled = result.dataframe

        assert abs(scaled["feature_1"].mean()) < 0.1
        assert abs(scaled["feature_1"].std() - 1.0) < 0.2


class TestModelTrainer:
    def test_classification_training(self, sample_classification_data):
        df = sample_classification_data
        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["target"]

        trainer = ModelTrainer(
            task_type="classification",
            algorithm="gradient_boosting",
        )
        result = trainer.train(X, y)

        assert result.model is not None
        assert result.task_type == "classification"
        assert 0 <= result.cv_mean <= 1
        assert result.training_time_seconds > 0

    def test_regression_training(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
        })
        y = pd.Series(X["x1"] * 2 + X["x2"] + np.random.randn(100) * 0.1)

        trainer = ModelTrainer(
            task_type="regression",
            algorithm="gradient_boosting",
        )
        result = trainer.train(X, y)

        assert result.model is not None
        assert result.task_type == "regression"

    def test_feature_importances(self, sample_classification_data):
        df = sample_classification_data
        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["target"]

        trainer = ModelTrainer(algorithm="gradient_boosting")
        result = trainer.train(X, y)

        assert result.feature_importances is not None
        assert len(result.feature_importances) == 3

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="not found"):
            ModelTrainer(algorithm="nonexistent")

    def test_predict(self, sample_classification_data):
        df = sample_classification_data
        X = df[["feature_1", "feature_2", "feature_3"]]
        y = df["target"]

        trainer = ModelTrainer()
        trainer.train(X, y)
        predictions = trainer.predict(X)

        assert len(predictions) == len(X)


class TestModelEvaluator:
    def test_classification_evaluation(self):
        y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])

        evaluator = ModelEvaluator(task_type="classification")
        result = evaluator.evaluate(y_true, y_pred)

        assert "accuracy" in result.metrics
        assert "precision" in result.metrics
        assert "recall" in result.metrics
        assert "f1_score" in result.metrics
        assert result.num_samples == 8

    def test_quality_gate_pass(self):
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])

        gate = QualityGateConfig(
            min_accuracy=0.8,
            min_precision=0.8,
            min_recall=0.8,
            min_f1_score=0.8,
        )
        evaluator = ModelEvaluator(quality_gate=gate)
        result = evaluator.evaluate(y_true, y_pred)

        assert result.passed_quality_gate is True
        assert len(result.gate_failures) == 0

    def test_quality_gate_fail(self):
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 0])

        gate = QualityGateConfig(min_accuracy=0.9)
        evaluator = ModelEvaluator(quality_gate=gate)
        result = evaluator.evaluate(y_true, y_pred)

        assert result.passed_quality_gate is False
        assert len(result.gate_failures) > 0

    def test_compare_models(self):
        evaluator = ModelEvaluator()

        from src.components.evaluator import EvaluationMetrics

        champion = EvaluationMetrics(
            metrics={"f1_score": 0.85, "accuracy": 0.87},
            task_type="classification",
            num_samples=100,
            passed_quality_gate=True,
        )
        challenger = EvaluationMetrics(
            metrics={"f1_score": 0.90, "accuracy": 0.91},
            task_type="classification",
            num_samples=100,
            passed_quality_gate=True,
        )

        result = evaluator.compare_models(champion, challenger)

        assert result["recommend_promotion"] is True
        assert result["improvement"] > 0
