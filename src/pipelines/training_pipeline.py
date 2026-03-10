"""Kubeflow training pipeline definition.

Orchestrates the full ML training workflow: data loading,
preprocessing, model training, evaluation, and conditional deployment.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.components.data_loader import DataLoader
from src.components.deployer import ModelDeployer
from src.components.evaluator import ModelEvaluator
from src.components.preprocessor import Preprocessor
from src.components.trainer import ModelTrainer
from src.config.settings import PlatformSettings, QualityGateConfig
from src.monitoring.metrics import MetricsCollector
from src.registry.model_registry import ModelRegistry, ModelStage
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineRunResult:
    """Result from a complete pipeline execution."""

    run_id: str
    status: str
    model_name: str
    model_version: str
    metrics: Dict[str, float]
    passed_quality_gate: bool
    deployed: bool
    artifact_path: Optional[str] = None


class TrainingPipeline:
    """End-to-end ML training pipeline with quality gate enforcement.

    Orchestrates data loading -> preprocessing -> training ->
    evaluation -> conditional deployment based on quality gates.
    """

    def __init__(
        self,
        settings: Optional[PlatformSettings] = None,
        model_name: str = "default-model",
        model_version: str = "1.0.0",
        task_type: str = "classification",
        algorithm: str = "gradient_boosting",
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        self.settings = settings or PlatformSettings()
        self.model_name = model_name
        self.model_version = model_version
        self.task_type = task_type

        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.trainer = ModelTrainer(
            task_type=task_type,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
        )
        self.evaluator = ModelEvaluator(
            task_type=task_type,
            quality_gate=self.settings.quality_gate,
        )
        self.deployer = ModelDeployer()
        self.registry = ModelRegistry()
        self.metrics = MetricsCollector()

    def run(
        self,
        data_path: str,
        target_column: str,
        auto_deploy: bool = True,
    ) -> PipelineRunResult:
        """Execute the full training pipeline.

        Args:
            data_path: Path to the training data file.
            target_column: Name of the target variable column.
            auto_deploy: Whether to auto-deploy on quality gate pass.

        Returns:
            PipelineRunResult with execution details.
        """
        import uuid

        run_id = str(uuid.uuid4())[:8]
        logger.info(f"Pipeline run {run_id} started for {self.model_name}")

        with self.metrics.track_duration(
            "pipeline_execution",
            labels={"pipeline": "training", "model": self.model_name},
        ):
            # Step 1: Load data
            logger.info(f"[{run_id}] Step 1/5: Loading data")
            splits = self.data_loader.load_splits(
                data_path, target_column
            )
            X_train, y_train = splits["train"]
            X_test, y_test = splits["test"]

            # Step 2: Preprocess
            logger.info(f"[{run_id}] Step 2/5: Preprocessing")
            prep_result = self.preprocessor.fit_transform(X_train)
            X_train_processed = prep_result.dataframe
            X_test_processed = self.preprocessor.transform(X_test)

            # Align columns after preprocessing
            common_cols = list(
                set(X_train_processed.columns)
                & set(X_test_processed.columns)
            )
            X_train_processed = X_train_processed[common_cols]
            X_test_processed = X_test_processed[common_cols]

            # Step 3: Train
            logger.info(f"[{run_id}] Step 3/5: Training model")
            training_result = self.trainer.train(
                X_train_processed, y_train
            )

            # Step 4: Evaluate
            logger.info(f"[{run_id}] Step 4/5: Evaluating model")
            predictions = self.trainer.predict(X_test_processed)
            eval_result = self.evaluator.evaluate(y_test, predictions)

            self.metrics.record_model_metrics(
                self.model_name,
                self.model_version,
                eval_result.metrics,
            )

            # Step 5: Conditional deployment
            deployed = False
            artifact_path = None

            if eval_result.passed_quality_gate and auto_deploy:
                logger.info(f"[{run_id}] Step 5/5: Deploying model")

                deploy_result = self.deployer.deploy(
                    model=training_result.model,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    metrics=eval_result.metrics,
                    hyperparameters=training_result.hyperparameters,
                )
                artifact_path = deploy_result.artifact_path

                self.registry.register_model(
                    name=self.model_name,
                    version=self.model_version,
                    artifact_path=artifact_path,
                    metrics=eval_result.metrics,
                    hyperparameters=training_result.hyperparameters,
                    description=(
                        f"Auto-deployed by training pipeline (run {run_id})"
                    ),
                )
                self.registry.transition_stage(
                    self.model_name,
                    self.model_version,
                    ModelStage.PRODUCTION,
                )

                deployed = True
            elif not eval_result.passed_quality_gate:
                logger.warning(
                    f"[{run_id}] Quality gate FAILED: "
                    f"{eval_result.gate_failures}"
                )
            else:
                logger.info(
                    f"[{run_id}] auto_deploy=False, skipping deployment"
                )

        status = "success" if eval_result.passed_quality_gate else "failed_gate"
        self.metrics.record_pipeline_run(
            "training", status, training_result.training_time_seconds
        )

        result = PipelineRunResult(
            run_id=run_id,
            status=status,
            model_name=self.model_name,
            model_version=self.model_version,
            metrics=eval_result.metrics,
            passed_quality_gate=eval_result.passed_quality_gate,
            deployed=deployed,
            artifact_path=artifact_path,
        )

        logger.info(
            f"Pipeline run {run_id} completed: status={status}, "
            f"deployed={deployed}"
        )

        return result
