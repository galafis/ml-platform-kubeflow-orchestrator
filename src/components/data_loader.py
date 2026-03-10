"""Data loading component for Kubeflow pipelines."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataLoadResult:
    """Result from data loading operation."""

    dataframe: pd.DataFrame
    num_rows: int
    num_columns: int
    column_types: Dict[str, str]
    missing_values: Dict[str, int]
    load_path: str

    def summary(self) -> Dict[str, Any]:
        return {
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "missing_total": sum(self.missing_values.values()),
            "load_path": self.load_path,
        }


class DataLoader:
    """Handles data ingestion from multiple sources for ML pipelines.

    Supports CSV, Parquet, and JSON formats with automatic schema
    detection and basic validation.
    """

    SUPPORTED_FORMATS = {".csv", ".parquet", ".json", ".jsonl"}

    def __init__(
        self,
        validate_schema: bool = True,
        max_missing_ratio: float = 0.3,
        sample_size: Optional[int] = None,
    ):
        self.validate_schema = validate_schema
        self.max_missing_ratio = max_missing_ratio
        self.sample_size = sample_size

    def load(self, path: str) -> DataLoadResult:
        """Load data from a file path.

        Args:
            path: Path to the data file (CSV, Parquet, or JSON).

        Returns:
            DataLoadResult with the loaded DataFrame and metadata.

        Raises:
            ValueError: If format is unsupported or validation fails.
        """
        file_path = Path(path)
        suffix = file_path.suffix.lower()

        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{suffix}'. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )

        logger.info(f"Loading data from {path}")

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in (".json", ".jsonl"):
            df = pd.read_json(path, lines=(suffix == ".jsonl"))
        else:
            raise ValueError(f"Unhandled format: {suffix}")

        if self.sample_size and len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
            logger.info(f"Sampled {self.sample_size} rows from dataset")

        missing = df.isnull().sum().to_dict()
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        if self.validate_schema:
            self._validate(df, missing)

        result = DataLoadResult(
            dataframe=df,
            num_rows=len(df),
            num_columns=len(df.columns),
            column_types=column_types,
            missing_values={k: int(v) for k, v in missing.items()},
            load_path=str(path),
        )

        logger.info(
            f"Loaded {result.num_rows} rows, "
            f"{result.num_columns} columns from {path}"
        )
        return result

    def _validate(
        self, df: pd.DataFrame, missing: Dict[str, Any]
    ) -> None:
        """Run basic data quality checks."""
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        for col, count in missing.items():
            ratio = count / len(df)
            if ratio > self.max_missing_ratio:
                logger.warning(
                    f"Column '{col}' has {ratio:.1%} missing values "
                    f"(threshold: {self.max_missing_ratio:.1%})"
                )

    def load_splits(
        self,
        path: str,
        target_column: str,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Load data and split into train/validation/test sets.

        Returns:
            Dictionary with 'train', 'validation', 'test' keys,
            each containing (features, target) tuple.
        """
        result = self.load(path)
        df = result.dataframe

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        np.random.seed(random_state)
        indices = np.random.permutation(len(df))

        test_count = int(len(df) * test_size)
        val_count = int(len(df) * validation_size)
        train_count = len(df) - test_count - val_count

        train_idx = indices[:train_count]
        val_idx = indices[train_count : train_count + val_count]
        test_idx = indices[train_count + val_count :]

        features_cols = [c for c in df.columns if c != target_column]

        splits = {
            "train": (
                df.iloc[train_idx][features_cols],
                df.iloc[train_idx][target_column],
            ),
            "validation": (
                df.iloc[val_idx][features_cols],
                df.iloc[val_idx][target_column],
            ),
            "test": (
                df.iloc[test_idx][features_cols],
                df.iloc[test_idx][target_column],
            ),
        }

        logger.info(
            f"Split data: train={train_count}, "
            f"val={val_count}, test={test_count}"
        )
        return splits
