"""Data preprocessing component for ML pipelines."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Result from preprocessing operations."""

    dataframe: pd.DataFrame
    transformations_applied: List[str]
    columns_dropped: List[str]
    columns_encoded: List[str]
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]


class Preprocessor:
    """Data preprocessing pipeline with configurable transformations.

    Handles missing values, categorical encoding, scaling,
    and feature selection in a reproducible manner.
    """

    def __init__(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        scale_features: bool = True,
        drop_threshold: float = 0.5,
        high_cardinality_threshold: int = 50,
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.scale_features = scale_features
        self.drop_threshold = drop_threshold
        self.high_cardinality_threshold = high_cardinality_threshold

        self._numeric_imputer: Optional[SimpleImputer] = None
        self._categorical_imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> PreprocessingResult:
        """Fit preprocessor on data and transform it.

        Args:
            df: Raw input DataFrame.

        Returns:
            PreprocessingResult with transformed data and metadata.
        """
        original_shape = df.shape
        transformations: List[str] = []
        columns_dropped: List[str] = []
        columns_encoded: List[str] = []

        result_df = df.copy()

        # Drop columns with excessive missing values
        missing_ratios = result_df.isnull().mean()
        cols_to_drop = missing_ratios[
            missing_ratios > self.drop_threshold
        ].index.tolist()

        if cols_to_drop:
            result_df = result_df.drop(columns=cols_to_drop)
            columns_dropped.extend(cols_to_drop)
            transformations.append(
                f"Dropped {len(cols_to_drop)} columns "
                f"(>{self.drop_threshold:.0%} missing)"
            )
            logger.info(f"Dropped columns: {cols_to_drop}")

        # Separate numeric and categorical columns
        numeric_cols = result_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_cols = result_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Impute missing values in numeric columns
        if numeric_cols:
            self._numeric_imputer = SimpleImputer(
                strategy=self.numeric_strategy
            )
            result_df[numeric_cols] = self._numeric_imputer.fit_transform(
                result_df[numeric_cols]
            )
            transformations.append(
                f"Imputed {len(numeric_cols)} numeric columns "
                f"(strategy={self.numeric_strategy})"
            )

        # Impute and encode categorical columns
        if categorical_cols:
            self._categorical_imputer = SimpleImputer(
                strategy=self.categorical_strategy
            )
            result_df[categorical_cols] = (
                self._categorical_imputer.fit_transform(
                    result_df[categorical_cols]
                )
            )

            for col in categorical_cols:
                n_unique = result_df[col].nunique()
                if n_unique > self.high_cardinality_threshold:
                    result_df = result_df.drop(columns=[col])
                    columns_dropped.append(col)
                    logger.info(
                        f"Dropped high-cardinality column '{col}' "
                        f"({n_unique} unique values)"
                    )
                else:
                    encoder = LabelEncoder()
                    result_df[col] = encoder.fit_transform(
                        result_df[col].astype(str)
                    )
                    self._label_encoders[col] = encoder
                    columns_encoded.append(col)

            transformations.append(
                f"Encoded {len(columns_encoded)} categorical columns"
            )

        # Scale numeric features
        if self.scale_features and numeric_cols:
            valid_numeric = [
                c for c in numeric_cols if c in result_df.columns
            ]
            if valid_numeric:
                self._scaler = StandardScaler()
                result_df[valid_numeric] = self._scaler.fit_transform(
                    result_df[valid_numeric]
                )
                transformations.append(
                    f"Scaled {len(valid_numeric)} numeric features"
                )

        self._fitted = True

        logger.info(
            f"Preprocessing complete: {original_shape} -> {result_df.shape}"
        )

        return PreprocessingResult(
            dataframe=result_df,
            transformations_applied=transformations,
            columns_dropped=columns_dropped,
            columns_encoded=columns_encoded,
            original_shape=original_shape,
            final_shape=result_df.shape,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using previously fitted preprocessor.

        Args:
            df: New DataFrame to transform.

        Returns:
            Transformed DataFrame.

        Raises:
            RuntimeError: If preprocessor hasn't been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before calling transform(). "
                "Use fit_transform() first."
            )

        result_df = df.copy()
        numeric_cols = result_df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_cols = result_df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if self._numeric_imputer and numeric_cols:
            valid = [
                c
                for c in numeric_cols
                if c in self._numeric_imputer.feature_names_in_
            ]
            if valid:
                result_df[valid] = self._numeric_imputer.transform(
                    result_df[valid]
                )

        for col in categorical_cols:
            if col in self._label_encoders:
                encoder = self._label_encoders[col]
                result_df[col] = result_df[col].astype(str).map(
                    lambda x, e=encoder: (
                        e.transform([x])[0]
                        if x in e.classes_
                        else -1
                    )
                )

        if self._scaler and numeric_cols:
            valid = [
                c
                for c in numeric_cols
                if c in self._scaler.feature_names_in_
            ]
            if valid:
                result_df[valid] = self._scaler.transform(
                    result_df[valid]
                )

        return result_df
