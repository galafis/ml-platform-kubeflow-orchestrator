"""Shared test fixtures for the ML platform test suite."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_data() -> pd.DataFrame:
    """Generate a synthetic classification dataset."""
    np.random.seed(42)
    n_samples = 200

    data = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples) * 2,
        "feature_3": np.random.uniform(0, 10, n_samples),
        "category_a": np.random.choice(["low", "medium", "high"], n_samples),
        "category_b": np.random.choice(["type_x", "type_y"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    })

    return data


@pytest.fixture
def sample_csv_path(sample_classification_data, tmp_path) -> str:
    """Write sample data to a temporary CSV file."""
    path = tmp_path / "sample_data.csv"
    sample_classification_data.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_parquet_path(sample_classification_data, tmp_path) -> str:
    """Write sample data to a temporary Parquet file."""
    path = tmp_path / "sample_data.parquet"
    sample_classification_data.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def temp_artifact_dir(tmp_path) -> str:
    """Provide a temporary directory for model artifacts."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    return str(artifact_dir)


@pytest.fixture
def temp_registry_dir(tmp_path) -> str:
    """Provide a temporary directory for model registry."""
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    return str(registry_dir)
