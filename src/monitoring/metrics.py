"""Prometheus metrics collection for ML platform monitoring."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricRecord:
    """A single metric data point."""

    name: str
    value: float
    labels: Dict[str, str]
    timestamp: str
    metric_type: str  # counter, gauge, histogram


class MetricsCollector:
    """Collects and exposes ML platform metrics.

    Tracks pipeline execution, model performance, and
    infrastructure health for Prometheus scraping.
    """

    def __init__(self, namespace: str = "ml_platform"):
        self.namespace = namespace
        self._metrics: List[MetricRecord] = []
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        full_name = f"{self.namespace}_{name}"
        self._counters[full_name] = (
            self._counters.get(full_name, 0) + value
        )
        self._record(full_name, self._counters[full_name], labels or {}, "counter")

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric to a specific value."""
        full_name = f"{self.namespace}_{name}"
        self._gauges[full_name] = value
        self._record(full_name, value, labels or {}, "gauge")

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an observation in a histogram metric."""
        full_name = f"{self.namespace}_{name}"
        if full_name not in self._histograms:
            self._histograms[full_name] = []
        self._histograms[full_name].append(value)
        self._record(full_name, value, labels or {}, "histogram")

    @contextmanager
    def track_duration(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Generator[None, None, None]:
        """Context manager to track operation duration.

        Usage:
            with metrics.track_duration("pipeline_execution"):
                run_pipeline()
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe_histogram(
                f"{name}_duration_seconds", duration, labels
            )
            logger.info(f"{name} completed in {duration:.3f}s")

    def record_pipeline_run(
        self,
        pipeline_name: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a pipeline execution metric."""
        labels = {
            "pipeline": pipeline_name,
            "status": status,
        }
        self.increment_counter("pipeline_runs_total", labels=labels)
        self.observe_histogram(
            "pipeline_duration_seconds", duration_seconds, labels
        )
        self.set_gauge(
            "pipeline_last_run_timestamp",
            time.time(),
            {"pipeline": pipeline_name},
        )

    def record_model_metrics(
        self,
        model_name: str,
        model_version: str,
        metrics: Dict[str, float],
    ) -> None:
        """Record model performance metrics."""
        for metric_name, value in metrics.items():
            self.set_gauge(
                f"model_{metric_name}",
                value,
                {"model": model_name, "version": model_version},
            )

    def get_prometheus_output(self) -> str:
        """Generate Prometheus-compatible text exposition format."""
        lines: List[str] = []
        seen_metrics = set()

        for record in self._metrics:
            if record.name not in seen_metrics:
                lines.append(
                    f"# TYPE {record.name} {record.metric_type}"
                )
                seen_metrics.add(record.name)

            label_str = ",".join(
                f'{k}="{v}"' for k, v in record.labels.items()
            )
            if label_str:
                lines.append(
                    f"{record.name}{{{label_str}}} {record.value}"
                )
            else:
                lines.append(f"{record.name} {record.value}")

        return "\n".join(lines)

    def _record(
        self,
        name: str,
        value: float,
        labels: Dict[str, str],
        metric_type: str,
    ) -> None:
        """Store a metric record internally."""
        self._metrics.append(
            MetricRecord(
                name=name,
                value=value,
                labels=labels,
                timestamp=datetime.utcnow().isoformat(),
                metric_type=metric_type,
            )
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histogram_counts": {
                k: len(v) for k, v in self._histograms.items()
            },
            "total_records": len(self._metrics),
        }
