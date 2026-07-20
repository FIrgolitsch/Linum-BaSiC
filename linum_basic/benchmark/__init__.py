"""Benchmark and auto-strategy utilities for linum-basic.

This package groups the strategy-resolution helpers consumed by
:mod:`linum_basic.fit` and :mod:`linum_basic._working_size`, and the
quality-metric helpers consumed by :mod:`linum_basic.tuning`. Only the
strategy resolver and the quality-metric modules are exposed at the
package level in this pull request; additional harness modules are added
by later pull requests in the chain.
"""

from linum_basic.benchmark.quality import (
    CALIBRATION_POLICY,
    FIRST_CLASS_METRICS,
    METRIC_DEFINITION_VERSION,
    MetricDelta,
    PerZMetricRow,
    QualityReport,
    QualityVerdict,
    ToleranceSpec,
    calibrate_tolerances,
    compute_deltas,
    compute_quality_report,
    evaluate_quality_gate,
)
from linum_basic.benchmark.strategies import (
    ALLOWED_OVERRIDE_KEYS,
    BASELINE_REFERENCE_ID,
    BATCHED_CUDA_WS_GUARD,
    BUILTIN_STRATEGIES,
    DEFAULT_PRODUCTION_WORKING_SIZE,
    EVIDENCE_POLICY_ID,
    OVERRIDE_SOURCES,
    AutoStrategyResult,
    StrategyResult,
    WorkloadContext,
    build_workload_context,
    estimate_strategy_vram_bytes,
    load_overrides,
    resolve_auto_strategy,
    resolve_strategy,
)

__all__ = [
    "ALLOWED_OVERRIDE_KEYS",
    "BASELINE_REFERENCE_ID",
    "BATCHED_CUDA_WS_GUARD",
    "BUILTIN_STRATEGIES",
    "CALIBRATION_POLICY",
    "DEFAULT_PRODUCTION_WORKING_SIZE",
    "EVIDENCE_POLICY_ID",
    "FIRST_CLASS_METRICS",
    "METRIC_DEFINITION_VERSION",
    "OVERRIDE_SOURCES",
    "AutoStrategyResult",
    "MetricDelta",
    "PerZMetricRow",
    "QualityReport",
    "QualityVerdict",
    "StrategyResult",
    "ToleranceSpec",
    "WorkloadContext",
    "build_workload_context",
    "calibrate_tolerances",
    "compute_deltas",
    "compute_quality_report",
    "estimate_strategy_vram_bytes",
    "evaluate_quality_gate",
    "load_overrides",
    "resolve_auto_strategy",
    "resolve_strategy",
]

__version__ = "0.1.0"
