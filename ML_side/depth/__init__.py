"""Depth proxy helpers extracted from the cohort-2 notebook."""

from .depth_estimator import (
    baseline_depth_score,
    estimate_depth,
    improved_depth_score,
)

__all__ = [
    "baseline_depth_score",
    "estimate_depth",
    "improved_depth_score",
]
