"""Small dependency-free statistics helpers used by causal audits."""

from __future__ import annotations

import math
from typing import Iterable

from .types import Estimate


def summarize(values: Iterable[float], confidence_z: float = 1.96) -> Estimate:
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("at least one sample is required")
    if any(not math.isfinite(value) for value in samples):
        raise ValueError("samples must be finite")
    count = len(samples)
    mean = sum(samples) / count
    if count == 1:
        stddev = 0.0
    else:
        stddev = math.sqrt(
            sum((value - mean) ** 2 for value in samples) / (count - 1)
        )
    stderr = stddev / math.sqrt(count)
    half_width = confidence_z * stderr
    return Estimate(
        mean=mean,
        stddev=stddev,
        stderr=stderr,
        ci_low=mean - half_width,
        ci_high=mean + half_width,
        count=count,
    )

