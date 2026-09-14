"""Default behavior-distance functions for the audit layer."""

from __future__ import annotations

import re
from typing import Any


class ExactMatchDistance:
    def __call__(self, left: Any, right: Any) -> float:
        return float(left != right)


class TokenJaccardDistance:
    """Distance in [0, 1]; useful only as a transparent default diagnostic."""

    def __call__(self, left: Any, right: Any) -> float:
        left_tokens = set(re.findall(r"\w+", str(left).lower()))
        right_tokens = set(re.findall(r"\w+", str(right).lower()))
        union = left_tokens | right_tokens
        if not union:
            return 0.0
        return 1.0 - len(left_tokens & right_tokens) / len(union)

