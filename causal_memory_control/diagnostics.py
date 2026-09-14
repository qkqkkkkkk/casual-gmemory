"""Zero/new-rollout diagnostics for existing P2 observations."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .types import MemoryUseEvent

if False:  # pragma: no cover - imports only for static type checkers
    from .oracle import OracleEvaluation


@dataclass(frozen=True)
class PivotalityObservation:
    event_id: str
    other_agent_answers: tuple[bool, ...]
    mismatch: bool

    @property
    def normalized_vote_margin(self) -> float:
        if not self.other_agent_answers:
            return 0.0
        yes = sum(self.other_agent_answers)
        no = len(self.other_agent_answers) - yes
        return abs(yes - no) / len(self.other_agent_answers)


@dataclass(frozen=True)
class PivotalityBin:
    label: str
    lower_exclusive: float
    upper_inclusive: float
    count: int
    mismatch_count: int
    mismatch_rate: float


def stratify_pivotality(
    observations: Sequence[PivotalityObservation],
    *,
    cutoffs: Sequence[float] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0),
) -> tuple[PivotalityBin, ...]:
    """Stratify mismatch rate by the other agents' normalized vote margin."""

    edges = tuple(sorted(set(float(value) for value in cutoffs)))
    if not edges or edges[0] < 0 or edges[-1] != 1.0:
        raise ValueError("cutoffs must be in [0,1] and end at 1.0")
    bins: list[list[PivotalityObservation]] = [[] for _ in edges]
    for observation in observations:
        margin = observation.normalized_vote_margin
        index = next(i for i, upper in enumerate(edges) if margin <= upper)
        bins[index].append(observation)
    results = []
    lower = -1.0
    for upper, rows in zip(edges, bins):
        mismatch_count = sum(row.mismatch for row in rows)
        label = "tie" if upper == 0.0 else f"({max(lower, 0.0):.3f},{upper:.3f}]"
        results.append(
            PivotalityBin(
                label=label,
                lower_exclusive=lower,
                upper_inclusive=upper,
                count=len(rows),
                mismatch_count=mismatch_count,
                mismatch_rate=mismatch_count / len(rows) if rows else 0.0,
            )
        )
        lower = upper
    return tuple(results)


def observation_count_distribution(events: Iterable[MemoryUseEvent]) -> dict[int, int]:
    """Return the empirical distribution of |O(m)| across use events."""

    return dict(sorted(Counter(event.observation_count for event in events).items()))


@dataclass(frozen=True)
class UtilityRetest:
    event_id: str
    utility: float


@dataclass(frozen=True)
class SignConsistency:
    event_id: str
    count: int
    positive: int
    neutral: int
    negative: int
    majority_fraction: float
    pairwise_agreement: float


@dataclass(frozen=True)
class NoiseFloorReport:
    per_event: tuple[SignConsistency, ...]
    weighted_pairwise_agreement: float
    weighted_majority_fraction: float


@dataclass(frozen=True)
class PretrainingGateReport:
    can_train: bool
    oracle_passed: bool
    noise_floor_passed: bool
    sign_consistency: float
    minimum_sign_consistency: float
    reasons: tuple[str, ...]


def evaluate_pretraining_gate(
    oracle_evaluation: "OracleEvaluation",
    noise_report: NoiseFloorReport,
    *,
    minimum_sign_consistency: float = 0.8,
    minimum_oracle_gain: float = 0.0,
) -> PretrainingGateReport:
    """Enforce oracle headroom and repeated-sign stability before fitting."""

    if not 0.0 <= minimum_sign_consistency <= 1.0:
        raise ValueError("minimum_sign_consistency must be in [0, 1]")
    if minimum_oracle_gain < 0:
        raise ValueError("minimum_oracle_gain must be non-negative")
    oracle_passed = oracle_evaluation.oracle_has_headroom(minimum_oracle_gain)
    consistency = noise_report.weighted_pairwise_agreement
    noise_passed = consistency >= minimum_sign_consistency
    reasons = []
    if not oracle_passed:
        reasons.append("oracle_policy_has_no_measurable_headroom")
    if not noise_passed:
        reasons.append("retest_sign_consistency_below_threshold")
    return PretrainingGateReport(
        can_train=oracle_passed and noise_passed,
        oracle_passed=oracle_passed,
        noise_floor_passed=noise_passed,
        sign_consistency=consistency,
        minimum_sign_consistency=minimum_sign_consistency,
        reasons=tuple(reasons),
    )


def oracle_noise_floor(
    retests: Iterable[UtilityRetest], *, neutral_epsilon: float = 0.0
) -> NoiseFloorReport:
    """Measure repeated-sign consistency for identical (memory, receiver) units."""

    if neutral_epsilon < 0:
        raise ValueError("neutral_epsilon must be non-negative")
    grouped: dict[str, list[int]] = defaultdict(list)
    for retest in retests:
        sign = 0
        if retest.utility > neutral_epsilon:
            sign = 1
        elif retest.utility < -neutral_epsilon:
            sign = -1
        grouped[retest.event_id].append(sign)
    if not grouped:
        raise ValueError("at least one retest is required")

    per_event = []
    agreeing_pairs = 0
    total_pairs = 0
    majority_correct = 0
    total_runs = 0
    for event_id in sorted(grouped):
        signs = grouped[event_id]
        counts = Counter(signs)
        pair_count = len(signs) * (len(signs) - 1) // 2
        pair_agree = sum(value * (value - 1) // 2 for value in counts.values())
        majority = max(counts.values())
        per_event.append(
            SignConsistency(
                event_id=event_id,
                count=len(signs),
                positive=counts[1],
                neutral=counts[0],
                negative=counts[-1],
                majority_fraction=majority / len(signs),
                pairwise_agreement=pair_agree / pair_count if pair_count else 1.0,
            )
        )
        agreeing_pairs += pair_agree
        total_pairs += pair_count
        majority_correct += majority
        total_runs += len(signs)
    return NoiseFloorReport(
        per_event=tuple(per_event),
        weighted_pairwise_agreement=(
            agreeing_pairs / total_pairs if total_pairs else 1.0
        ),
        weighted_majority_fraction=majority_correct / total_runs,
    )
