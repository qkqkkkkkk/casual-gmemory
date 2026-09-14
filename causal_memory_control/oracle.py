"""Oracle-controllability gate and matched-budget policy comparison."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class OracleExample:
    event_id: str
    q_use: float
    q_drop: float
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if not math.isfinite(float(self.q_use)) or not math.isfinite(float(self.q_drop)):
            raise ValueError("potential outcomes must be finite")
        if any(not math.isfinite(float(score)) for score in self.scores.values()):
            raise ValueError("policy scores must be finite")

    @property
    def utility(self) -> float:
        return self.q_use - self.q_drop


@dataclass(frozen=True)
class PolicyEvaluation:
    policy: str
    mean_reward: float
    accepted_count: int
    accepted_rate: float
    gain_vs_always_use: float
    gain_vs_never_use: float
    regret_to_oracle: float
    decisions: tuple[bool, ...]


@dataclass(frozen=True)
class OracleEvaluation:
    delta: float
    matched_budget: int
    policies: tuple[PolicyEvaluation, ...]

    def policy(self, name: str) -> PolicyEvaluation:
        for policy in self.policies:
            if policy.policy == name:
                return policy
        raise KeyError(name)

    def oracle_has_headroom(self, min_gain: float = 0.0) -> bool:
        oracle = self.policy("oracle").mean_reward
        strongest_non_oracle = max(
            policy.mean_reward
            for policy in self.policies
            if policy.policy != "oracle"
        )
        return oracle > strongest_non_oracle + min_gain


class OracleControllability:
    """Evaluate g*(x)=1[Q_use-Q_drop > delta] before fitting a predictor."""

    def __init__(self, *, delta: float = 0.0):
        if delta < 0:
            raise ValueError("delta must be non-negative")
        self.delta = delta

    def evaluate(
        self,
        examples: Sequence[OracleExample],
        *,
        score_names: Sequence[str] = (
            "similarity",
            "reliability",
            "llm_judge",
            "local_filter",
        ),
    ) -> OracleEvaluation:
        if not examples:
            raise ValueError("at least one oracle example is required")
        oracle_decisions = tuple(example.utility > self.delta for example in examples)
        matched_budget = sum(oracle_decisions)
        decisions: dict[str, tuple[bool, ...]] = {
            "always_use": tuple(True for _ in examples),
            "never_use": tuple(False for _ in examples),
            "oracle": oracle_decisions,
        }
        for score_name in score_names:
            if all(score_name in example.scores for example in examples):
                ranked = sorted(
                    range(len(examples)),
                    key=lambda index: (
                        -float(examples[index].scores[score_name]),
                        examples[index].event_id,
                    ),
                )
                accepted = set(ranked[:matched_budget])
                decisions[f"{score_name}_matched_budget"] = tuple(
                    index in accepted for index in range(len(examples))
                )

        always_value = _policy_value(examples, decisions["always_use"])
        never_value = _policy_value(examples, decisions["never_use"])
        oracle_value = _policy_value(examples, decisions["oracle"])
        policies = []
        for name, policy_decisions in decisions.items():
            value = _policy_value(examples, policy_decisions)
            accepted_count = sum(policy_decisions)
            policies.append(
                PolicyEvaluation(
                    policy=name,
                    mean_reward=value,
                    accepted_count=accepted_count,
                    accepted_rate=accepted_count / len(examples),
                    gain_vs_always_use=value - always_value,
                    gain_vs_never_use=value - never_value,
                    regret_to_oracle=oracle_value - value,
                    decisions=policy_decisions,
                )
            )
        return OracleEvaluation(
            delta=self.delta,
            matched_budget=matched_budget,
            policies=tuple(policies),
        )


def _policy_value(examples: Sequence[OracleExample], decisions: Sequence[bool]) -> float:
    return sum(
        example.q_use if use else example.q_drop
        for example, use in zip(examples, decisions)
    ) / len(examples)

