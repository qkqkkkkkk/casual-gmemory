"""Matched-seed counterfactual audit at the receiver injection boundary."""

from __future__ import annotations

import math
from typing import Iterable, Optional

from .behavior import ExactMatchDistance
from .interventions import InterventionBuilder
from .protocols import BehaviorDistance, BranchRunner
from .stats import summarize
from .types import (
    ArmAuditResult,
    AuditCheckpoint,
    CounterfactualAuditResult,
    InterventionArm,
    MemoryUseEvent,
    PairedRun,
)


class CounterfactualAudit:
    """Replay USE and controls from the same checkpoint and matched seeds.

    The runner owns checkpoint restoration and downstream execution.  USE is
    evaluated once per seed and shared across all requested controls, avoiding
    accidental differences in upstream outputs and unnecessary LLM calls.
    """

    def __init__(
        self,
        runner: BranchRunner,
        *,
        intervention_builder: Optional[InterventionBuilder] = None,
        behavior_distance: Optional[BehaviorDistance] = None,
        repeats: int = 3,
        base_seed: int = 17,
    ):
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        self.runner = runner
        self.intervention_builder = intervention_builder or InterventionBuilder()
        self.behavior_distance = behavior_distance or ExactMatchDistance()
        self.repeats = repeats
        self.base_seed = base_seed

    def run(
        self,
        checkpoint: AuditCheckpoint,
        event: MemoryUseEvent,
        *,
        arms: Iterable[InterventionArm] = (InterventionArm.DROP,),
        seeds: Optional[Iterable[int]] = None,
    ) -> CounterfactualAuditResult:
        controls = tuple(dict.fromkeys(arms))
        if not controls:
            raise ValueError("at least one control arm is required")
        if InterventionArm.USE in controls:
            raise ValueError("USE is the treatment baseline, not a control arm")

        matched_seeds = (
            tuple(seeds)
            if seeds is not None
            else tuple(self.base_seed + index for index in range(self.repeats))
        )
        if not matched_seeds:
            raise ValueError("at least one seed is required")
        if len(set(matched_seeds)) != len(matched_seeds):
            raise ValueError("seeds must be unique")

        pairs_by_arm: dict[InterventionArm, list[PairedRun]] = {
            arm: [] for arm in controls
        }
        for repeat_index, seed in enumerate(matched_seeds):
            use_request = self.intervention_builder.build(
                checkpoint,
                event,
                InterventionArm.USE,
                seed=seed,
                repeat_index=repeat_index,
            )
            use_outcome = self.runner.run(use_request)
            for arm in controls:
                control_request = self.intervention_builder.build(
                    checkpoint,
                    event,
                    arm,
                    seed=seed,
                    repeat_index=repeat_index,
                )
                control_outcome = self.runner.run(control_request)
                distance = float(
                    self.behavior_distance(use_outcome.behavior, control_outcome.behavior)
                )
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError("behavior distance must be finite and non-negative")
                local_utility = None
                if (
                    use_outcome.local_metric is not None
                    and control_outcome.local_metric is not None
                ):
                    local_utility = (
                        use_outcome.local_metric - control_outcome.local_metric
                    )
                pairs_by_arm[arm].append(
                    PairedRun(
                        repeat_index=repeat_index,
                        seed=seed,
                        use=use_outcome,
                        control=control_outcome,
                        behavior_distance=distance,
                        team_utility=use_outcome.team_reward
                        - control_outcome.team_reward,
                        local_utility=local_utility,
                    )
                )

        results = []
        for arm, pairs_list in pairs_by_arm.items():
            pairs = tuple(pairs_list)
            local_values = tuple(
                pair.local_utility
                for pair in pairs
                if pair.local_utility is not None
            )
            results.append(
                ArmAuditResult(
                    arm=arm,
                    pairs=pairs,
                    behavior_effect=summarize(
                        pair.behavior_distance for pair in pairs
                    ),
                    team_utility=summarize(pair.team_utility for pair in pairs),
                    local_utility=summarize(local_values) if local_values else None,
                )
            )
        return CounterfactualAuditResult(
            event=event,
            checkpoint_id=checkpoint.checkpoint_id,
            arms=tuple(results),
        )

