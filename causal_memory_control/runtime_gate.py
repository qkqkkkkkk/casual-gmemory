"""Runtime KEEP/DROP gate at GMemory's retrieve-to-expose boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .estimator import AmortizedUtilityEstimator
from .features import HashEmbedder, TeamExposureFeatureBuilder, task_context_metadata
from .gmemory_adapter import AdaptedGMemoryRetrieval, GMemoryRetrievalAdapter
from .insight_metadata import load_insight_catalog
from .jsonl import read_jsonl
from .types import (
    ExposureAction,
    MemoryCandidate,
    PotentialOutcomePrediction,
    RetrievalMetadata,
    TeamMemoryExposureEvent,
)


CHECKPOINT_SCHEMA = "gmemory-team-exposure-gate-v1"
RUNTIME_LOG_SCHEMA = "gmemory-team-exposure-runtime-v1"


@dataclass(frozen=True)
class ExposureDecision:
    event_id: str
    memory_id: str
    memory_type: str
    action: ExposureAction
    upper_utility_bound: float
    reason: str
    prediction: PotentialOutcomePrediction
    retrieval_rank: int | None = None
    memory_content: str = ""
    memory_metadata: Mapping[str, Any] = field(default_factory=dict)
    policy_scores: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "action": self.action.value,
            "upper_utility_bound": self.upper_utility_bound,
            "reason": self.reason,
            "prediction": asdict(self.prediction),
            "retrieval_rank": self.retrieval_rank,
            "memory_content": self.memory_content,
            "memory_metadata": dict(self.memory_metadata),
            "policy_scores": dict(self.policy_scores),
        }


class GMemoryExposureGate:
    """Architecture-agnostic team exposure gate.

    The learned policy is deliberately one-sided: it overrides GMemory only
    when the upper uncertainty bound is confidently harmful.  Every other
    prediction, including a cold start, keeps the host-native exposure.
    """

    MODES = frozenset(("always_keep", "always_drop", "learned"))

    def __init__(
        self,
        *,
        mode: str = "learned",
        estimator: AmortizedUtilityEstimator | None = None,
        feature_builder: TeamExposureFeatureBuilder | None = None,
        delta: float = 0.0,
        kappa: float = 1.96,
        exposure_agent_ids: Sequence[str] = ("host",),
        candidate_kinds: Sequence[str] = ("trajectory", "insight"),
        candidate_ranks: Sequence[int] | None = None,
        candidate_kind_ranks: Mapping[str, Sequence[int]] | None = None,
        max_drops: int | None = 1,
        task_metadata: Mapping[str, Any] | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
        memory_dir: str | Path | None = None,
        log_path: str | Path | None = None,
        resume: bool = False,
    ):
        if mode not in self.MODES:
            raise ValueError(f"unsupported gate mode: {mode}")
        if delta < 0 or kappa < 0:
            raise ValueError("delta and kappa must be non-negative")
        if mode == "learned" and (estimator is None or not estimator.fitted):
            raise ValueError("learned mode requires a fitted estimator")
        if max_drops is not None and max_drops < 0:
            raise ValueError("max_drops must be non-negative or None")
        if not exposure_agent_ids:
            raise ValueError("at least one host exposure recipient is required")
        unknown_kinds = set(candidate_kinds) - {"trajectory", "insight"}
        if unknown_kinds:
            raise ValueError(f"unsupported candidate kinds: {sorted(unknown_kinds)}")
        if candidate_ranks is not None and any(int(rank) < 1 for rank in candidate_ranks):
            raise ValueError("candidate ranks must be positive")
        normalized_kind_ranks = _normalize_candidate_kind_ranks(
            candidate_kind_ranks
        )
        unknown_rank_kinds = set(normalized_kind_ranks) - {
            "trajectory",
            "insight",
        }
        if unknown_rank_kinds:
            raise ValueError(
                "unsupported candidate kind/rank keys: "
                f"{sorted(unknown_rank_kinds)}"
            )
        self.mode = mode
        self.estimator = estimator
        self.feature_builder = feature_builder or TeamExposureFeatureBuilder()
        self.delta = float(delta)
        self.kappa = float(kappa)
        self.exposure_agent_ids = tuple(str(value) for value in exposure_agent_ids)
        self.candidate_kinds = frozenset(candidate_kinds)
        self.candidate_ranks = (
            frozenset(int(rank) for rank in candidate_ranks)
            if candidate_ranks is not None
            else None
        )
        self.candidate_kind_ranks = (
            {
                kind: frozenset(ranks)
                for kind, ranks in normalized_kind_ranks.items()
            }
            if candidate_kind_ranks is not None
            else None
        )
        self.max_drops = max_drops
        self.task_metadata = dict(task_metadata or {})
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.log_path = Path(log_path) if log_path is not None else None
        self.completed_task_ids: set[str] = set()
        if self.log_path is not None and self.log_path.exists():
            if not resume:
                raise FileExistsError(
                    f"gate log already exists: {self.log_path}; use resume or a fresh path"
                )
            self.completed_task_ids = self._load_completed_task_ids()
        self.memory_dir = Path(memory_dir) if memory_dir is not None else None
        insight_catalog = (
            load_insight_catalog(self.memory_dir)
            if self.memory_dir is not None
            else {}
        )
        self.adapter = GMemoryRetrievalAdapter(insight_catalog=insight_catalog)
        self._task_id: Any = None
        self._task_config: Mapping[str, Any] = {}
        self._task_decisions: list[ExposureDecision] = []
        self._retrieved_count = 0
        self._retrieval_fingerprint = ""

    def task_is_complete(self, task_id: Any) -> bool:
        return str(task_id) in self.completed_task_ids

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        **kwargs: Any,
    ) -> "GMemoryExposureGate":
        payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported GMemory exposure-gate checkpoint")
        if "jev_policy" in payload:
            from .jev_gate import gate_from_checkpoint

            return gate_from_checkpoint(checkpoint_path, payload, **kwargs)
        kwargs.pop("jev_cache_path", None)
        feature_config = payload.get("feature_builder", {})
        feature_type = feature_config.get("type")
        if feature_type not in ("team-exposure-features-v1", "team-exposure-features-v2", "team-exposure-features-v3"):
            raise ValueError("unsupported exposure feature schema")
        feature_version = int(feature_type[-1])
        # Old checkpoints learned from bare strings. Restoring nonzero support
        # counts for them would silently change their input distribution.
        if feature_version == 1:
            kwargs.pop("memory_dir", None)
        dimensions = int(feature_config.get("hash_dimensions", 256))
        if payload["estimator"].get("schema") == "mlp-utility-estimator-v1":
            from .neural_estimator import MLPUtilityEstimator

            estimator = MLPUtilityEstimator.from_dict(payload["estimator"])
        elif payload["estimator"].get("schema") == "st-direct-utility-estimator-v1":
            from .st_estimator import STUtilityEstimator

            estimator = STUtilityEstimator.from_dict(payload["estimator"])
        else:
            estimator = AmortizedUtilityEstimator.from_dict(payload["estimator"])
        if feature_version == 3:
            from .semantic_features import SentenceTransformerEmbedder, SemanticTeamFeatureBuilder

            config = feature_config["encoder"]
            if config.get("type") != "sentence-transformer-chunk-mean-v1":
                raise ValueError("unsupported semantic encoder")
            builder = SemanticTeamFeatureBuilder(SentenceTransformerEmbedder(
                config["model"], revision=config.get("revision"),
                chunk_tokens=int(config["chunk_tokens"]),
                expected_dimensions=config.get("dimensions"),
                cache_path=kwargs.pop("semantic_cache_path", Path(checkpoint_path).with_suffix(".embeddings.sqlite")),
            ))
        else:
            kwargs.pop("semantic_cache_path", None)
            builder = TeamExposureFeatureBuilder(HashEmbedder(dimensions), version=feature_version)
        checkpoint_metadata = dict(payload.get("training_metadata", {}))
        trained_kinds = set(checkpoint_metadata.get("candidate_kinds", ()))
        if trained_kinds:
            requested_kinds = set(
                kwargs.get("candidate_kinds", ("trajectory", "insight"))
            )
            kwargs["candidate_kinds"] = tuple(sorted(requested_kinds & trained_kinds))
        trained_kind_ranks = checkpoint_metadata.get("candidate_kind_ranks")
        if isinstance(trained_kind_ranks, Mapping) and trained_kind_ranks:
            normalized_trained = _normalize_candidate_kind_ranks(
                trained_kind_ranks
            )
            requested_kind_ranks = kwargs.get("candidate_kind_ranks")
            normalized_requested = (
                _normalize_candidate_kind_ranks(requested_kind_ranks)
                if isinstance(requested_kind_ranks, Mapping)
                else None
            )
            requested_global_ranks = kwargs.get("candidate_ranks")
            global_rank_scope = (
                {int(rank) for rank in requested_global_ranks}
                if requested_global_ranks is not None
                else None
            )
            requested_kinds = set(
                kwargs.get("candidate_kinds", ("trajectory", "insight"))
            )
            kwargs["candidate_kind_ranks"] = {
                kind: tuple(
                    sorted(
                        ranks
                        & (
                            normalized_requested.get(kind, set())
                            if normalized_requested is not None
                            else ranks
                        )
                        & (
                            global_rank_scope
                            if global_rank_scope is not None
                            else ranks
                        )
                    )
                )
                for kind, ranks in normalized_trained.items()
                if kind in requested_kinds
            }
            kwargs.pop("candidate_ranks", None)
        elif checkpoint_metadata.get("candidate_ranks"):
            trained_ranks = checkpoint_metadata["candidate_ranks"]
            requested_ranks = kwargs.get("candidate_ranks")
            kwargs["candidate_ranks"] = tuple(
                sorted(
                    set(int(rank) for rank in trained_ranks)
                    if requested_ranks is None
                    else set(int(rank) for rank in trained_ranks)
                    & set(int(rank) for rank in requested_ranks)
                )
            )
        return cls(
            estimator=estimator,
            feature_builder=builder,
            checkpoint_metadata=checkpoint_metadata,
            **kwargs,
        )

    def set_exposure_agent_ids(self, values: Sequence[str]) -> None:
        if not values:
            raise ValueError("at least one host exposure recipient is required")
        self.exposure_agent_ids = tuple(str(value) for value in values)

    def begin_task(self, task_id: Any, task_config: Mapping[str, Any]) -> None:
        self._task_id = task_id
        self._task_config = dict(task_config)
        self._task_decisions = []
        self._retrieved_count = 0
        self._retrieval_fingerprint = ""

    def filter_retrieval(
        self,
        result: tuple[Sequence[Any], Sequence[Any], Sequence[Any]],
        *,
        query_task: str,
        task_state: str = "",
        **_: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        retrieval = self.adapter.adapt(result)
        self._retrieval_fingerprint = _digest(query_task, task_state, json.dumps(
            [(candidate.memory_id, candidate.content) for candidate in retrieval.candidates],
            ensure_ascii=False,
        ))
        self._retrieved_count = len(retrieval.candidates)
        provisional: list[ExposureDecision] = []
        for index, candidate in enumerate(retrieval.candidates):
            event = self._build_event(
                retrieval,
                candidate,
                index=index,
                query_task=query_task,
                task_state=task_state,
            )
            provisional.append(replace(
                self._decide(event),
                retrieval_rank=event.retrieval.rank,
                memory_content=candidate.content,
                memory_metadata=candidate.metadata,
            ))

        # Limit simultaneous learned interventions because each training label
        # estimates dropping one candidate while holding the remaining set fixed.
        learned_drop_ids = {
            decision.memory_id
            for decision in sorted(
                (row for row in provisional if row.action == ExposureAction.DROP),
                key=lambda row: row.upper_utility_bound,
            )[: self.max_drops]
        } if self.mode in {"learned", "jev_direct", "jev_decomposed"} and self.max_drops is not None else {
            row.memory_id for row in provisional if row.action == ExposureAction.DROP
        }

        decisions = []
        for decision in provisional:
            if decision.action == ExposureAction.DROP and decision.memory_id not in learned_drop_ids:
                decision = replace(
                    decision,
                    action=ExposureAction.KEEP,
                    reason="kept_by_max_drops_budget",
                )
            decisions.append(decision)
        self._task_decisions.extend(decisions)
        dropped = {row.memory_id for row in decisions if row.action == ExposureAction.DROP}
        successful, failed, insights = result
        kept_successful = [
            raw
            for candidate_id, raw in zip(retrieval.successful_ids, successful)
            if candidate_id not in dropped
        ]
        kept_insights = [
            raw
            for candidate_id, raw in zip(retrieval.insight_ids, insights)
            if candidate_id not in dropped
        ]
        return kept_successful, list(failed), kept_insights

    def end_task(
        self,
        reward: float,
        done: bool,
        *,
        outcome_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        kept = sum(row.action == ExposureAction.KEEP for row in self._task_decisions)
        row = {
            "schema": RUNTIME_LOG_SCHEMA,
            "task_id": self._task_id,
            "task": self._task_summary(),
            "run_metadata": self.task_metadata,
            "checkpoint_metadata": self.checkpoint_metadata,
            "mode": self.mode,
            "delta": self.delta,
            "kappa": self.kappa,
            "max_drops": self.max_drops,
            "exposure_count": len(self.exposure_agent_ids),
            "retrieved_count": self._retrieved_count,
            "retrieval_fingerprint": self._retrieval_fingerprint,
            "kept_count": kept,
            "dropped_count": len(self._task_decisions) - kept,
            "keep_rate": kept / len(self._task_decisions) if self._task_decisions else 1.0,
            "reward": float(reward),
            "done": bool(done),
            "outcome": dict(outcome_metadata or {}),
            "decisions": [decision.to_dict() for decision in self._task_decisions],
        }
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.completed_task_ids.add(str(self._task_id))
        return row

    def _build_event(
        self,
        retrieval: AdaptedGMemoryRetrieval,
        candidate: MemoryCandidate,
        *,
        index: int,
        query_task: str,
        task_state: str,
    ) -> TeamMemoryExposureEvent:
        event_id = "team-exposure-" + _digest(
            self._task_id, query_task, task_state, candidate.memory_id
        )[:24]
        memory_index = int(candidate.metadata.get("retrieval_index", index))
        return TeamMemoryExposureEvent(
            event_id=event_id,
            query=str(query_task),
            task_state=str(task_state or query_task),
            memory=candidate,
            candidate_set=retrieval.candidates,
            exposure_agent_ids=self.exposure_agent_ids,
            retrieval=RetrievalMetadata(
                candidate_id=candidate.memory_id,
                source="g-memory",
                rank=memory_index + 1,
            ),
            task_metadata=self._event_task_metadata(),
        )

    def _decide(self, event: TeamMemoryExposureEvent) -> ExposureDecision:
        if event.memory.memory_type not in self.candidate_kinds:
            prediction = _fixed_prediction(0.0, "out_of_scope")
            return ExposureDecision(
                event.event_id,
                event.memory.memory_id,
                event.memory.memory_type,
                ExposureAction.KEEP,
                0.0,
                "candidate_kind_out_of_scope",
                prediction,
            )
        allowed_kind_ranks = (
            self.candidate_kind_ranks.get(event.memory.memory_type, frozenset())
            if self.candidate_kind_ranks is not None
            else self.candidate_ranks
        )
        if (
            allowed_kind_ranks is not None
            and event.retrieval is not None
            and event.retrieval.rank not in allowed_kind_ranks
        ):
            prediction = _fixed_prediction(0.0, "out_of_scope")
            return ExposureDecision(
                event.event_id,
                event.memory.memory_id,
                event.memory.memory_type,
                ExposureAction.KEEP,
                0.0,
                "candidate_rank_out_of_scope",
                prediction,
            )
        if self.mode == "always_keep":
            prediction = _fixed_prediction(1.0, "always_keep")
            action, upper, reason = ExposureAction.KEEP, 1.0, "always_keep_control"
        elif self.mode == "always_drop":
            prediction = _fixed_prediction(-1.0, "always_drop")
            action, upper, reason = ExposureAction.DROP, -1.0, "always_drop_control"
        else:
            prediction = self.estimator.predict(self.feature_builder.build(event))
            upper = prediction.utility + self.kappa * prediction.uncertainty
            if prediction.calibrated and upper < -self.delta:
                action, reason = ExposureAction.DROP, "confidently_harmful"
            else:
                action, reason = ExposureAction.KEEP, "default_keep"
        return ExposureDecision(
            event.event_id,
            event.memory.memory_id,
            event.memory.memory_type,
            action,
            upper,
            reason,
            prediction,
        )

    def _task_summary(self) -> dict[str, Any]:
        keys = (
            "task_main",
            "task_description",
            "game_name",
            "problem_index",
            "claim_id",
            "claim",
            "hotpot_id",
            "question",
            "id",
            "task",
        )
        return {key: self._task_config[key] for key in keys if key in self._task_config}

    def _event_task_metadata(self) -> dict[str, Any]:
        if self.feature_builder.version >= 2:
            # Experiment seeds and snapshot settings are audit metadata, not
            # task state. Diagnostic examples do not contain those fields.
            return task_context_metadata(self._task_config)
        metadata = dict(self.task_metadata)
        excluded = {
            "answer",
            "label",
            "gold_label",
            "task_main",
            "task_description",
            "claim",
            "claim_id",
            "id",
            "task_id",
            "task",
            "few_shots",
        }
        metadata.update(
            {
                key: value
                for key, value in self._task_config.items()
                if key not in excluded and isinstance(value, (str, int, float, bool))
            }
        )
        return metadata

    def _load_completed_task_ids(self) -> set[str]:
        completed: set[str] = set()
        for line_number, row in enumerate(read_jsonl(self.log_path, repair_tail=True), 1):
            if row.get("schema") != RUNTIME_LOG_SCHEMA:
                raise ValueError(
                    f"unexpected gate-log schema at line {line_number}: {row.get('schema')}"
                )
            if row.get("mode") != self.mode:
                raise ValueError(
                    f"gate-log mode mismatch at line {line_number}: {row.get('mode')}"
                )
            if row.get("run_metadata", {}) != self.task_metadata:
                raise ValueError(f"gate-log run metadata mismatch at line {line_number}")
            if row.get("checkpoint_metadata", {}) != self.checkpoint_metadata:
                raise ValueError(
                    f"gate-log checkpoint metadata mismatch at line {line_number}"
                )
            task_id = str(row.get("task_id"))
            if task_id in completed:
                raise ValueError(f"duplicate task_id {task_id!r} in gate log")
            completed.add(task_id)
        return completed

    def close(self):
        close = getattr(self.feature_builder.embedder, "close", None)
        if close:
            close()


def save_gate_checkpoint(
    path: str | Path,
    estimator: AmortizedUtilityEstimator,
    *,
    hash_dimensions: int = 256,
    feature_version: int = 1,
    feature_config: Mapping[str, Any] | None = None,
    training_metadata: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "feature_builder": {
            "type": f"team-exposure-features-v{feature_version}",
            "hash_dimensions": int(hash_dimensions),
            **dict(feature_config or {}),
        },
        "estimator": estimator.to_dict(),
        "training_metadata": dict(training_metadata or {}),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)


def _normalize_candidate_kind_ranks(
    value: Mapping[str, Sequence[int]] | None,
) -> dict[str, set[int]]:
    if value is None:
        return {}
    normalized: dict[str, set[int]] = {}
    for raw_kind, raw_ranks in value.items():
        kind = str(raw_kind)
        if isinstance(raw_ranks, (str, bytes)):
            raise ValueError(
                f"candidate ranks for {kind!r} must be a sequence of integers"
            )
        try:
            ranks = {int(rank) for rank in raw_ranks}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"candidate ranks for {kind!r} must be iterable"
            ) from exc
        if any(rank < 1 for rank in ranks):
            raise ValueError("candidate kind/ranks must be positive")
        normalized[kind] = ranks
    return normalized


def _fixed_prediction(utility: float, source: str) -> PotentialOutcomePrediction:
    return PotentialOutcomePrediction(
        q_use=max(utility, 0.0),
        q_drop=max(-utility, 0.0),
        utility=utility,
        utility_class="positive" if utility > 0 else "negative" if utility < 0 else "neutral",
        uncertainty=0.0,
        source=source,
        calibrated=True,
        training_samples=0,
    )


def _digest(*values: Any) -> str:
    return hashlib.sha256(
        "\x1f".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()
