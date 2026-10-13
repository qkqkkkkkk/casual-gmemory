"""Dataset-specific contracts for the shared GMemory gate comparison matrix.

An adapter owns loading/splitting, diagnostic collection, checkpoint validation,
evaluation and metric definitions. Gate training and matrix orchestration stay
shared. Additional datasets can provide a factory via --dataset-adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import run_hotpotqa_gate_comparison as shared


@dataclass(frozen=True)
class MatrixMetric:
    name: str
    label: str
    summary_field: str
    gate_field: str
    delta_field: str
    interval_field: str


class DatasetAdapter(ABC):
    """Version this identity whenever a dataset's experiment semantics change."""

    name: str
    version = "1"
    display_name: str
    benchmark: str
    training_metric: str
    matrix_schema: str
    default_matrix_dir: Path
    metrics: Sequence[MatrixMetric]

    @abstractmethod
    def parse_args(self, argv): ...

    @abstractmethod
    def resolve_plan(self, args): ...

    @abstractmethod
    def validate_args(self, args): ...

    @abstractmethod
    def configuration(self, args): ...

    @abstractmethod
    def plan(self, args): ...

    @abstractmethod
    def diagnostic_command(self, args): ...

    @abstractmethod
    def diagnostic_complete(self, args): ...

    @abstractmethod
    def train_command(self, args): ...

    @abstractmethod
    def checkpoint_complete(self, args): ...

    @abstractmethod
    def evaluation_command(self, args, *, mode, output, resume): ...

    @abstractmethod
    def evaluation_complete(self, args, output, *, mode, command=None): ...

    @abstractmethod
    def validate_checkpoint(self, payload, args, snapshot): ...

    def checkpoint_context(self, args, snapshot):
        return {"benchmark": self.benchmark, "source_md5": snapshot["source_md5"],
                "snapshot_design_hash": snapshot["design_hash"],
                **{key: getattr(args, key) for key in (
                    "model", "graph_type", "node_num", "embedding_model",
                    "successful_topk", "failed_topk", "insights_topk", "threshold",
                )}}


class HotpotQADatasetAdapter(DatasetAdapter):
    name, display_name = "hotpotqa", "HotpotQA"
    benchmark, training_metric = "HotpotQA_distractor_offline", "team_score"
    matrix_schema = "hotpotqa-semantic-jev-matrix-v1"
    default_matrix_dir = Path("causal_memory_control/results/hotpotqa_gate_semantic_jev_v3")
    metrics = (
        MatrixMetric("answer_f1", "F1", "mean_answer_f1", "gate_answer_f1", "answer_f1_delta", "answer_f1_delta_95ci"),
        MatrixMetric("answer_em", "EM", "answer_exact_match_rate", "gate_answer_em", "answer_em_delta", "answer_em_delta_95ci"),
    )

    def parse_args(self, argv):
        return shared.parse_args(argv)

    def resolve_plan(self, args):
        shared._resolve_plan(args)

    def validate_args(self, args):
        shared._validate_args(args)

    def configuration(self, args):
        return shared._configuration(args)

    def plan(self, args):
        return shared._plan(args)

    def diagnostic_command(self, args):
        return shared._diagnostic_command(args)

    def diagnostic_complete(self, args):
        return shared._diagnostic_complete(args)

    def train_command(self, args):
        return shared._train_command(args)

    def checkpoint_complete(self, args):
        return shared._checkpoint_complete(args)

    def evaluation_command(self, args, *, mode, output, resume):
        return shared._evaluation_command(args, mode=mode, output=output, resume=resume)

    def evaluation_complete(self, args, output, *, mode, command=None):
        return shared._validated_evaluation_complete(args, output, mode=mode, command=command)

    def validate_checkpoint(self, payload, args, snapshot):
        from .hotpotqa_experiment import validate_checkpoint_for_hotpotqa

        validate_checkpoint_for_hotpotqa(payload,
            evaluation_ids=snapshot["evaluation_ids"][args.training_claims:args.evaluation_count],
            expected_context=self.checkpoint_context(args, snapshot),
            candidate_kinds=_candidate_kinds(args))


class FeverDatasetAdapter(DatasetAdapter):
    """Existing binary offline FEVER task, with balanced registered sampling."""

    name, display_name = "fever", "FEVER (binary)"
    benchmark, training_metric = "FEVER_binary_offline", "success"
    matrix_schema = "fever-semantic-jev-matrix-v1"
    default_matrix_dir = Path("causal_memory_control/results/fever_gate_semantic_jev_v1")
    metrics = (
        MatrixMetric("accuracy", "Accuracy", "mean_reward", "gate_mean_reward", "reward_delta", "reward_delta_95ci"),
        MatrixMetric("completion_rate", "Completion", "completion_rate", "gate_completion_rate", "completion_rate_delta", "completion_rate_delta_95ci"),
    )

    def parse_args(self, argv):
        return shared.parse_args(argv, description="Shared gate comparison on binary offline FEVER",
            defaults={"data": Path("data/fever/fever_dev.jsonl"),
                "memory_dir": Path("causal_diagnostic/memory_snapshots/fever_gate_matrix_support50_7b_v1/g-memory"),
                "diagnostic_results": Path("causal_diagnostic/results/fever_gate_matrix_training_7b_v1"),
                "checkpoint": Path("causal_memory_control/checkpoints/fever_gate_matrix_v1.json"),
                "output_dir": self.default_matrix_dir})

    def resolve_plan(self, args):
        from causal_diagnostic.fever_oracle_recipient.data import BINARY_LABELS, load_binary_fever

        examples = load_binary_fever(args.data)
        counts = Counter(row["label"] for row in examples)
        args.valid_source_questions = len(examples)
        if args.support_count < 2 or args.support_count % 2:
            raise SystemExit("FEVER --support-count must be a positive even total (balanced per label)")
        if args.use_all_data:
            if len({counts[label] for label in BINARY_LABELS}) != 1:
                raise SystemExit("FEVER --use-all-data requires equal SUPPORTS/REFUTES counts; the balanced split cannot use every row otherwise")
            args.evaluation_count = len(examples) - args.support_count
        if args.evaluation_count < 2 or args.evaluation_count % 2:
            raise SystemExit("FEVER --evaluation-count must be a positive even total")
        args.support_per_label = args.support_count // 2
        args.evaluation_per_label = args.evaluation_count // 2
        if any(counts[label] < args.support_per_label + args.evaluation_per_label for label in BINARY_LABELS):
            raise SystemExit(f"FEVER balanced split exceeds available binary rows: {dict(counts)}")
        args.training_claims = math.floor(args.evaluation_count * args.training_fraction)
        if args.use_all_data:
            # Match the legacy all-data plan's per-label rounding exactly.
            args.training_claims = 2 * math.floor(args.evaluation_per_label * args.training_fraction)
        args.final_claims = args.evaluation_count - args.training_claims
        args.candidate_scope, args.diagnostic_scope = "all_retrieved", "gate_training"
        args.candidate_kind, args.candidate_index = "trajectory", 0
        args.isolated_recipients = f"solver_0,solver_{args.node_num - 1}"
        args.binary_label_counts = {label: counts[label] for label in BINARY_LABELS}

    def validate_args(self, args):
        shared._validate_args(args)

    def configuration(self, args):
        return {**shared._configuration(args), "pipeline_schema": self.matrix_schema}

    def plan(self, args):
        return {**shared._plan(args), "binary_label_counts": args.binary_label_counts,
                "task": "SUPPORTS/REFUTES only; NOT ENOUGH INFO excluded",
                "sampling": "balanced registered support/evaluation split"}

    def diagnostic_command(self, args):
        from .run_fever_gate_comparison import _diagnostic_command

        return _diagnostic_command(args)

    def diagnostic_complete(self, args):
        from .run_fever_gate_comparison import _diagnostic_complete

        return _diagnostic_complete(args)

    def train_command(self, args):
        return shared._train_command(args, metric=self.training_metric)

    def checkpoint_complete(self, args):
        complete = shared._checkpoint_complete(args, benchmark=self.benchmark, metric=self.training_metric)
        if complete:
            snapshot = shared._load_json(args.memory_dir / "causal_snapshot_manifest.json")
            self.validate_checkpoint(shared._load_json(args.checkpoint), args, snapshot)
        return complete

    def evaluation_command(self, args, *, mode, output, resume):
        return shared._evaluation_command(args, mode=mode, output=output, resume=resume,
                                          runner_module="causal_memory_control.fever_experiment")

    def evaluation_complete(self, args, output, *, mode, command=None):
        from . import fever_experiment

        return shared._validated_evaluation_complete(args, output, mode=mode, command=command,
            runner=fever_experiment, benchmark=self.benchmark, completed_field="completed_claims")

    def validate_checkpoint(self, payload, args, snapshot):
        from .fever_experiment import validate_checkpoint_for_fever

        validate_checkpoint_for_fever(payload,
            evaluation_ids=snapshot["evaluation_ids"][args.training_claims:args.evaluation_count],
            expected_context=self.checkpoint_context(args, snapshot),
            candidate_kinds=_candidate_kinds(args))
        ranks = payload["training_metadata"].get("candidate_kind_ranks", {})
        missing = [(kind, rank) for kind, count in (("trajectory", args.successful_topk), ("insight", args.insights_topk))
                   for rank in range(1, count + 1) if rank not in ranks.get(kind, [])]
        if missing:
            raise ValueError(f"FEVER checkpoint has no counterfactual coverage for {missing}")


def _candidate_kinds(args):
    return [kind for kind, count in (("trajectory", args.successful_topk), ("insight", args.insights_topk)) if count]


DATASET_ADAPTERS = {"hotpotqa": HotpotQADatasetAdapter, "fever": FeverDatasetAdapter}


def load_dataset_adapter(name: str, factory: str | None = None) -> DatasetAdapter:
    if factory:
        try:
            module, function = factory.split(":", 1)
            adapter = getattr(importlib.import_module(module), function)()
        except (ValueError, ImportError, AttributeError, TypeError) as exc:
            raise SystemExit(f"cannot load --dataset-adapter {factory}: {exc}") from exc
    else:
        if name not in DATASET_ADAPTERS:
            raise SystemExit(f"unknown dataset {name!r}; choose {', '.join(DATASET_ADAPTERS)} or provide --dataset-adapter module:factory")
        adapter = DATASET_ADAPTERS[name]()
    if not isinstance(adapter, DatasetAdapter):
        raise SystemExit("dataset factory must return a DatasetAdapter instance")
    if adapter.name != name:
        raise SystemExit(f"dataset adapter name {adapter.name!r} differs from --dataset {name!r}")
    if not adapter.metrics or not adapter.version:
        raise SystemExit("dataset adapter must define version and metrics")
    return adapter
