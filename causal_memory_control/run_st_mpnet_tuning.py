"""Offline MPNet ST head/chunk tuning on the original QA fit/dev partitions.

No collection, encoder fine-tuning, MiniLM refitting or test-based selection.
Only the dev-selected shared configuration can enter the explicit test stage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import re
import statistics
from types import SimpleNamespace

from . import reevaluate_st_policy as reuse
from .interactive_data import digest, locked, save, sha
from .run_st_encoder_benchmark import offline_models, partitions, compare_encoders
from .st_policy import resolve_rule, validate_st_checkpoint


SCHEMA = "semantic-st-mpnet-dev-tuning-v1"
RESULTS = Path("causal_memory_control/results")
HEAD_KEYS = {"learning_rate", "l2", "beta", "hidden_sizes", "epochs"}
POLICY_KEYS = ("recommended_kappa", "recommended_max_drops", "delta", "calibration_task_ids")
DEFAULT_TRIALS = [
    {"id": "original", "head": {}},
    {"id": "lr_3e4", "head": {"learning_rate": .0003}},
    {"id": "l2_1e3", "head": {"l2": .001}},
    {"id": "beta_0", "head": {"beta": 0.}},
    {"id": "beta_003", "head": {"beta": .003}},
    {"id": "beta_03", "head": {"beta": .03}},
    {"id": "wide", "head": {"hidden_sizes": [256, 128]}},
    {"id": "epochs_20", "head": {"epochs": 20}},
    {"id": "chunk_128", "head": {}, "chunk_tokens": 128},
    {"id": "chunk_320", "head": {}, "chunk_tokens": 320},
]
SELECTION = {
    "scope": "original_dev_labels_only",
    "metric": "equal_dataset_mean_of_class_balanced_mse_relative_to_original_profile",
    "within_class": "mean_of_task_mean_squared_utility_errors",
    "classes": "negative / exact_zero / positive observed U=Q_use-Q_drop",
    "class_weight": "equal_over_present_dev_classes",
    "normalizer_floor": 1e-8,
    "tie_break": "candidate_order_original_first",
    "test_outcomes_used": False,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--datasets", default="fever,hotpotqa")
    parser.add_argument("--qa-root", type=Path, default=RESULTS / "all_trained_gates_with_old_st_retest_v2")
    parser.add_argument("--mpnet-reference-root", type=Path, default=RESULTS / "st_mpnet_qa_v1",
                        help="Optional completed original MPNet run; read only during test evaluation")
    parser.add_argument("--rule-reference-root", type=Path, default=RESULTS / "st_positive_unlimited_retest_v1")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "st_mpnet_tuning_qa_v1")
    parser.add_argument("--semantic-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--semantic-revision")
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--trials-json", type=Path, help="Predeclared JSON list: id, head overrides, optional chunk_tokens")
    parser.add_argument("--stage", choices=("plan", "tune", "evaluate", "report"), default="plan")
    parser.add_argument("--refit-head", action="store_true")
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("this independent tuning entry supports fever,hotpotqa only")
    if args.stage == "tune" and not args.refit_head:
        parser.error("--stage tune requires --refit-head (existing labels, new small heads)")
    for key in ("qa_root", "mpnet_reference_root", "rule_reference_root", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    for root in (args.qa_root, args.mpnet_reference_root, args.rule_reference_root):
        reuse.separate_output(args.output_dir, root)
    if Path(args.semantic_model).is_dir():
        args.semantic_model = str(Path(args.semantic_model).resolve())
        reuse.separate_output(args.output_dir, args.semantic_model)
    return args


def trial_specs(args):
    values = reuse.read(args.trials_json) if args.trials_json else deepcopy(DEFAULT_TRIALS)
    if not isinstance(values, list) or not values or values[0] != {"id": "original", "head": {}}:
        raise ValueError("trial list must start with the unchanged original profile")
    ids = set()
    for spec in values:
        if (not isinstance(spec, dict) or set(spec) - {"id", "head", "chunk_tokens"}
                or not isinstance(spec.get("id"), str)
                or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", spec["id"])
                or spec["id"] in ids or not isinstance(spec.get("head"), dict)
                or set(spec["head"]) - HEAD_KEYS):
            raise ValueError("invalid/duplicate trial ID or unsupported head override")
        ids.add(spec["id"])
        if "chunk_tokens" in spec and (type(spec["chunk_tokens"]) is not int or spec["chunk_tokens"] < 8):
            raise ValueError("chunk_tokens must be an integer >= 8")
        for key, value in spec["head"].items():
            if key == "hidden_sizes":
                valid = isinstance(value, list) and value and all(type(v) is int and v > 0 for v in value)
            elif key == "epochs":
                valid = type(value) is int and value > 0
            else:
                valid = (type(value) in (int, float) and math.isfinite(value)
                         and (value > 0 if key == "learning_rate" else value >= 0))
            if not valid:
                raise ValueError(f"invalid trial parameter {key}: {value}")
    return values


def effective(job, spec):
    config = {**job["payload"]["estimator"]["config"], **spec["head"]}
    chunk = spec.get("chunk_tokens", job["payload"]["feature_builder"]["encoder"]["chunk_tokens"])
    return config, chunk


def prepare(args):
    """Read manifests, partitions and measured training inputs; never test logs."""
    from .source_run import import_source_dataset

    identity, jobs = {}, {}
    reevaluation = args.qa_root / "reevaluation_manifest.json"
    trained = Path(reuse.read(reevaluation)["trained_root"]) / "csm_st_manifest.json"
    source_root = Path(reuse.read(trained)["source_root"]).resolve()
    reuse.separate_output(args.output_dir, source_root)
    for path in (reevaluation, trained, source_root / "benchmark_manifest.json"):
        reuse.pin(identity, path)
    specs = trial_specs(args)
    for name in args.datasets:
        adapter, source_args = import_source_dataset(SimpleNamespace(source_root=source_root, shared_argv=[]), name)
        run_path = args.qa_root / name / "semantic_st/run_manifest.json"
        previous = reuse.read(run_path)["run"]
        checkpoint = Path(previous["checkpoint"]).resolve()
        payload = validate_st_checkpoint(checkpoint)
        if sha(checkpoint) != previous["checkpoint_sha256"]:
            raise ValueError(f"{name}: original ST checkpoint changed")
        if payload["feature_builder"]["type"] != "team-exposure-features-v3":
            raise ValueError("MPNet tuning requires the original semantic v3 ST features")
        snapshot_path = source_args.memory_dir / "causal_snapshot_manifest.json"
        snapshot = reuse.read(snapshot_path)
        fit, dev = partitions(payload)
        test = {str(t) for t in snapshot["evaluation_ids"][source_args.training_claims:source_args.evaluation_count]}
        training = {str(t) for t in snapshot["evaluation_ids"][:source_args.training_claims]}
        support = {str(t) for t in snapshot["support_ids"]}
        # Historical heads may use only a measured subset of the registered
        # training pool. Preserve that exact fit/dev split instead of requiring
        # unused training tasks to have labels or adding them to the new fit.
        checkpoint_tasks = fit | dev
        recorded_test = {str(t) for t in previous["evaluation_ids"]}
        partition_counts = {"registered_training_tasks": len(training),
            "checkpoint_training_tasks": len(checkpoint_tasks),
            "unused_registered_training_tasks": len(training - checkpoint_tasks),
            "fit_tasks": len(fit), "dev_tasks": len(dev),
            "support_tasks": len(support), "test_tasks": len(test)}
        conflicts = {key: sorted(values) for key, values in {
            "checkpoint_tasks_outside_training_pool": checkpoint_tasks - training,
            "checkpoint_tasks_in_test_or_support": checkpoint_tasks & (test | support),
            "registered_training_overlap": training & (test | support),
            "support_test_overlap": support & test,
            "missing_recorded_test_tasks": test - recorded_test,
            "unexpected_recorded_test_tasks": recorded_test - test,
        }.items() if values}
        if conflicts:
            details = {"counts": partition_counts,
                "conflicts": {key: {"count": len(values), "first_ids": values[:10]}
                              for key, values in conflicts.items()}}
            raise ValueError(f"{name}: original fit/dev/support/test partition mismatch or leakage; "
                             + json.dumps(details, ensure_ascii=False, sort_keys=True))
        adapter.validate_checkpoint(payload, source_args, snapshot)
        metadata = payload["training_metadata"]
        recorded = [Path(p) for p in metadata["training_sources"]]
        sources = [p / "branches.jsonl" if p.is_dir() else p for p in recorded]
        if {str(p): sha(s) for p, s in zip(recorded, sources)} != metadata["training_source_sha256"]:
            raise ValueError(f"{name}: original measured training labels changed")
        job = jobs[name] = {"name": name, "checkpoint": checkpoint, "payload": payload,
            "sources": sources, "memory": source_args.memory_dir, "fit": fit, "dev": dev,
            "partition_counts": partition_counts,
            "deployment": {k: previous[k] for k in ("kappa", "delta", "max_drops")}}
        for root in (checkpoint.parent, job["memory"], *(p.parent for p in sources)):
            reuse.separate_output(args.output_dir, root)
        for path in (run_path, checkpoint, snapshot_path, source_args.data,
                     *sources, *(p.parent / "run_manifest.json" for p in sources)):
            reuse.pin(identity, path)
        for path in sorted(job["memory"].rglob("*")):
            if path.is_file() and not path.name.endswith(("-shm", "-wal")):
                reuse.pin(identity, path)
    for module in ("run_st_mpnet_tuning", "run_st_encoder_benchmark", "reevaluate_st_policy",
                   "st_policy", "st_encoder_evaluate", "st_policy_evaluate", "st_estimator",
                   "semantic_features", "st_data", "train_gate", "runtime_gate", "features",
                   "gmemory_adapter", "task_state", "dataset_adapters", "source_run",
                   "fever_experiment", "hotpotqa_experiment", "compare_gate_runs", "csm_st_worker",
                   "interactive_data", "estimator", "types", "insight_metadata",
                   "run_hotpotqa_gate_comparison", "run_full_gate_benchmark"):
        reuse.pin(identity, Path(__file__).with_name(module + ".py"))
    if args.trials_json:
        reuse.pin(identity, args.trials_json)
    manifest = {"schema": SCHEMA, "datasets": args.datasets, "encoder": {
        "model": args.semantic_model, "revision": args.semantic_revision, "trained": False, "device": "cpu"},
        "train_device": args.train_device, "trials": specs, "selection": SELECTION,
        "sources": {name: {"checkpoint": str(j["checkpoint"]), "fit_tasks": len(j["fit"]),
            "dev_tasks": len(j["dev"]), "rule": resolve_rule("dataset_default", name),
            "partition_counts": j["partition_counts"],
            "deployment": j["deployment"],
            "effective_trials": {s["id"]: {"head": effective(j, s)[0], "chunk_tokens": effective(j, s)[1]}
                                 for s in specs}} for name, j in jobs.items()},
        "qa_root": str(args.qa_root), "mpnet_reference_root": str(args.mpnet_reference_root),
        "rule_reference_root": str(args.rule_reference_root),
        "source_sha256": identity, "new_diagnostic_tasks": 0, "minilm_refitted": False,
        "original_results_overwritten": False, "maximum_head_fits": len(specs) * len(jobs)}
    return jobs, manifest


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".tuning.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"tuning output is already running/locked: {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def checked_encoder(args, name, chunk):
    from .csm_st_worker import dedup_identity
    from .semantic_features import SentenceTransformerEmbedder

    root = args.output_dir / name / "features" / f"chunk_{chunk}"
    encoder = SentenceTransformerEmbedder(args.semantic_model, revision=args.semantic_revision,
        chunk_tokens=chunk, device="cpu", cache_path=root / "semantic.sqlite")
    try:
        spec = {k: v for k, v in encoder.config().items() if k != "dimensions"}
        identity = dedup_identity(encoder, spec)
        if not identity.get("local_model_sha256") and not identity.get("resolved_commit"):
            raise ValueError("cannot identify frozen MPNet weights; supply a complete local model directory")
        locked(args.output_dir / "encoder_weights_identity.json",
               {k: v for k, v in identity.items() if k != "chunk_tokens"})
        locked(root / "encoder_identity.json", identity)
        if not Path(args.semantic_model).is_dir():
            encoder.revision = identity["resolved_commit"]
        return encoder, identity
    except BaseException:
        encoder.close()
        raise


def load_examples(job, encoder):
    from .semantic_features import SemanticTeamFeatureBuilder
    from .st_data import load_st_examples

    examples = load_st_examples(job["sources"], metric=job["payload"]["training_metadata"]["metric"],
        task_ids={int(t) for t in job["fit"] | job["dev"]},
        hash_dimensions=job["payload"]["feature_builder"].get("hash_dimensions", 256),
        feature_version=3, memory_dir=job["memory"], feature_builder=SemanticTeamFeatureBuilder(encoder), event_sink=[])
    fit = [e for e in examples if str(e.group_id) in job["fit"]]
    dev = [e for e in examples if str(e.group_id) in job["dev"]]
    coverage = job["payload"]["training_metadata"].get("candidate_coverage")
    for part, values in (("fit", fit), ("dev", dev)):
        expected = {str(v) for v in coverage[part]["supervised_task_ids"]} if coverage else job[part]
        if not values or {str(e.group_id) for e in values} != expected:
            raise ValueError(f"{job['name']}: changed/incomplete supervised {part} coverage")
    return fit, dev


def utility_metrics(estimator, examples):
    """Task-balanced within sign class; keep exact-zero labels distinct."""
    grouped = defaultdict(lambda: defaultdict(list))
    errors, signs, predictions = [], [], []
    for start in range(0, len(examples), 256):
        batch = examples[start:start + 256]
        predictions.extend(estimator.predict_many([e.features for e in batch]))
    for example, prediction in zip(examples, predictions):
        u, p = example.q_use - example.q_drop, prediction.utility
        if not math.isfinite(p):
            raise ValueError("non-finite dev utility prediction")
        label = "negative" if u < 0 else "positive" if u > 0 else "zero"
        error = (p - u) ** 2
        grouped[label][str(example.group_id)].append(error)
        errors.append(error)
        if u != 0:
            signs.append((p > 0) == (u > 0) and p != 0)
    if not errors:
        raise ValueError("cannot select on empty dev supervision")
    classes = {key: {"n": sum(map(len, tasks.values())), "tasks": len(tasks),
                    "task_balanced_mse": statistics.mean(statistics.mean(v) for v in tasks.values())}
               for key, tasks in sorted(grouped.items())}
    return {"n": len(errors), "tasks": len({str(e.group_id) for e in examples}),
        "utility_rmse": statistics.mean(errors) ** .5,
        "class_balanced_mse": statistics.mean(c["task_balanced_mse"] for c in classes.values()),
        "classes": classes, "nonzero_sign_accuracy": statistics.mean(signs) if signs else None}


def trial_root(args, name, spec):
    return args.output_dir / name / "trials" / spec["id"]


def binding_for(manifest, name, spec, identity):
    return {"manifest_sha256": digest(manifest), "dataset": name, "trial": spec,
            "encoder_identity": identity}


def require_trial(args, job, spec, manifest, identity):
    root = trial_root(args, job["name"], spec)
    record = reuse.read(root / "trial.json")
    if record["binding"] != binding_for(manifest, job["name"], spec, identity):
        raise ValueError("trial binding changed; use a new output directory")
    if any(sha(root / filename) != value for filename, value in record["artifacts"].items()):
        raise ValueError("completed tuning trial artifact changed")
    payload = validate_st_checkpoint(root / "checkpoint.json")
    config, chunk = effective(job, spec)
    if (payload["estimator"]["config"] != config or partitions(payload) != partitions(job["payload"])
            or payload["training_metadata"].get("mpnet_tuning") != record["binding"]
            or payload["feature_builder"]["encoder"]["chunk_tokens"] != chunk):
        raise ValueError("completed tuning checkpoint config/partition mismatch")
    return reuse.read(root / "training_summary.json")


def fit_trial(args, job, spec, manifest, encoder, identity, fit, dev):
    from .runtime_gate import save_gate_checkpoint
    from .st_estimator import STUtilityEstimator

    root = trial_root(args, job["name"], spec)
    estimator = STUtilityEstimator(**effective(job, spec)[0], device=args.train_device)
    if not estimator.fit(fit, progress=True):
        raise ValueError("insufficient fit supervision; refusing an untrained head")
    metrics = utility_metrics(estimator, dev)
    binding = binding_for(manifest, job["name"], spec, identity)
    metadata = deepcopy(job["payload"]["training_metadata"])
    metadata.update(mpnet_tuning=binding, encoder_identity=identity,
        policy_parameter_origin="unchanged_original_ST_dev_selection",
        original_policy_calibration=deepcopy(metadata["policy_calibration"]),
        training_events=len(fit) + len(dev), estimator_fit_events=len(fit), calibration_events=0,
        dev_evaluation_events=len(dev))
    metadata["policy_calibration"] = {k: metadata["original_policy_calibration"][k] for k in POLICY_KEYS}
    metadata["policy_calibration"].update(enabled=False, selection_scope="fixed_original_parameters")
    metadata["estimator"].update(config=estimator.config, training_diagnostics=estimator.training_diagnostics)
    summary = {"binding": binding, "estimator_config": estimator.config, "encoder": encoder.config(),
        "fit_tasks": sorted(job["fit"]), "dev_tasks": sorted(job["dev"]),
        "fit_events": len(fit), "dev_metrics": metrics, "encoder_trained": False,
        "training_diagnostics": estimator.training_diagnostics, "loss_history": estimator.loss_history}
    save(root / "training_summary.json", summary)
    save_gate_checkpoint(root / "checkpoint.json", estimator, feature_version=3,
        hash_dimensions=job["payload"]["feature_builder"].get("hash_dimensions", 256),
        feature_config={"encoder": encoder.config()}, training_metadata=metadata)
    locked(root / "trial.json", {"binding": binding,
        "artifacts": {p: sha(root / p) for p in ("checkpoint.json", "training_summary.json")}})
    return summary


def select_shared(manifest, summaries):
    scores = []
    for spec in manifest["trials"]:
        relative = {}
        for name in manifest["datasets"]:
            anchor = summaries[name]["original"]["dev_metrics"]["class_balanced_mse"]
            value = summaries[name][spec["id"]]["dev_metrics"]["class_balanced_mse"]
            if not math.isfinite(value) or value < 0 or not math.isfinite(anchor) or anchor < 0:
                raise ValueError("invalid dev selection metrics")
            relative[name] = value / max(anchor, SELECTION["normalizer_floor"])
        scores.append({"trial_id": spec["id"], "score": statistics.mean(relative.values()),
                       "per_dataset_relative_mse": relative})
    ranked = sorted(scores, key=lambda row: row["score"])  # stable, original wins exact ties
    return {"manifest_sha256": digest(manifest), "criterion": SELECTION,
        "selected_trial": ranked[0]["trial_id"], "leaderboard": ranked}


def dev_report(args, selection):
    lines = ["| Trial | Shared dev score (lower is better) | " + " | ".join(args.datasets) + " |",
             "| --- | --- | " + " | ".join("---" for _ in args.datasets) + " |"]
    for row in selection["leaderboard"]:
        lines.append("| " + " | ".join([row["trial_id"], f"{row['score']:.6f}",
            *(f"{row['per_dataset_relative_mse'][n]:.6f}" for n in args.datasets)]) + " |")
    lines += ["", f"Selected shared configuration: {selection['selected_trial']}",
              "Dev utility prediction errors only; these are not task accuracy/F1 results."]
    (args.output_dir / "dev_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_trials(args, jobs, manifest, *, train):
    summaries, selected_hashes = {}, {}
    for name, job in jobs.items():
        summaries[name], selected_hashes[name] = {}, {}
        chunks = dict.fromkeys(effective(job, s)[1] for s in manifest["trials"])
        for chunk in chunks:
            specs = [s for s in manifest["trials"] if effective(job, s)[1] == chunk]
            encoder, identity = checked_encoder(args, name, chunk)
            fit, dev = [], []
            try:
                pending = [s for s in specs if not (trial_root(args, name, s) / "trial.json").exists()]
                if pending and not train:
                    raise ValueError("tuning is incomplete; finish --stage tune before test evaluation")
                fit, dev = load_examples(job, encoder) if pending else ([], [])
                for spec in specs:
                    root = trial_root(args, name, spec)
                    print(f"[MPNet tuning] {name}/{spec['id']} chunk={chunk} "
                          + ("resume" if (root / "trial.json").exists() else "fit"), flush=True)
                    summary = (require_trial(args, job, spec, manifest, identity) if (root / "trial.json").exists()
                               else fit_trial(args, job, spec, manifest, encoder, identity, fit, dev))
                    summaries[name][spec["id"]] = {"dev_metrics": summary["dev_metrics"]}
                    selected_hashes[name][spec["id"]] = sha(root / "checkpoint.json")
                    del summary
            finally:
                encoder.close()
                # Release the large feature arrays before loading another chunk.
                del fit, dev
    selection = select_shared(manifest, summaries)
    choice = selection["selected_trial"]
    selection["selected_checkpoints_sha256"] = {n: values[choice] for n, values in selected_hashes.items()}
    selection["trial_checkpoints_sha256"] = selected_hashes
    locked(args.output_dir / "selection.json", selection)
    dev_report(args, selection)
    return selection


def same_rule_reference(item, rule):
    """Use an optional MiniLM rule retest only when predictions/retrieval match."""
    old = reuse.indexed(reuse.rows(item["old"] / "memory_gate.jsonl"))
    ref_path = item.get("rule_reference")
    other = reuse.indexed(reuse.rows(ref_path)) if ref_path else {}
    if other and set(other) != set(old):
        raise ValueError("MiniLM rule reference has different test IDs")
    matched = []
    for task_id, left in old.items():
        expected, _ = reuse.selection(left, rule)
        found = None
        for row in (left, other.get(task_id)):
            if row is None:
                continue
            if sorted(d["memory_id"] for d in row["decisions"] if d["action"].upper() == "DROP") != expected:
                continue
            keys = ("memory_id", "prediction", "memory_content", "memory_metadata", "retrieval_rank")
            if (row["retrieval_fingerprint"] != left["retrieval_fingerprint"] or
                    [[d.get(k) for k in keys] for d in row["decisions"]] !=
                    [[d.get(k) for k in keys] for d in left["decisions"]]):
                raise ValueError("MiniLM same-rule reference candidates/predictions changed")
            found = row
            break
        if found is None:
            return None
        matched.append(found)
    return matched


def validate_mpnet_reference(args, job, manifest, item):
    """Reject incompatible optional controls before generating any test answers."""
    name = job["name"]
    path = item["mpnet_reference"]
    checkpoint = args.mpnet_reference_root / name / "checkpoint.json"
    run = reuse.read(path / "run_manifest.json")["run"]
    original = validate_st_checkpoint(checkpoint)
    encoder = original["feature_builder"]["encoder"]
    if (run["checkpoint_sha256"] != sha(checkpoint)
            or any(run[k] != getattr(item["args"], k) for k in ("kappa", "delta", "max_drops"))
            or original["estimator"]["config"] != job["payload"]["estimator"]["config"]
            or partitions(original) != partitions(job["payload"])
            or original["training_metadata"]["training_source_sha256"] !=
               job["payload"]["training_metadata"]["training_source_sha256"]
            or encoder["model"] != args.semantic_model
            or encoder["chunk_tokens"] != effective(job, manifest["trials"][0])[1]):
        raise ValueError("original MPNet reference uses different head, labels, encoder or deletion parameters")
    current = reuse.read(args.output_dir / name / "features" /
        f"chunk_{encoder['chunk_tokens']}" / "encoder_identity.json")
    if original["training_metadata"].get("encoder_identity") != current:
        raise ValueError("original MPNet reference has different frozen weights")
    if resolve_rule("dataset_default", name) == "positive_unlimited":
        for row in reuse.rows(path / "memory_gate.jsonl"):
            expected, _ = reuse.selection(row, "positive_unlimited")
            actual = sorted(d["memory_id"] for d in row["decisions"] if d["action"].upper() == "DROP")
            if expected != actual:
                raise ValueError("original MPNet reference did not apply positive_unlimited")


def evaluate(args, jobs, manifest, selection):
    from .compare_gate_runs import compare
    from .interactive_progress import run_logged
    from .run_full_gate_benchmark import merge_caches

    eval_identity, results = {}, {}
    items = {}
    # This is deliberately the first stage that reads any held-out outcomes.
    for name in args.datasets:
        suite = SimpleNamespace(**{**vars(args), "st_deletion_rule": "dataset_default"})
        item = items[name] = reuse.prepare_qa(suite, name, eval_identity)
        ref = args.rule_reference_root / name / "semantic_st/memory_gate.jsonl"
        if ref.is_file():
            item["rule_reference"] = ref
            reuse.pin(eval_identity, ref)
        mpnet = args.mpnet_reference_root / name / "semantic_st"
        if (mpnet / "memory_gate.jsonl").is_file():
            for path in (mpnet / "memory_gate.jsonl", mpnet / "run_manifest.json",
                         args.mpnet_reference_root / name / "checkpoint.json"):
                reuse.pin(eval_identity, path)
            item["mpnet_reference"] = mpnet
            validate_mpnet_reference(args, jobs[name], manifest, item)
        item["minilm_rows"] = same_rule_reference(item, resolve_rule("dataset_default", name))
        for cache in (item["baseline"] / "llm_cache.sqlite",
                      ref.parent / "llm_cache.sqlite", mpnet / "llm_cache.sqlite"):
            if cache.is_file():
                if any(Path(str(cache) + suffix).exists() for suffix in ("-wal", "-journal")):
                    raise ValueError(f"reference cache must be closed: {cache}")
                reuse.pin(eval_identity, cache)
    locked(args.output_dir / "evaluation_manifest.json", {"manifest_sha256": digest(manifest),
        "selection_sha256": sha(args.output_dir / "selection.json"), "source_sha256": eval_identity})
    spec = next(s for s in manifest["trials"] if s["id"] == selection["selected_trial"])
    for name, item in items.items():
        child, adapter = deepcopy(item["args"]), item["adapter"]
        child.checkpoint = trial_root(args, name, spec) / "checkpoint.json"
        encoder_config = reuse.read(child.checkpoint)["feature_builder"]["encoder"]
        child.semantic_model, child.semantic_revision = encoder_config["model"], encoder_config.get("revision")
        child.memory_dir = reuse.private_memory(item["source_memory"], args.output_dir / name / "frozen_memory/g-memory")
        child.semantic_cache = args.output_dir / name / "features" / f"chunk_{encoder_config['chunk_tokens']}" / "semantic.sqlite"
        output = args.output_dir / name / "semantic_st"
        caches = [Path(p) for p in eval_identity if Path(p).name == "llm_cache.sqlite"
                  and (p.startswith(str(args.qa_root / name) + "/")
                       or p.startswith(str(args.rule_reference_root / name) + "/")
                       or p.startswith(str(args.mpnet_reference_root / name) + "/"))]
        seed_cache = args.output_dir / name / "seed_cache.sqlite"
        merge_caches(caches, seed_cache)
        host = adapter.evaluation_command(child, mode="learned", output=output,
                                          resume=(output / "run_manifest.json").exists())
        host[host.index("--cache-seed-from") + 1] = str(seed_cache)
        rule = resolve_rule("dataset_default", name)
        command = [*host[:3], "causal_memory_control.st_encoder_evaluate", "--dataset", name,
                   "--reference-log", str(item["old"] / "memory_gate.jsonl"),
                   "--st-deletion-rule", rule, *host[4:]]
        if not adapter.evaluation_complete(child, output, mode="learned", command=host):
            run_logged(command, args.output_dir / "logs" / (name + ".log"), cwd=Path.cwd())
        if not adapter.evaluation_complete(child, output, mode="learned", command=host):
            raise ValueError("selected MPNet test evaluation is incomplete")
        target = output / "memory_gate.jsonl"
        comparison = compare(item["baseline"] / "memory_gate.jsonl", target,
                             bootstrap_samples=2000, seed=child.gate_evaluation_seed)
        if not comparison["paired_control_valid"] or not comparison["policy_valid"]:
            raise ValueError("selected MPNet baseline/retrieval audit failed")
        save(output / "comparison.json", comparison)
        result = results[name] = {"n": comparison["paired_tasks"], "rule": rule,
            "baseline": comparison["baseline_mean_reward"], "tuned_mpnet": comparison["gate_mean_reward"],
            "original_minilm_same_rule": None, "original_mpnet_same_rule": None}
        if name == "hotpotqa":
            result.update(baseline_em=comparison["baseline_answer_em"], tuned_mpnet_em=comparison["gate_answer_em"])
        minilm = item["minilm_rows"]
        references = {}
        if minilm:
            path = args.output_dir / name / "original_minilm_same_rule.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in minilm), encoding="utf-8")
            references["original_minilm_same_rule"] = path
        if item.get("mpnet_reference"):
            references["original_mpnet_same_rule"] = item["mpnet_reference"] / "memory_gate.jsonl"
        for label, path in references.items():
            baseline_audit = compare(item["baseline"] / "memory_gate.jsonl", path,
                                    bootstrap_samples=2000, seed=child.gate_evaluation_seed)
            paired = compare_encoders(path, target, seed=child.gate_evaluation_seed)
            if not all(r["paired_control_valid"] and r["policy_valid"] for r in (baseline_audit, paired)):
                raise ValueError(f"{label}: held-out paired comparison failed")
            save(output / (label + "_comparison.json"), paired)
            records = reuse.rows(path)
            result[label] = statistics.mean(r["reward"] for r in records)
            if name == "hotpotqa":
                result[label + "_em"] = statistics.mean(r["outcome"]["answer_em"] for r in records)
    save(args.output_dir / "summary.json", {"schema": SCHEMA, "selected_trial": selection["selected_trial"],
        "selection_scope": "dev_only", "datasets": results})
    lines = ["| Dataset | Metric | N | Always Use (%) | Original MiniLM, same rule (%) | Original MPNet, same rule (%) | Tuned MPNet (%) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, result in results.items():
        for metric, suffix in (("Accuracy", ""),) if name == "fever" else (("Answer F1", ""), ("Answer EM", "_em")):
            values = [result.get(k + suffix) for k in ("baseline", "original_minilm_same_rule", "original_mpnet_same_rule", "tuned_mpnet")]
            lines.append("| " + " | ".join([name, metric, str(result["n"]),
                *("—" if v is None else f"{100*v:.2f}" for v in values)]) + " |")
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    args = parse_args(argv)
    jobs, manifest = prepare(args)
    if args.stage == "plan":
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest
    if args.stage == "report":
        if reuse.read(args.output_dir / "manifest.json") != manifest:
            raise ValueError("registered input/configuration changed")
        for filename in ("dev_results.md", "results.md"):
            path = args.output_dir / filename
            if path.exists():
                print(path.read_text())
        return reuse.read(args.output_dir / "selection.json")
    with output_lock(args.output_dir), offline_models():
        locked(args.output_dir / "manifest.json", manifest)
        if args.stage == "evaluate" and not (args.output_dir / "selection.json").is_file():
            raise ValueError("run --stage tune before the explicit selected-only test stage")
        selection = collect_trials(args, jobs, manifest, train=args.stage == "tune")
        if args.stage == "evaluate":
            return evaluate(args, jobs, manifest, selection)
        print(f"[完成] MPNet 开发集共享配置：{selection['selected_trial']}；测试评测尚未启动。", flush=True)
        return selection


if __name__ == "__main__":
    main()
