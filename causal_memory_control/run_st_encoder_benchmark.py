"""Independent frozen-encoder ablation using existing ST diagnostic labels.

Changing embedding coordinates requires a newly fitted ST head. This command
never collects support/diagnostic data and never trains an encoder. Planning is
the default; fitting a head additionally requires explicit --refit-head.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import statistics
import sys

from . import reevaluate_st_policy as reuse
from .interactive_data import digest, locked, save, sha
from .st_policy import PositiveSTGate, RULES, resolve_rule, validate_st_checkpoint


SCHEMA = "semantic-st-frozen-encoder-comparison-v1"
DEFAULT_ENCODER = "sentence-transformers/all-mpnet-base-v2"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--datasets", default="fever,hotpotqa")
    parser.add_argument("--qa-root", type=Path, default=reuse.DEFAULT_RESULTS / "all_trained_gates_with_old_st_retest_v2")
    parser.add_argument("--interactive-root", type=Path,
                        default=reuse.DEFAULT_RESULTS / "pddl_alfworld_gate_expanded_v10_formatwarn_no_ce")
    parser.add_argument("--rule-reference-root", type=Path,
                        default=reuse.DEFAULT_RESULTS / "st_positive_unlimited_retest_v1",
                        help="Optional completed ST rule retest, for comparing encoders under the SAME rule")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-model", default=DEFAULT_ENCODER,
                        help="Frozen SentenceTransformer model or local directory; does not change GMemory retrieval")
    parser.add_argument("--semantic-revision")
    parser.add_argument("--train-device", default="cpu", help="Device for the new small ST heads only")
    parser.add_argument("--st-deletion-rule", choices=RULES, default="dataset_default")
    parser.add_argument("--stage", choices=("plan", "fit", "evaluate", "all"), default="plan")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--refit-head", action="store_true",
                        help="Explicitly permit fitting a new small head on existing labels; never fits the encoder")
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa", "pddl"}:
        parser.error("select fever,hotpotqa,pddl; ALFWorld is never started")
    if args.stage in ("fit", "all") and not args.plan_only and not args.refit_head:
        parser.error("changing encoder requires a newly fitted ST head; use --stage plan or explicitly --refit-head")
    for key in ("qa_root", "interactive_root", "rule_reference_root", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if Path(args.semantic_model).is_dir():
        args.semantic_model = str(Path(args.semantic_model).resolve())
    for root in (args.qa_root, args.interactive_root, args.rule_reference_root):
        reuse.separate_output(args.output_dir, root)
    return args


def partitions(payload):
    metadata = payload["training_metadata"]
    fit = {str(v) for v in metadata["fit_task_ids"]}
    dev = {str(v) for v in metadata.get("dev_task_ids",
        metadata.get("policy_calibration", {}).get("calibration_task_ids", []))}
    tasks = {str(v) for v in metadata["training_task_ids"]}
    if not fit or not dev or fit & dev or fit | dev != tasks:
        raise ValueError("original ST checkpoint lacks a valid, disjoint fit/dev partition")
    return fit, dev


def prepare(args):
    prepared, original = reuse.prepare(args)
    identity = dict(original["source_sha256"])
    jobs = []
    for name, item in prepared.items():
        if name == "pddl":
            for fold in item["jobs"]:
                path = fold["source"] / "semantic_st/checkpoint.json"
                data = fold["config"]["experiments"]["pddl"]
                paths = [fold["source"] / "diagnostic/tasks" / (digest(t) + ".json")
                         for t in sorted(data["split"]["fit"] + data["split"]["dev"])]
                jobs.append({"name": name, "label": fold["label"], "item": fold,
                    "checkpoint": path if path.exists() else None, "sources": paths,
                    "memory": fold["memory"]})
        else:
            path = item["args"].checkpoint
            metadata = reuse.read(path)["training_metadata"]
            recorded = [Path(p) for p in metadata["training_sources"]]
            sources = [p / "branches.jsonl" if p.is_dir() else p for p in recorded]
            if {str(p): sha(source) for p, source in zip(recorded, sources)} != metadata["training_source_sha256"]:
                raise ValueError(f"{name}: original measured diagnostic sources changed")
            for source in sources:
                reuse.pin(identity, source.parent / "run_manifest.json")
            reuse.pin(identity, item["baseline"] / "llm_cache.sqlite")
            jobs.append({"name": name, "label": name, "item": item, "checkpoint": path,
                         "sources": sources, "memory": item["source_memory"]})
    plans = []
    for job in jobs:
        for path in job["sources"]:
            reuse.pin(identity, path)
        if job["checkpoint"] is None:
            plans.append({"dataset": job["label"], "head_status": "skipped_no_candidate_supervision"})
            continue
        payload = validate_st_checkpoint(job["checkpoint"])
        if job["name"] == "pddl":
            metadata = payload["training_metadata"]
            measured = {reuse.read(p)["task_id"]: sha(p) for p in job["sources"]}
            if (measured != metadata["diagnostic_sha256"] or
                    metadata["config_digest"] != digest(job["item"]["config"])):
                raise ValueError("PDDL original checkpoint diagnostic labels/configuration changed")
        encoder = payload["feature_builder"]["encoder"]
        if (encoder["model"], encoder.get("revision")) == (args.semantic_model, args.semantic_revision):
            raise ValueError("select a different encoder/revision; this is already the original ST encoder")
        fit, dev = partitions(payload)
        job["payload"] = payload
        plans.append({"dataset": job["label"], "old_encoder": encoder,
            "new_encoder": args.semantic_model, "fit_tasks": len(fit), "dev_tasks": len(dev),
            "small_head_refit_required": True, "encoder_training": False,
            "estimator_config": payload["estimator"]["config"],
            "deletion_rule": resolve_rule(args.st_deletion_rule, job["name"]),
            "deployment_parameters": ({k: getattr(job["item"]["args"], k) for k in ("kappa", "delta", "max_drops")}
                if job["name"] != "pddl" else payload["training_metadata"]["policy_calibration"]),
            "calibration": "original numeric kappa/delta/budget reused; no new dev selection"})
    for path in [Path(__file__), *(Path(__file__).with_name(name + ".py") for name in (
            "st_encoder_evaluate", "st_estimator", "semantic_features", "st_data", "train_gate",
            "runtime_gate", "features", "gmemory_adapter", "task_state", "dataset_adapters",
            "fever_experiment", "hotpotqa_experiment", "compare_gate_runs"))]:
        reuse.pin(identity, path)
    manifest = {"schema": SCHEMA, "encoder": {"model": args.semantic_model, "revision": args.semantic_revision},
        "train_device": args.train_device, "datasets": args.datasets, "jobs": plans,
        "rules": {name: resolve_rule(args.st_deletion_rule, name) for name in args.datasets},
        "new_diagnostic_tasks": 0, "encoder_training": False,
        "original_results_overwritten": False, "pddl_requires_original_root_lock": "pddl" in args.datasets,
        "source_sha256": identity}
    # Pin an optional same-rule reference rather than comparing two different policies silently.
    for name in args.datasets:
        path = args.rule_reference_root / name / ("policies.jsonl" if name == "pddl" else "semantic_st/memory_gate.jsonl")
        if path.is_file():
            reuse.pin(identity, path)
            prepared[name]["rule_reference"] = path
            cache = path.parent / "llm_cache.sqlite"
            if name != "pddl" and cache.is_file():
                if any(Path(str(cache) + suffix).exists() for suffix in ("-wal", "-journal")):
                    raise ValueError(f"a completed, closed rule-reference cache is required: {cache}")
                reuse.pin(identity, cache)
                prepared[name]["rule_reference_cache"] = cache
            if name == "pddl":
                stores = prepared[name]["reference_stores"] = {}
                for fold in prepared[name]["jobs"]:
                    store = args.rule_reference_root / fold["label"] / "actions"
                    if not (store / "identity.json").is_file():
                        continue
                    registered = reuse.read(store / "identity.json")
                    if (registered.get("schema") != reuse.SCHEMA or
                            registered.get("original_action_identity") !=
                            reuse.read(fold["source"] / "evaluation/actions/identity.json")):
                        raise ValueError("PDDL rule-reference action store has different original inputs")
                    for artifact in [store / "identity.json", *sorted((store / "parts").glob("*.json"))]:
                        reuse.pin(identity, artifact)
                    stores[fold["label"]] = store
    return prepared, jobs, manifest


@contextmanager
def offline_models():
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update({key: "1" for key in keys})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def head_binding(args, job):
    return {"experiment": SCHEMA, "original_checkpoint_sha256": sha(job["checkpoint"]) if job["checkpoint"] else None,
        "model": args.semantic_model, "revision": args.semantic_revision,
        "diagnostic_sha256": {str(p): sha(p) for p in job["sources"]}}


def checked_encoder(args, job):
    """Verify real weights before reading any cached vectors; pin Hub commits."""
    from .csm_st_worker import dedup_identity
    from .semantic_features import SentenceTransformerEmbedder

    output = args.output_dir / job["label"]
    encoder = SentenceTransformerEmbedder(args.semantic_model, revision=args.semantic_revision,
        chunk_tokens=job["payload"]["feature_builder"]["encoder"]["chunk_tokens"],
        cache_path=output / "semantic.sqlite")
    try:
        spec = {k: v for k, v in encoder.config().items() if k != "dimensions"}
        identity = dedup_identity(encoder, spec)
        if not identity.get("local_model_sha256") and not identity.get("resolved_commit"):
            raise ValueError("cannot resolve frozen encoder weights; use a complete local model directory")
        locked(output / "encoder_identity.json", identity)
        if not Path(args.semantic_model).is_dir():
            encoder.revision = identity["resolved_commit"]
        return encoder, identity
    except BaseException:
        encoder.close()
        raise


def fit_head(args, job):
    """Refit only the ST ensemble; use the old task split, loss and settings."""
    output = args.output_dir / job["label"]
    checkpoint = output / "checkpoint.json"
    binding = head_binding(args, job)
    if job["checkpoint"] is None:
        locked(output / "training_status.json", {**binding, "trained": False,
            "status": "skipped_no_candidate_supervision", "deployment": "empty_candidates_only"})
        return None
    if checkpoint.exists():
        return require_head(args, job)
    from .estimator import PotentialOutcomeExample
    from .runtime_gate import save_gate_checkpoint
    from .semantic_features import SemanticTeamFeatureBuilder
    from .st_data import CompactFeatures, load_st_examples
    from .st_estimator import STUtilityEstimator

    old = job["payload"]
    fit_ids, dev_ids = partitions(old)
    encoder, identity = checked_encoder(args, job)
    builder = SemanticTeamFeatureBuilder(encoder)
    try:
        if job["name"] == "pddl":
            from .interactive_runtime import event_from_row
            config = job["item"]["config"]
            examples, layouts = [], {}
            for path in job["sources"]:
                row = reuse.read(path)
                if (row["config_digest"] != digest(config) or row["repeats"] != config["repeats"]
                        or len(row["q_drop"]) != len(row["candidates"])):
                    raise ValueError("PDDL measured diagnostic configuration/coverage changed")
                for i in range(len(row["candidates"])):
                    event = event_from_row(row, i)
                    examples.append(PotentialOutcomeExample(CompactFeatures(builder.build(event), layouts),
                        row["q_use"], row["q_drop"][i], event.event_id, row["task_id"]))
        else:
            examples = load_st_examples(job["sources"], metric=old["training_metadata"]["metric"],
                task_ids={int(t) for t in fit_ids | dev_ids},
                hash_dimensions=old["feature_builder"].get("hash_dimensions", 256), feature_version=3,
                memory_dir=job["memory"], feature_builder=builder, event_sink=[])
        fit = [e for e in examples if str(e.group_id) in fit_ids]
        dev = [e for e in examples if str(e.group_id) in dev_ids]
        original_coverage = old["training_metadata"].get("candidate_coverage")
        expected_fit = {str(v) for v in original_coverage["fit"]["supervised_task_ids"]} if original_coverage else fit_ids
        expected_dev = {str(v) for v in original_coverage["dev"]["supervised_task_ids"]} if original_coverage else dev_ids
        if {str(e.group_id) for e in fit} != expected_fit or {str(e.group_id) for e in dev} != expected_dev:
            raise ValueError("new encoder examples do not preserve the original supervised task split")
        estimator = STUtilityEstimator(**{**old["estimator"]["config"], "device": args.train_device})
        if not estimator.fit(fit, progress=True):
            raise ValueError("insufficient measured fitting examples; refusing an untrained ST head")
        metadata = deepcopy(old["training_metadata"])
        metadata["encoder_comparison"] = binding
        metadata["encoder_identity"] = identity
        metadata["policy_parameter_origin"] = "unchanged_original_ST_dev_selection"
        metadata["original_policy_calibration"] = metadata["policy_calibration"]
        metadata["policy_calibration"] = {k: metadata["original_policy_calibration"][k]
            for k in ("recommended_kappa", "recommended_max_drops", "delta", "calibration_task_ids")}
        metadata["policy_calibration"].update(enabled=False, selection_scope="fixed_original_encoder_parameters")
        metadata.update(training_events=len(examples), estimator_fit_events=len(fit), calibration_events=0,
            dev_evaluation_events=len(dev))
        if isinstance(metadata.get("estimator"), dict):
            metadata["estimator"].update(config=estimator.config, training_diagnostics=estimator.training_diagnostics)
        errors = []
        for start in range(0, len(dev), 256):
            batch = dev[start:start + 256]
            errors += [(p.utility - e.q_use + e.q_drop) ** 2
                       for p, e in zip(estimator.predict_many([e.features for e in batch]), batch)]
        save(output / "training_summary.json", {"encoder_comparison": binding,
            "estimator_config": estimator.config, "fit_tasks": sorted(fit_ids), "dev_tasks": sorted(dev_ids),
            "fit_events": len(fit), "dev_events": len(dev), "dev_utility_rmse": statistics.mean(errors) ** .5,
            "training_diagnostics": estimator.training_diagnostics, "loss_history": estimator.loss_history,
            "policy_parameters_reused": metadata["policy_calibration"], "encoder_trained": False})
        save_gate_checkpoint(checkpoint, estimator, feature_version=3,
            hash_dimensions=old["feature_builder"].get("hash_dimensions", 256),
            feature_config={"encoder": encoder.config()}, training_metadata=metadata)
        return checkpoint
    finally:
        encoder.close()


def require_head(args, job):
    output = args.output_dir / job["label"]
    if job["checkpoint"] is None:
        status = reuse.read(output / "training_status.json")
        if status != {**head_binding(args, job), "trained": False,
                "status": "skipped_no_candidate_supervision", "deployment": "empty_candidates_only"}:
            raise ValueError("invalid empty-only training status")
        return None
    checkpoint = output / "checkpoint.json"
    payload = validate_st_checkpoint(checkpoint)
    if (payload["training_metadata"].get("encoder_comparison") != head_binding(args, job)
            or payload["estimator"]["config"] != job["payload"]["estimator"]["config"]
            or partitions(payload) != partitions(job["payload"])):
        raise ValueError("new ST checkpoint changed encoder protocol, loss settings or fit/dev split")
    encoder, identity = checked_encoder(args, job)
    try:
        actual = {k: v for k, v in payload["feature_builder"]["encoder"].items() if k != "dimensions"}
        if (payload["training_metadata"].get("encoder_identity") != identity
                or actual != {k: v for k, v in encoder.config().items() if k != "dimensions"}):
            raise ValueError("new ST checkpoint encoder identity differs from its frozen weights")
        for key in ("recommended_kappa", "recommended_max_drops", "delta", "calibration_task_ids"):
            if payload["training_metadata"]["policy_calibration"][key] != job["payload"]["training_metadata"]["policy_calibration"][key]:
                raise ValueError("original deletion parameters changed")
        from .st_estimator import STUtilityEstimator
        STUtilityEstimator.from_dict(payload["estimator"])
    finally:
        encoder.close()
    return checkpoint


def compare_encoders(old_path, new_path, *, seed):
    """Both policies may drop memories; only equal selections must match."""
    from .compare_gate_runs import compare

    result = compare(old_path, new_path, bootstrap_samples=2000, seed=seed)
    old, new = reuse.indexed(reuse.rows(old_path)), reuse.indexed(reuse.rows(new_path))
    mismatches = []
    for task_id, left in old.items():
        right = new[task_id]
        drops = lambda row: sorted(d["memory_id"] for d in row["decisions"] if d["action"].upper() == "DROP")
        if drops(left) == drops(right):
            fields = [k for k in ("reward", "done") if left[k] != right[k]]
            if left.get("outcome", {}).get("prediction") != right.get("outcome", {}).get("prediction"):
                fields.append("prediction")
            if fields:
                mismatches.append({"task_id": task_id, "different_fields": fields})
    result.pop("no_drop_mismatches")
    result.update(reference_role="original_encoder_same_rule", same_selection_mismatches=mismatches,
        paired_control_valid=not mismatches and not result["retrieval_mismatches"])
    return result


def run_qa(args, name, item, job):
    from .compare_gate_runs import compare
    from .run_full_gate_benchmark import merge_caches
    from .interactive_progress import run_logged

    child, adapter = deepcopy(item["args"]), item["adapter"]
    child.checkpoint = require_head(args, job)
    payload = reuse.read(child.checkpoint)
    child.semantic_model = payload["feature_builder"]["encoder"]["model"]
    child.semantic_revision = payload["feature_builder"]["encoder"].get("revision")
    rule = resolve_rule(args.st_deletion_rule, name)
    if rule == "positive_unlimited":
        child.kappa, child.delta, child.max_drops = 0.0, 0.0, -1
    child.memory_dir = reuse.private_memory(item["source_memory"], args.output_dir / name / "frozen_memory/g-memory")
    output = args.output_dir / name / "semantic_st"
    child.semantic_cache = args.output_dir / name / "semantic.sqlite"
    seed_cache = args.output_dir / name / "seed_cache.sqlite"
    caches = [item["cache"], item["baseline"] / "llm_cache.sqlite"]
    if item.get("rule_reference_cache"):
        caches.append(item["rule_reference_cache"])
    merge_caches(caches, seed_cache)
    host = adapter.evaluation_command(child, mode="learned", output=output,
                                      resume=(output / "run_manifest.json").exists())
    host[host.index("--cache-seed-from") + 1] = str(seed_cache)
    command = [*host[:3], "causal_memory_control.st_encoder_evaluate", "--dataset", name,
        "--reference-log", str(item["old"] / "memory_gate.jsonl"), "--st-deletion-rule", rule, *host[4:]]
    if not adapter.evaluation_complete(child, output, mode="learned", command=host):
        run_logged(command, args.output_dir / "logs" / (name + ".log"), cwd=Path.cwd())
    if not adapter.evaluation_complete(child, output, mode="learned", command=host):
        raise ValueError("encoder evaluation did not produce complete held-out results")
    comparison = compare(item["baseline"] / "memory_gate.jsonl", output / "memory_gate.jsonl",
                         bootstrap_samples=2000, seed=child.gate_evaluation_seed)
    if not comparison["paired_control_valid"] or not comparison["policy_valid"]:
        raise ValueError("new-encoder paired baseline/retrieval audit failed")
    save(output / "comparison.json", comparison)
    old_rows = reuse.indexed(reuse.rows(item["old"] / "memory_gate.jsonl"))
    reference = reuse.indexed(reuse.rows(item["rule_reference"])) if item.get("rule_reference") else {}
    if reference and set(reference) != set(old_rows):
        raise ValueError("same-rule reference has different evaluation task IDs")
    same_rule = []
    for task_id, old in old_rows.items():
        drops, _ = reuse.selection(old, rule)
        ref = next((r for r in (old, reference.get(task_id)) if r is not None and
            sorted(d["memory_id"] for d in r["decisions"] if d["action"].upper() == "DROP") == drops), None)
        if ref is None:
            same_rule = []
            break
        if (ref["retrieval_fingerprint"] != old["retrieval_fingerprint"] or
                [[d.get(k) for k in ("memory_id", "prediction", "memory_content", "memory_metadata", "retrieval_rank")]
                 for d in ref["decisions"]] != [[d.get(k) for k in
                ("memory_id", "prediction", "memory_content", "memory_metadata", "retrieval_rank")]
                 for d in old["decisions"]]):
            raise ValueError("same-rule reference uses different original candidates/predictions")
        same_rule.append(ref)
    result = {"rule": rule, "n": comparison["paired_tasks"], "metric": "Accuracy" if name == "fever" else "Answer F1",
        "baseline": comparison["baseline_mean_reward"], "old_encoder_same_rule":
        statistics.mean(r["reward"] for r in same_rule) if same_rule else None,
        "new_encoder": comparison["gate_mean_reward"], "comparison": comparison}
    if name == "hotpotqa":
        result.update(baseline_em=comparison["baseline_answer_em"], new_em=comparison["gate_answer_em"], old_em=
            statistics.mean(r["outcome"]["answer_em"] for r in same_rule) if same_rule else None)
    if same_rule:
        reference_log = args.output_dir / name / "old_encoder_same_rule.jsonl"
        reference_log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in same_rule))
        audit = compare(item["baseline"] / "memory_gate.jsonl", reference_log,
            bootstrap_samples=2000, seed=child.gate_evaluation_seed)
        if not audit["paired_control_valid"] or not audit["policy_valid"]:
            raise ValueError("old-encoder same-rule reference failed its baseline audit")
        paired = compare_encoders(reference_log, output / "memory_gate.jsonl", seed=child.gate_evaluation_seed)
        if not paired["paired_control_valid"] or not paired["policy_valid"]:
            raise ValueError("old/new encoder comparison is not paired under the same rule")
        save(output / "encoder_comparison.json", paired)
    return result


def run_pddl(args, item, jobs):
    from .interactive_runtime import ActionStore, NativeRunner, execute
    from .interactive_training import select
    from .interactive_worker import verify_inputs
    from .runtime_gate import GMemoryExposureGate

    rule = resolve_rule(args.st_deletion_rule, "pddl")
    records = []
    for job in jobs:
        fold = job["item"]
        source, config = fold["source"], fold["config"]
        output = args.output_dir / job["label"]
        checkpoint = require_head(args, job)
        gate = runner = None
        if checkpoint:
            policy = reuse.read(checkpoint)["training_metadata"]["policy_calibration"]
            cls = PositiveSTGate if rule == "positive_unlimited" else GMemoryExposureGate
            gate = cls.from_checkpoint(checkpoint, mode="learned", semantic_cache_path=output / "semantic.sqlite",
                kappa=policy["recommended_kappa"], delta=policy["delta"], max_drops=policy["recommended_max_drops"])
        identity = reuse.read(source / "evaluation/actions/identity.json")
        original = ActionStore(source / "evaluation/actions", identity)
        reference_path = item.get("reference_stores", {}).get(job["label"])
        reference = ActionStore(reference_path, reuse.read(reference_path / "identity.json")) if reference_path else None
        new = ActionStore(output / "actions", {"schema": SCHEMA, "rule": rule,
            "checkpoint_sha256": sha(checkpoint) if checkpoint else None, "original_identity": identity})
        tasks = {t["id"]: t for t in config["experiments"]["pddl"]["tasks"]}
        try:
            for work in fold["work"]:
                baseline = work["baseline"]
                drops, decisions = select(gate, baseline)
                measured = original.get(work["task_id"], baseline["seed"], baseline, drops)
                if measured is None and reference is not None:
                    measured = reference.get(work["task_id"], baseline["seed"], baseline, drops)
                branch = measured or new.get(work["task_id"], baseline["seed"], baseline, drops)
                if branch is None:
                    if runner is None:
                        verify_inputs(config, "pddl")
                        reuse.copy_cache(source / "responses.sqlite", output / "responses.sqlite")
                        runner = NativeRunner(config, output, fold["memory"])
                    branch = execute(runner, new, tasks[work["task_id"]], baseline["seed"], drops, baseline)
                old_drops, _ = reuse.selection(work["old"], rule)
                old_branch = original.get(work["task_id"], baseline["seed"], baseline, old_drops)
                if old_branch is None and reference is not None:
                    old_branch = reference.get(work["task_id"], baseline["seed"], baseline, old_drops)
                record = {"task_id": work["task_id"], "fold": job["label"], "rule": rule,
                    "baseline_outcome": baseline["outcome"], "outcome": branch["outcome"],
                    "old_encoder_same_rule_outcome": old_branch["outcome"] if old_branch else None,
                    "retrieval_fingerprint": baseline["retrieval_fingerprint"], "seed": baseline["seed"],
                    "drop_ids": drops, "decisions": decisions, "measurement_reused": measured is not None,
                    "trained": checkpoint is not None, "checkpoint_sha256": sha(checkpoint) if checkpoint else None}
                save(output / "policy_parts" / (digest(work["task_id"]) + ".json"), record)
                records.append(record)
        finally:
            if gate:
                gate.close()
            if runner:
                runner.close()
    target = args.output_dir / "pddl"
    target.mkdir(exist_ok=True)
    (target / "policies.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    mean = lambda field, key: statistics.mean(r[field][key] for r in records)
    complete = all(r["old_encoder_same_rule_outcome"] is not None for r in records)
    return {"rule": rule, "n": len(records), "metric": "Success", "baseline": mean("baseline_outcome", "success"),
        "new_encoder": mean("outcome", "success"), "old_encoder_same_rule":
        mean("old_encoder_same_rule_outcome", "success") if complete else None,
        "baseline_progress": mean("baseline_outcome", "progress"),
        "new_progress": mean("outcome", "progress"), "old_progress":
        mean("old_encoder_same_rule_outcome", "progress") if complete else None,
        "untrained_empty_only_tasks": sum(not r["trained"] for r in records)}


def report(args, results):
    save(args.output_dir / "summary.json", {"schema": SCHEMA, "encoder": args.semantic_model, "datasets": results})
    table = []
    for name, value in results.items():
        metrics = [(value["metric"], value["baseline"], value["old_encoder_same_rule"], value["new_encoder"])]
        if name == "hotpotqa":
            metrics.append(("Answer EM", value["baseline_em"], value["old_em"], value["new_em"]))
        if name == "pddl":
            metrics.append(("Progress", value["baseline_progress"], value["old_progress"], value["new_progress"]))
        for metric, baseline, old, new in metrics:
            table.append({"Dataset": name, "Rule": value["rule"], "Metric": metric, "N": value["n"],
                "Always Use (%)": 100 * baseline if baseline is not None else None,
                "Original encoder SAME rule (%)": 100 * old if old is not None else None,
                "New encoder (%)": 100 * new})
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    fields = list(table[0])
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in table:
        lines.append("| " + " | ".join("—" if row[k] is None else f"{row[k]:.2f}"
                     if isinstance(row[k], float) else str(row[k]) for k in fields) + " |")
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    args = parse_args(argv)
    planning = args.plan_only or args.stage == "plan"
    from .resume_interactive_dataset import experiment_lock

    with ExitStack() as stack:
        if not planning:
            args.output_dir.parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(experiment_lock(args.output_dir))
            if "pddl" in args.datasets:
                # Same mixed root as the ongoing ALFWorld continuation: fail, never stop it.
                stack.enter_context(experiment_lock(args.interactive_root))
        prepared, jobs, manifest = prepare(args)
        print(json.dumps({**{k: v for k, v in manifest.items() if k != "source_sha256"},
                          "plan_only": planning}, ensure_ascii=False, indent=2), flush=True)
        if planning:
            return manifest
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (args.output_dir / "manifest.json").exists():
            raise ValueError("refusing a nonempty output without an encoder-comparison manifest")
        locked(args.output_dir / "manifest.json", manifest)
        stack.enter_context(offline_models())
        state = {"stage": args.stage, "status": "running"}
        try:
            if args.stage in ("fit", "all"):
                for job in jobs:
                    state["current_dataset"] = job["label"]
                    save(args.output_dir / "progress.json", state)
                    fit_head(args, job)
            if args.stage in ("evaluate", "all"):
                results = {}
                for name, item in prepared.items():
                    state["current_dataset"] = name
                    save(args.output_dir / "progress.json", state)
                    selected = [job for job in jobs if job["name"] == name]
                    results[name] = run_pddl(args, item, selected) if name == "pddl" else run_qa(args, name, item, selected[0])
                    report(args, results)
            state["status"] = "completed"
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            save(args.output_dir / "progress.json", state)
    return args.output_dir


if __name__ == "__main__":
    main()
