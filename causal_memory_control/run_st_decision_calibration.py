"""Diagnose and calibrate existing QA ST decisions, without collecting or fitting.

Only --stage evaluate can generate held-out answers. All new artifacts live in
an independent output directory; original checkpoints and policies are inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from . import reevaluate_st_policy as reuse
from . import run_st_mpnet_tuning as tuning
from .interactive_data import digest, locked, save, sha
from .run_st_encoder_benchmark import offline_models, partitions, compare_encoders
from .st_decision_calibration import PROTOCOL, calibrate, diagnose, groups
from .st_policy import validate_st_checkpoint


SCHEMA = "semantic-st-dev-decision-calibration-v1"
VARIANTS = ("original_minilm", "original_mpnet", "tuned_mpnet")
RESULTS = Path("causal_memory_control/results")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--datasets", default="fever,hotpotqa")
    parser.add_argument("--tuning-root", type=Path, default=RESULTS / "st_mpnet_tuning_qa_gpu3_v1")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "st_decision_calibration_qa_v1")
    parser.add_argument("--stage", choices=("plan", "calibrate", "evaluate", "report"), default="plan")
    args = parser.parse_args(argv)
    args.datasets = list(dict.fromkeys(v.strip() for v in args.datasets.split(",") if v.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("this independent entry supports fever,hotpotqa")
    args.tuning_root, args.output_dir = args.tuning_root.resolve(), args.output_dir.resolve()
    reuse.separate_output(args.output_dir, args.tuning_root)
    return args


def prepare(args):
    """Check splits and input hashes, without encoders, test logs or writes."""
    tuned_manifest = reuse.read(args.tuning_root / "manifest.json")
    selection = reuse.read(args.tuning_root / "selection.json")
    if (tuned_manifest.get("schema") != tuning.SCHEMA
            or selection["manifest_sha256"] != digest(tuned_manifest)
            or not set(args.datasets) <= set(tuned_manifest["datasets"])):
        raise ValueError("invalid original tuning manifest/selection")
    args.qa_root = Path(tuned_manifest["qa_root"])
    args.mpnet_reference_root = Path(tuned_manifest["mpnet_reference_root"])
    args.rule_reference_root = Path(tuned_manifest["rule_reference_root"])
    for source in (args.qa_root, args.mpnet_reference_root, args.rule_reference_root):
        reuse.separate_output(args.output_dir, source)
    # Reuse the strict original split/label/snapshot validators; no trial fits.
    base_args = SimpleNamespace(**{**vars(args), "semantic_model": tuned_manifest["encoder"]["model"],
        "semantic_revision": tuned_manifest["encoder"].get("revision"),
        "train_device": "cpu", "trials_json": None})
    jobs, base_manifest = tuning.prepare(base_args)
    identity = dict(base_manifest["source_sha256"])
    for path in (args.tuning_root / "manifest.json", args.tuning_root / "selection.json"):
        reuse.pin(identity, path)
    for filename in ("run_st_decision_calibration.py", "st_decision_calibration.py", "st_single_drop_evaluate.py"):
        reuse.pin(identity, Path(__file__).with_name(filename))
    for name, job in jobs.items():
        checkpoint = args.tuning_root / name / "trials" / selection["selected_trial"] / "checkpoint.json"
        if sha(checkpoint) != selection["selected_checkpoints_sha256"][name]:
            raise ValueError(f"{name}: selected tuned checkpoint changed")
        paths = {"original_minilm": job["checkpoint"],
                 "original_mpnet": args.mpnet_reference_root / name / "checkpoint.json",
                 "tuned_mpnet": checkpoint}
        job["models"] = {}
        for variant, path in paths.items():
            payload = validate_st_checkpoint(path)
            if (payload["feature_builder"].get("type") != "team-exposure-features-v3"
                    or partitions(payload) != (job["fit"], job["dev"])
                    or payload["training_metadata"]["metric"] != job["payload"]["training_metadata"]["metric"]
                    or payload["training_metadata"]["training_source_sha256"] !=
                       job["payload"]["training_metadata"]["training_source_sha256"]):
                raise ValueError(f"{name}/{variant}: fit/dev split or measured labels differ")
            if variant == "original_mpnet" and payload["estimator"]["config"] != job["payload"]["estimator"]["config"]:
                raise ValueError(f"{name}: original MPNet head configuration differs")
            if variant != "original_minilm" and not payload["training_metadata"].get("encoder_identity"):
                raise ValueError(f"{name}/{variant}: missing frozen encoder identity")
            reuse.pin(identity, path)
            reuse.separate_output(args.output_dir, path.parent)
            encoder = payload["feature_builder"]["encoder"]
            if Path(encoder["model"]).is_dir():
                reuse.separate_output(args.output_dir, encoder["model"])
            job["models"][variant] = {"checkpoint": path, "payload": payload}
    manifest = {"schema": SCHEMA, "datasets": args.datasets, "protocol": PROTOCOL,
        "source_sha256": identity, "tuning_root": str(args.tuning_root),
        "qa_root": str(args.qa_root), "mpnet_reference_root": str(args.mpnet_reference_root),
        "rule_reference_root": str(args.rule_reference_root), "selected_head_trial": selection["selected_trial"],
        "models": {n: {v: {"checkpoint": str(m["checkpoint"]), "checkpoint_sha256": sha(m["checkpoint"]),
                           "encoder": m["payload"]["feature_builder"]["encoder"]}
                       for v, m in j["models"].items()} for n, j in jobs.items()},
        "partitions": {n: {"fit_ids": sorted(j["fit"]), "dev_ids": sorted(j["dev"]),
                            **j["partition_counts"]} for n, j in jobs.items()},
        "test_evaluations": len(jobs) * len(VARIANTS),
        "test_outcomes_read_during_calibration": False, "new_training": False,
        "new_collection": False, "original_results_overwritten": False}
    return jobs, manifest


def feature_cache(args, name, payload):
    key = digest(payload["feature_builder"]["encoder"])[:20]
    return args.output_dir / name / "features" / key / "semantic.sqlite"


def checked_gate(args, name, model):
    """Verify real frozen weights before consulting any copied feature cache."""
    from .csm_st_worker import dedup_identity
    from .runtime_gate import GMemoryExposureGate
    from .semantic_features import SentenceTransformerEmbedder

    spec = model["payload"]["feature_builder"]["encoder"]
    probe = SentenceTransformerEmbedder(spec["model"], revision=spec.get("revision"),
        chunk_tokens=spec["chunk_tokens"], expected_dimensions=spec.get("dimensions"), device="cpu")
    try:
        identity = dedup_identity(probe, {k: v for k, v in spec.items() if k != "dimensions"})
    finally:
        probe.close()
    if not identity.get("local_model_sha256") and not identity.get("resolved_commit"):
        raise ValueError("cannot identify frozen encoder; make its pinned model available offline")
    expected = model["payload"]["training_metadata"].get("encoder_identity")
    # A Hub encoder may have been saved with revision=resolved_commit, although
    # its historical identity records revision=None (before resolution).
    def same_weights(left, right):
        return ({k: v for k, v in left.items() if k != "revision"} ==
                {k: v for k, v in right.items() if k != "revision"}
                and all(r.get("revision") in (None, r.get("resolved_commit")) for r in (left, right))) or left == right

    if expected is not None and not same_weights(identity, expected):
        raise ValueError(f"{name}: frozen encoder weights differ from the checkpoint")
    cache = feature_cache(args, name, model["payload"])
    locked(cache.parent / "encoder_identity.json", identity)
    # MPNet's original tuning caches already contain dev vectors. Copy with a
    # read-only SQLite backup only after confirming their actual weight identity.
    source = args.tuning_root / name / "features" / f"chunk_{spec['chunk_tokens']}"
    if (not cache.exists() and (source / "semantic.sqlite").is_file()
            and (source / "encoder_identity.json").is_file()
            and same_weights(reuse.read(source / "encoder_identity.json"), identity)):
        closed_cache(source / "semantic.sqlite")
        reuse.copy_cache(source / "semantic.sqlite", cache)
    gate = GMemoryExposureGate.from_checkpoint(model["checkpoint"],
        memory_dir=None, semantic_cache_path=cache)
    return gate, identity


def dev_predictions(args, job, model, gate):
    from .st_data import load_st_examples

    builder, metadata = gate.feature_builder, []

    class RecordedBuilder:
        version = 3

        def build(self, event):
            kind = event.memory.memory_type
            rank = event.retrieval.rank
            allowed = gate.candidate_kind_ranks.get(kind, frozenset()) if gate.candidate_kind_ranks is not None else gate.candidate_ranks
            metadata.append({"memory_id": event.memory.memory_id, "memory_type": kind, "rank": rank,
                "order": next(i for i, c in enumerate(event.candidate_set) if c.memory_id == event.memory.memory_id),
                "memory_content": event.memory.content,
                "in_scope": kind in gate.candidate_kinds and (allowed is None or rank in allowed)})
            return builder.build(event)

    payload = model["payload"]
    examples = load_st_examples(job["sources"], metric=payload["training_metadata"]["metric"],
        task_ids={int(t) for t in job["dev"]}, hash_dimensions=payload["feature_builder"].get("hash_dimensions", 256),
        feature_version=3, memory_dir=job["memory"], feature_builder=RecordedBuilder(), event_sink=[], progress_label="ST dev only")
    if len(metadata) != len(examples) or {str(e.group_id) for e in examples} != job["dev"]:
        raise ValueError("original dev supervision is incomplete or changed")
    records = []
    for start in range(0, len(examples), 256):
        batch = examples[start:start + 256]
        # Runtime scores one event at a time. Match its float32 computation too,
        # since small batched-GEMM differences can otherwise flip near-zero ties.
        predictions = [gate.estimator.predict(e.features) for e in batch]
        if len(predictions) != len(batch):
            raise ValueError("missing dev predictions")
        for i, (example, pred) in enumerate(zip(batch, predictions), start):
            records.append({**metadata[i], "task_id": str(example.group_id), "event_id": example.event_id,
                "predicted_u": pred.utility, "uncertainty": pred.uncertainty, "calibrated": pred.calibrated,
                "q_use": example.q_use, "q_drop": example.q_drop})
    groups(records, job["dev"])
    return records


def original_policy(job):
    if job["name"] == "hotpotqa":
        return {"rule": "positive_unlimited", "kappa": 0., "delta": 0., "max_drops": None}
    old = job["deployment"]
    return {"rule": "calibrated", **old, "max_drops": None if old["max_drops"] == -1 else old["max_drops"]}


def offline_calibrate(args, jobs, manifest):
    policies = {}
    for name, job in jobs.items():
        policies[name] = {}
        reference_labels = None
        for variant, model in job["models"].items():
            root = args.output_dir / name / variant
            gate, identity = checked_gate(args, name, model)
            try:
                binding = {"manifest_sha256": digest(manifest), "checkpoint_sha256": sha(model["checkpoint"]),
                    "encoder_identity": identity}
                predictions_path = root / "dev_predictions.json"
                completion = root / "dev_predictions.complete.json"
                if completion.exists():
                    status = reuse.read(completion)
                    if status["binding"] != binding or status["sha256"] != sha(predictions_path):
                        raise ValueError("saved dev prediction artifact changed")
                    records = reuse.read(predictions_path)
                else:
                    print(f"[开发集诊断] {name}/{variant}；只推理，不训练", flush=True)
                    records = dev_predictions(args, job, model, gate)
                    save(predictions_path, records)
                    save(completion, {"binding": binding, "sha256": sha(predictions_path)})
                labels = {(r["task_id"], r["memory_id"]): [r[k] for k in ("memory_type", "rank", "order", "memory_content", "q_use", "q_drop")] for r in records}
                if reference_labels is not None and labels != reference_labels:
                    raise ValueError("encoders do not share the same measured dev candidates/labels")
                reference_labels = labels
                choice = calibrate(records, job["dev"])
                locked(root / "calibration.json", {**binding, **choice})
                locked(root / "policy.json", {**binding, "schema": SCHEMA,
                    "dataset": name, "variant": variant, "checkpoint": str(model["checkpoint"]),
                    "policy": choice["selected"]["policy"], "selection_scope": "original_dev_only"})
                save(root / "diagnostics.json", diagnose(records, job["dev"], original_policy(job), choice["selected"]["policy"]))
                policies[name][variant] = {"policy": choice["selected"]["policy"],
                    "policy_sha256": sha(root / "policy.json"), "dev_gain": choice["selected"]["mean_observed_gain"]}
            finally:
                gate.feature_builder.embedder.close()
    result = {"manifest_sha256": digest(manifest), "datasets": policies}
    locked(args.output_dir / "selection.json", result)
    dev_report(args)
    print("[完成] 开发集诊断与单条删除校准；测试评测尚未启动。", flush=True)
    return result


def dev_report(args):
    lines = ["Development diagnostics only; selected dev gains are not test results.", "",
        "| Dataset | Checkpoint | Rule | Dev tasks | Drop tasks | Improved | Harmed | Neutral drops | Measured gain (pp) | Single-drop oracle (pp) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in args.datasets:
        for variant in VARIANTS:
            diag = reuse.read(args.output_dir / name / variant / "diagnostics.json")
            for rule, row in diag["policies"].items():
                fmt = lambda v: "unmeasured" if v is None else f"{100*v:.3f}"
                lines.append("| " + " | ".join(map(str, [name, variant, rule, row["tasks"], row["drop_tasks"],
                    row["improved_tasks"], row["harmed_tasks"], row["neutral_drop_tasks"],
                    fmt(row["mean_observed_gain"]), fmt(row["best_observed_single_drop_mean_gain"])])) + " |")
    lines.extend(["", "Selected policies (KEEP ALL has Max drops=0):", "",
        "| Dataset | Checkpoint | Kappa | Delta | Max drops | Selected dev gain (pp) |",
        "| --- | --- | --- | --- | --- | --- |"])
    for name in args.datasets:
        for variant in VARIANTS:
            selected = reuse.read(args.output_dir / name / variant / "calibration.json")["selected"]
            policy = selected["policy"]
            lines.append("| " + " | ".join(map(str, [name, variant, policy["kappa"], policy["delta"],
                         policy["max_drops"], f"{100*selected['mean_observed_gain']:.3f}"])) + " |")
    (args.output_dir / "dev_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def closed_cache(path):
    if not Path(path).is_file() or any(Path(str(path) + s).exists() for s in ("-wal", "-journal")):
        raise ValueError(f"reference cache must exist and be closed: {path}")


def require_selection(args, manifest):
    if reuse.read(args.output_dir / "manifest.json") != manifest:
        raise ValueError("registered input/configuration changed; use a new output directory")
    selected = reuse.read(args.output_dir / "selection.json")
    if selected["manifest_sha256"] != digest(manifest):
        raise ValueError("decision selection manifest changed")
    for name, models in selected["datasets"].items():
        for variant, item in models.items():
            path = args.output_dir / name / variant / "policy.json"
            if sha(path) != item["policy_sha256"]:
                raise ValueError("dev-selected policy changed")
    return selected


def evaluate(args, jobs, manifest, selected):
    from .compare_gate_runs import compare
    from .interactive_progress import run_logged
    from .run_full_gate_benchmark import merge_caches

    identity, items = {}, {}
    # Held-out outcomes are deliberately read only in this explicit stage.
    for name in args.datasets:
        suite = SimpleNamespace(**{**vars(args), "st_deletion_rule": "dataset_default"})
        item = items[name] = reuse.prepare_qa(suite, name, identity)
        ref = args.rule_reference_root / name / "semantic_st/memory_gate.jsonl"
        if ref.exists():
            item["rule_reference"] = ref
        old_rows = tuning.same_rule_reference(item, "positive_unlimited" if name == "hotpotqa" else "calibrated")
        if old_rows is None:
            raise ValueError(f"{name}: missing completed MiniLM same-rule reference")
        item["minilm_rows"] = old_rows
        item["references"] = {
            "original_mpnet": args.mpnet_reference_root / name / "semantic_st",
            "tuned_mpnet": args.tuning_root / name / "semantic_st"}
        caches = {item["cache"], item["baseline"] / "llm_cache.sqlite"}
        if ref.exists():
            reuse.pin(identity, ref)
            if (ref.parent / "llm_cache.sqlite").exists():
                caches.add(ref.parent / "llm_cache.sqlite")
        for variant, path in item["references"].items():
            run = reuse.read(path / "run_manifest.json")["run"]
            if run["checkpoint_sha256"] != sha(jobs[name]["models"][variant]["checkpoint"]):
                raise ValueError(f"{name}/{variant}: reference checkpoint differs")
            for f in (path / "run_manifest.json", path / "memory_gate.jsonl"):
                reuse.pin(identity, f)
            caches.add(path / "llm_cache.sqlite")
        for path in caches:
            closed_cache(path)
            reuse.pin(identity, path)
        item["caches"] = sorted(caches)
    locked(args.output_dir / "evaluation_manifest.json", {"manifest_sha256": digest(manifest),
        "selection_sha256": sha(args.output_dir / "selection.json"), "source_sha256": identity})
    results = {}
    for name, item in items.items():
        minilm = args.output_dir / name / "original_minilm_reference.jsonl"
        if not minilm.exists():
            minilm.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in item["minilm_rows"]), encoding="utf-8")
        else:
            if reuse.rows(minilm) != item["minilm_rows"]:
                raise ValueError("copied MiniLM reference changed")
        references = {"original_minilm": minilm,
            **{v: p / "memory_gate.jsonl" for v, p in item["references"].items()}}
        seed_cache = args.output_dir / name / "seed_cache.sqlite"
        merge_caches(item["caches"], seed_cache)
        results[name] = {}
        for variant, model in jobs[name]["models"].items():
            root = args.output_dir / name / variant
            policy = selected["datasets"][name][variant]["policy"]
            gate, _ = checked_gate(args, name, model)
            gate.feature_builder.embedder.close()
            child, adapter = deepcopy(item["args"]), item["adapter"]
            child.checkpoint = model["checkpoint"]
            spec = model["payload"]["feature_builder"]["encoder"]
            child.semantic_model, child.semantic_revision = spec["model"], spec.get("revision")
            child.kappa, child.delta, child.max_drops = (policy[k] for k in ("kappa", "delta", "max_drops"))
            child.semantic_cache = feature_cache(args, name, model["payload"])
            child.memory_dir = reuse.private_memory(item["source_memory"], root / "frozen_memory/g-memory")
            output = root / "semantic_st"
            host = adapter.evaluation_command(child, mode="learned", output=output,
                                              resume=(output / "run_manifest.json").exists())
            host[host.index("--cache-seed-from") + 1] = str(seed_cache)
            command = [*host[:3], "causal_memory_control.st_single_drop_evaluate", "--dataset", name,
                "--policy", str(root / "policy.json"), "--reference-log", str(references[variant]), *host[4:]]
            # Audit the completed references against the original Always Use
            # control before allowing any new model/service call.
            audit = compare(item["baseline"] / "memory_gate.jsonl", references[variant],
                            bootstrap_samples=2000, seed=child.gate_evaluation_seed)
            if not audit["paired_control_valid"] or not audit["policy_valid"]:
                raise ValueError("original reference baseline/retrieval audit failed")
            if not adapter.evaluation_complete(child, output, mode="learned", command=host):
                run_logged(command, root / "evaluate.log", cwd=Path.cwd())
            if not adapter.evaluation_complete(child, output, mode="learned", command=host):
                raise ValueError("single-drop test evaluation is incomplete; resume the same command")
            target = output / "memory_gate.jsonl"
            baseline = compare(item["baseline"] / "memory_gate.jsonl", target,
                               bootstrap_samples=2000, seed=child.gate_evaluation_seed)
            paired = compare_encoders(references[variant], target, seed=child.gate_evaluation_seed)
            paired["reference_role"] = "same_checkpoint_original_rule"
            if not all(r["paired_control_valid"] and r["policy_valid"] for r in (baseline, paired)):
                raise ValueError("single-drop paired evaluation audit failed")
            if any(sum(d["action"].upper() == "DROP" for d in r["decisions"]) > policy["max_drops"] for r in reuse.rows(target)):
                raise ValueError("single-drop budget audit failed")
            save(output / "baseline_comparison.json", baseline)
            save(output / "original_rule_comparison.json", paired)
            results[name][variant] = {"n": baseline["paired_tasks"], "policy": policy,
                "baseline": baseline["baseline_mean_reward"], "original_rule": paired["baseline_mean_reward"],
                "single_drop": baseline["gate_mean_reward"], "delta_vs_original": paired["reward_delta"],
                "delta_vs_original_95ci": paired["reward_delta_95ci"], "drop_tasks": baseline["tasks_with_drop"]}
            if name == "hotpotqa":
                results[name][variant].update(baseline_em=baseline["baseline_answer_em"],
                    original_rule_em=paired["baseline_answer_em"], single_drop_em=baseline["gate_answer_em"])
            save(args.output_dir / "summary.json", {"schema": SCHEMA, "datasets": results, "selection_scope": "original_dev_only"})
            test_report(args, results)
    return results


def test_report(args, results):
    lines = ["Exploratory evaluation on previously examined test partitions. No test-based rule selection.", "",
        "| Dataset | Checkpoint | Metric | N | Always Use (%) | Original rule (%) | Dev-calibrated single drop (%) |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, models in results.items():
        for variant, row in models.items():
            for label, suffix in (("Accuracy", ""),) if name == "fever" else (("Answer F1", ""), ("Answer EM", "_em")):
                lines.append("| " + " | ".join([name, variant, label, str(row["n"]),
                    *(f"{100*row[k+suffix]:.2f}" for k in ("baseline", "original_rule", "single_drop"))]) + " |")
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    jobs, manifest = prepare(args)
    if args.stage == "plan":
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest
    if args.stage == "report":
        selected = require_selection(args, manifest)
        for filename in ("dev_diagnostics.md", "results.md"):
            path = args.output_dir / filename
            if path.exists():
                print(path.read_text())
        return selected
    with tuning.output_lock(args.output_dir), offline_models():
        locked(args.output_dir / "manifest.json", manifest)
        if args.stage == "calibrate":
            return offline_calibrate(args, jobs, manifest)
        selected = require_selection(args, manifest)
        return evaluate(args, jobs, manifest, selected)


if __name__ == "__main__":
    main()
