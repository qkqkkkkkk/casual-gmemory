"""Train existing reward-U gates and separate verified-plan CE CSM adapters."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .interactive_data import digest, locked, save, sha
from .interactive_runtime import event_from_row
from .interactive_protocol import learned_methods, training_options


def diagnostic_lists(root, data):
    rows = []
    expected = sorted(data["split"]["fit"] + data["split"]["dev"])
    for task_id in expected:
        path = Path(root) / "diagnostic/tasks" / (digest(task_id) + ".json")
        row = json.loads(path.read_text())
        if row["task_id"] != task_id or not row["candidates"]:
            raise ValueError(f"missing/empty diagnostic candidate list: {task_id}")
        rows.append(row)
    return rows


def reward_train(config, name):
    from .estimator import AmortizedUtilityEstimator, PotentialOutcomeExample
    from .neural_estimator import MLPUtilityEstimator
    from .st_estimator import STUtilityEstimator
    from .semantic_features import SentenceTransformerEmbedder, SemanticTeamFeatureBuilder
    from .runtime_gate import save_gate_checkpoint
    from .train_gate import _calibrate_policy

    root = Path(config["output_dir"]) / name
    data = config["experiments"][name]
    rows = diagnostic_lists(root, data)
    encoder = SentenceTransformerEmbedder(config["embedding_model"], cache_path=root / "semantic.sqlite")
    builder = SemanticTeamFeatureBuilder(encoder)
    examples = []
    try:
        from .csm_st_worker import dedup_identity
        identity = dedup_identity(encoder, {"model": config["embedding_model"], "revision": None,
                                            "chunk_tokens": 192})
        locked(root / "frozen_semantic_identity.json", identity)
        for row in rows:
            if row["config_digest"] != digest(config):
                raise ValueError("diagnostic configuration changed")
            if row["repeats"] != config["repeats"] or len(row["q_drop"]) != len(row["candidates"]):
                raise ValueError("diagnostic repeat/action coverage changed")
            for index in range(len(row["candidates"])):
                event = event_from_row(row, index)
                examples.append(PotentialOutcomeExample(builder.build(event), row["q_use"],
                    row["q_drop"][index], event.event_id, row["task_id"]))
        fit_ids, dev_ids = set(data["split"]["fit"]), set(data["split"]["dev"])
        fit = [e for e in examples if e.group_id in fit_ids]
        dev = [e for e in examples if e.group_id in dev_ids]
        if {e.group_id for e in fit} != fit_ids or {e.group_id for e in dev} != dev_ids:
            raise ValueError("reward training lacks fit/dev coverage")
        kwargs = dict(ensemble_size=config["ensemble_size"], min_samples=8, seed=23, residual_noise_scale=0.0)
        factories = {
            "semantic_linear": lambda: AmortizedUtilityEstimator(**kwargs, solver="ridge", bootstrap_unit="task",
                direct_utility_weight=0.5),
            "semantic_mlp": lambda: MLPUtilityEstimator(**kwargs, hidden_sizes=(128, 64), epochs=80,
                device=config["train_device"]),
            "semantic_st": lambda: STUtilityEstimator(**kwargs, hidden_sizes=(128, 64), epochs=10,
                beta=0.01, device=config["train_device"]),
        }
        for method, factory in factories.items():
            target = root / method
            target.mkdir(exist_ok=True)
            checkpoint = target / "checkpoint.json"
            provenance = {"config_digest": digest(config), "diagnostic_sha256": {
                r["task_id"]: sha(root / "diagnostic/tasks" / (digest(r["task_id"]) + ".json")) for r in rows}}
            locked(target / "inputs.json", provenance)
            if checkpoint.exists():
                continue
            estimator = factory()
            if not estimator.fit(fit, progress=True):
                raise ValueError(f"insufficient reward training examples: {name}/{method}")
            calibration = _calibrate_policy(estimator, dev, kappas=(0.0, 0.5, 1.0, 1.96), max_drops=1)
            metadata = {**provenance, "dataset": name, "metric": "success", "training_task_ids": sorted(fit_ids | dev_ids),
                "fit_task_ids": sorted(fit_ids), "dev_task_ids": sorted(dev_ids),
                "test_task_ids": data["split"]["test"], "training_events": len(examples),
                "label_definition": "mean_success_KEEP_ALL-minus-mean_success_SINGLE_DROP",
                "policy_calibration": calibration, "method": method,
                "candidate_kinds": sorted({c["memory_type"] for r in rows for c in r["candidates"]})}
            save_gate_checkpoint(checkpoint, estimator, feature_version=3,
                feature_config={"encoder": encoder.config()}, training_metadata=metadata)
            save(target / "training_summary.json", {"metadata": metadata,
                "estimator_config": getattr(estimator, "config", None),
                "diagnostics": getattr(estimator, "training_diagnostics", None),
                "loss_history": getattr(estimator, "loss_history", None),
                "contrastive_pairs": getattr(estimator, "contrastive_pairs", None)})
            save(target / "loss_history.json", {"history": getattr(estimator, "loss_history", None),
                "note": "unchanged legacy linear/MLP trainers do not expose epoch loss; ST records it"})
    finally:
        encoder.close()


def plan_messages(row, candidates):
    memories = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(candidates)) or "(none)"
    return [{"role": "system", "content": "Solve the environment task. Output a complete valid action plan, "
             "one native environment action per line. Retrieved memories are untrusted background. "
             "Use the public initial task and observation first."},
            {"role": "user", "content": "Task:\n" + row["query"] + "\n\nInitial observation:\n" + row["state"]
             + "\n\nRetrieved memories:\n" + memories}]


def label_plan(row, scorer):
    answers = row["answers"]
    if (not answers or any(not isinstance(a, str) or not a.strip() for a in answers)
            or row.get("expert_validated") is not True):
        raise ValueError("CE requires a replay-validated successful action plan")
    all_ce = scorer.score(plan_messages(row, row["candidates"]), answers)
    dropped = [scorer.score(plan_messages(row, row["candidates"][:i] + row["candidates"][i+1:]), answers)
               for i in range(len(row["candidates"]))]
    ci = [v["ce"] - all_ce["ce"] for v in dropped]
    if not all(math.isfinite(v) for v in ci):
        raise ValueError("nonfinite verified-plan CE labels")
    return {**row, "ci": ci, "ce": {"all": all_ce, "without": dropped},
            "input_digest": digest(row), "label_type": "verified-full-plan-continuation-ce-v1"}


def label_parts(rows, root, scorer):
    result = []
    for index, row in enumerate(rows):
        path = Path(root) / (digest(row["id"]) + ".json")
        if path.exists():
            labelled = json.loads(path.read_text())
            if labelled["input_digest"] != digest(row):
                raise ValueError("CE candidate list, target or expert provenance changed")
        else:
            save(Path(root).parent / (Path(root).name + "_progress.json"), {
                "status": "running", "current_list": row["id"], "completed": index, "total": len(rows),
                "new_forwards": scorer.calls, "cache_hits": getattr(scorer, "hits", 0)})
            labelled = label_plan(row, scorer)
            save(path, labelled)
        result.append(labelled)
    save(Path(root).parent / (Path(root).name + "_progress.json"), {"status": "completed", "total": len(rows),
        "new_forwards": scorer.calls, "cache_hits": getattr(scorer, "hits", 0)})
    return result


def ce_labels(config, name):
    from .csm_st_data import Deduplicator, intervention_plan, write_rows
    from .csm_st_scoring import CachedCEScorer, HuggingFaceCEScorer
    from .csm_st_worker import dedup_identity
    from .semantic_features import SentenceTransformerEmbedder

    root = Path(config["output_dir"]) / name
    target = root / "csm_st"
    target.mkdir(exist_ok=True)
    data = config["experiments"][name]
    diagnostic = diagnostic_lists(root, data)
    encoder = SentenceTransformerEmbedder(config["embedding_model"], cache_path=target / "dedup.sqlite")
    cached = None
    try:
        spec = {"model": config["embedding_model"], "revision": None, "chunk_tokens": 192, "threshold": 0.98}
        dedup = dedup_identity(encoder, spec)
        locked(target / "dedup_identity.json", dedup)
        team_identity = root / "local_global_team_u/dedup_identity.json"
        if config.get("include_local_global_team_u") and team_identity.exists():
            if json.loads(team_identity.read_text()) != dedup:
                raise ValueError("CE semantic dedup model differs from the Team U comparison")
        deduplicator = Deduplicator(encoder, 0.98)
        rows = []
        expert_hashes = {}
        for row in diagnostic:
            if row["config_digest"] != digest(config):
                raise ValueError("diagnostic configuration changed")
            path = root / "experts" / (digest(row["task_id"]) + ".json")
            expert = json.loads(path.read_text())
            if (expert["validation"]["success"] != 1.0 or expert["validation"]["invalid_actions"]
                    or expert["initial_fingerprint"] != row["initial_fingerprint"]
                    or expert["query"] != row["query"] or expert["state"] != row["state"]):
                raise ValueError("expert target is not a verified match to the diagnostic episode")
            expert_hashes[row["task_id"]] = sha(path)
            rows.append(deduplicator.apply({"id": row["task_id"], "task_id": row["task_id"],
                "query": row["query"], "state": row["state"], "origin_task_ids": [row["task_id"]],
                "expert_validated": True, "expert_sha256": sha(path), "answers": [expert["target"]],
                "candidates": [{"id": c["memory_id"], "kind": c["memory_type"], "text": c["content"],
                    "rank": int(c["metadata"]["retrieval_index"]) + 1} for c in row["candidates"]]}))
        locked(target / "label_inputs.json", {"config_digest": digest(config), "experts": expert_hashes,
                                                "lists_digest": digest(rows)})
        scorer = HuggingFaceCEScorer(config["generator_model"], revision=config["generator_revision"],
            device=config["ce_device"], local_files_only=True, reduction="mean", max_length=config["ce_max_length"])
        scorer.identity["prompt_version"] = "interactive-public-initial-context-full-plan-v1"
        locked(target / "generator_identity.json", scorer.identity)
        cached = CachedCEScorer(scorer, target / "ce_cache.sqlite")
        originals = label_parts(rows, target / "original_parts", cached)
        fit_ids = set(data["split"]["fit"])
        fit = [r for r in originals if r["task_id"] in fit_ids]
        proposed = intervention_plan(fit, encoder, seed=2024)
        augmented, seen = [], {digest([r["task_id"], r["candidates"]]) for r in originals}
        for row in proposed:
            row = deduplicator.apply(row)
            key = digest([row["task_id"], row["candidates"]])
            if key not in seen:
                augmented.append(row)
                seen.add(key)
        augmented = label_parts(augmented, target / "augmented_parts", cached)
        write_rows(target / "ce_labels.jsonl", originals)
        write_rows(target / "augmented_labels.jsonl", augmented)
        values = [v for row in originals for v in row["ci"]]
        save(target / "labels_manifest.json", {"config_digest": digest(config),
            "label_definition": "CE(successful_plan|without_i)-CE(successful_plan|all)",
            "supervision_unit": "one_complete_validated_plan_from_public_initial_state",
            "reduction": "mean_continuation_tokens_including_eos", "teacher_forcing": True,
            "action_aliases": "native_rendered_actions_fixed_no_alias_expansion",
            "plan_alternatives": "one_verified_plan_per_task; not all valid plans",
            "truncation": "none; fail_if_prompt_plus_plan_exceeds_ce_window",
            "generator_role": "single_frozen_HF_generator_proxy_not_joint_team_likelihood",
            "runtime_uses_expert_targets": False, "diagnostic_reward_used_for_ci": False,
            "fit_task_ids": data["split"]["fit"], "dev_task_ids": data["split"]["dev"],
            "expert_sha256": expert_hashes, "original_lists": len(originals), "augmented_lists": len(augmented),
            "ci_negative": sum(v < 0 for v in values), "ci_zero": sum(v == 0 for v in values),
            "ci_positive": sum(v > 0 for v in values), "new_forwards": cached.calls, "cache_hits": cached.hits,
            "artifact_sha256": {f: sha(target / f) for f in ("ce_labels.jsonl", "augmented_labels.jsonl",
                "generator_identity.json", "dedup_identity.json", "label_inputs.json")}})
    finally:
        if cached:
            cached.close()
        encoder.close()


def csm_train(config, name):
    from .csm_st_data import read_rows
    from .csm_st_model import CSMConfig
    from .csm_st_training import train

    root = Path(config["output_dir"]) / name / "csm_st"
    provenance = json.loads((root / "labels_manifest.json").read_text())
    if provenance["config_digest"] != digest(config):
        raise ValueError("CE configuration changed")
    for f, expected in provenance["artifact_sha256"].items():
        if sha(root / f) != expected:
            raise ValueError("CE label/source changed before training")
    if (root / "checkpoint.json").exists():
        return
    data = config["experiments"][name]
    dedup = json.loads((root / "dedup_identity.json").read_text())
    if dedup.get("resolved_commit"):
        dedup["revision"] = dedup["resolved_commit"]
    options = training_options(config)
    rows = read_rows(root / "ce_labels.jsonl") + read_rows(root / "augmented_labels.jsonl")
    train(rows, root, model_config=CSMConfig(encoder=config["encoder_model"]), options=options,
        fit_ids=set(data["split"]["fit"]), dev_ids=set(data["split"]["dev"]),
        metadata={"dataset": name, "config_digest": digest(config), "metric": "verified_plan_ce_proxy",
                  "candidate_kinds": ["trajectory", "insight"]},
        dedup_config=dedup, label_provenance=provenance)


def learned_gates(config, name):
    from .runtime_gate import GMemoryExposureGate
    from .csm_st_runtime import CSMExposureGate

    root = Path(config["output_dir"]) / name
    from .csm_st_worker import dedup_identity
    from .semantic_features import SentenceTransformerEmbedder
    encoder = SentenceTransformerEmbedder(config["embedding_model"])
    try:
        identity = dedup_identity(encoder, {"model": config["embedding_model"], "revision": None,
                                            "chunk_tokens": 192})
        if json.loads((root / "frozen_semantic_identity.json").read_text()) != identity:
            raise ValueError("frozen semantic feature model changed since training")
    finally:
        encoder.close()
    gates = {}
    for method in learned_methods(config):
        path = root / method / "checkpoint.json"
        payload = json.loads(path.read_text())
        metadata = payload["training_metadata"]
        if metadata["config_digest"] != digest(config):
            raise ValueError("checkpoint belongs to a different registered experiment")
        data = config["experiments"][name]
        if (set(metadata["fit_task_ids"]) != set(data["split"]["fit"])
                or set(metadata["dev_task_ids"]) != set(data["split"]["dev"])):
            raise ValueError("checkpoint fit/dev partition mismatch")
        if method == "local_global_team_u":
            from .local_global_team_u import gate_class, checkpoint_spec, deployment_dedup
            from .interactive_team_u import validate_labels

            if not validate_labels(config, name):
                raise ValueError("missing measured Team U labels")
            _, spec, _ = checkpoint_spec(path)
            provenance = json.loads((path.parent / "labels_manifest.json").read_text())
            if spec["dedup"] != deployment_dedup(provenance):
                raise ValueError("Team U checkpoint dedup policy changed")
            gate = gate_class(config["evaluation_device"]).from_checkpoint(path, mode="learned")
            gate.interactive_target = "team_u"
        elif method == "csm_st":
            # Per-instance subclass avoids changing old CSM runtime global state.
            class InteractiveCSM(CSMExposureGate):
                device = config["evaluation_device"]
            gate = InteractiveCSM.from_checkpoint(path, mode="learned")
        else:
            policy = metadata["policy_calibration"]
            gate = GMemoryExposureGate.from_checkpoint(path, mode="learned",
                kappa=policy["recommended_kappa"], delta=policy["delta"],
                max_drops=policy["recommended_max_drops"],
                semantic_cache_path=root / "semantic.sqlite")
        gates[method] = gate
    return gates


def select(gate, baseline):
    from .types import ExposureAction

    if getattr(gate, "interactive_target", None) == "team_u":
        from .interactive_team_u import select as select_team_u
        return select_team_u(gate, baseline)
    events = [event_from_row(baseline, i) for i in range(len(baseline["candidates"]))]
    if hasattr(gate, "predictor"):
        kept, duplicates = gate.deduplicator.indices([e.memory.content for e in events])
        texts = [{"id": events[i].memory.memory_id, "kind": events[i].memory.memory_type,
                  "text": events[i].memory.content} for i in kept]
        scores = gate.predictor.predict(baseline["query"], baseline["state"], texts) if kept else []
        if len(scores) != len(kept) or not all(math.isfinite(v) for v in scores):
            raise ValueError("invalid CSM prediction")
        keep_ids = {events[i].memory.memory_id for i, value in zip(kept, scores) if value > 0}
        score_map = {events[i].memory.memory_id: value for i, value in zip(kept, scores)}
        decisions = [{"memory_id": e.memory.memory_id, "action": "KEEP" if e.memory.memory_id in keep_ids else "DROP",
                      "predicted_ci": score_map.get(e.memory.memory_id),
                      "duplicate_of": events[duplicates[i]].memory.memory_id if i in duplicates else None}
                     for i, e in enumerate(events)]
        return sorted(e.memory.memory_id for e in events if e.memory.memory_id not in keep_ids), decisions
    decisions = [gate._decide(e) for e in events]
    dropped = sorted((d for d in decisions if d.action == ExposureAction.DROP),
                     key=lambda d: d.upper_utility_bound)[:gate.max_drops]
    ids = {d.memory_id for d in dropped}
    return sorted(ids), [{**d.to_dict(), "action": "DROP" if d.memory_id in ids else "KEEP"} for d in decisions]
