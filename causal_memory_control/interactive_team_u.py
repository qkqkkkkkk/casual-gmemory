"""PDDL/ALFWorld adapter for the existing QA Local–Global + Team U method.

Measured KEEP ALL minus SINGLE DROP labels stay on their original lists.
No expert targets, CE teacher or recombined-candidate labels are used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from .interactive_data import digest, locked, save, sha
from .interactive_protocol import training_options


def measured_lists(config, name):
    from .interactive_runtime import ActionStore
    from .interactive_training import diagnostic_lists

    root = Path(config["output_dir"]) / name
    identity = {"config_digest": digest(config), "snapshot_sha256": sha(root / "snapshot.json"), "phase": "diagnostic"}
    store_root = root / "diagnostic/actions"
    # Missing collection must fail without creating an empty source store.
    if json.loads((store_root / "identity.json").read_text()) != identity:
        raise ValueError("Team U diagnostic action-store identity changed")
    store = ActionStore(store_root, identity)
    rows, hashes = [], {}

    def register(path):
        hashes[str(path.relative_to(root))] = sha(path)

    register(root / "snapshot.json")
    register(store_root / "identity.json")
    from .interactive_progress import show_progress
    source = diagnostic_lists(root, config["experiments"][name])
    show_progress("Team U 标签核验", 0, len(source))
    for number, raw in enumerate(source):
        if raw["config_digest"] != digest(config) or raw["repeats"] != config["repeats"]:
            raise ValueError("Team U diagnostic configuration/repeats changed")
        candidates = raw["candidates"]
        ids = [c["memory_id"] for c in candidates]
        if len(set(ids)) != len(ids) or any(c["memory_type"] not in ("trajectory", "insight") for c in candidates):
            raise ValueError("Team U requires unique original trajectory/insight candidates")
        references = {(r["repeat"], r["candidate_id"]): r for r in raw["branches"]}
        expected = {(repeat, cid) for repeat in range(config["repeats"]) for cid in ids}
        if len(references) != len(raw["branches"]) or references.keys() != expected:
            raise ValueError("Team U requires exactly paired KEEP/SINGLE DROP repeats")
        use_samples, drop_samples = [], [[] for _ in ids]
        for repeat in range(config["repeats"]):
            seed = config["diagnostic_seed"] + repeat
            keep_key = store.key(raw["task_id"], seed, raw, [])
            keep_path = store_root / "parts" / (keep_key + ".json")
            initial = json.loads(keep_path.read_text())
            keep = store.get(raw["task_id"], seed, initial, [])
            if keep is None or any(keep[k] != raw[k] for k in
                    ("task_id", "query", "state", "task_config", "exposure_count", "candidates",
                     "initial_fingerprint", "retrieval_fingerprint")):
                raise ValueError("Team U KEEP context differs from its original diagnostic list")
            if keep["seed"] != seed:
                raise ValueError("Team U KEEP sample seed changed")
            register(keep_path)
            use_samples.append(keep["outcome"]["success"])
            for index, cid in enumerate(ids):
                ref = references[repeat, cid]
                drop_key = store.key(raw["task_id"], seed, keep, [cid])
                if ref["keep_key"] != keep_key or ref["drop_key"] != drop_key or ref["sample_seed"] != seed:
                    raise ValueError("Team U branch references are not paired")
                dropped = store.get(raw["task_id"], seed, keep, [cid])
                if dropped is None:
                    raise ValueError("Team U missing SINGLE DROP branch")
                if any(dropped[k] != keep[k] for k in ("query", "state", "task_config", "exposure_count")):
                    raise ValueError("Team U DROP public context differs from KEEP")
                register(store_root / "parts" / (drop_key + ".json"))
                drop_samples[index].append(dropped["outcome"]["success"])
        if any(type(v) not in (int, float) or v not in (0, 1) for v in use_samples + sum(drop_samples, [])):
            raise ValueError("Team U requires explicit binary Success outcomes")
        q_use, q_drop = statistics.mean(use_samples), [statistics.mean(v) for v in drop_samples]
        if (raw["use_samples"] != use_samples or raw["drop_samples"] != drop_samples
                or raw["q_use"] != q_use or raw["q_drop"] != q_drop):
            raise ValueError("Team U aggregate differs from measured branch outcomes")
        register(root / "diagnostic/tasks" / (digest(raw["task_id"]) + ".json"))
        rows.append({"id": raw["task_id"], "task_id": raw["task_id"], "origin_task_ids": [raw["task_id"]],
            "query": raw["query"], "state": raw["state"],
            "candidates": [{"id": c["memory_id"], "kind": c["memory_type"], "text": c["content"],
                "rank": int(c["metadata"]["retrieval_index"]) + 1} for c in candidates],
            "team_u": [q_use - v for v in q_drop], "q_use": [q_use] * len(ids), "q_drop": q_drop,
            "paired_samples_per_candidate": [config["repeats"]] * len(ids),
            "label_type": "mean_team_reward_use_minus_single_drop"})
        show_progress("Team U 标签核验", number+1, len(source), raw["task_id"])
    return rows, hashes


def validate_labels(config, name):
    from .local_global_team_u import validate_labels as validate

    root = Path(config["output_dir"]) / name / "local_global_team_u"
    if not validate(root, config):
        return False
    provenance = json.loads((root / "labels_manifest.json").read_text())
    for relative, expected in provenance["source_sha256"].items():
        path = root.parent / relative
        if root.parent.resolve() not in path.resolve().parents or sha(path) != expected:
            raise ValueError("Team U measured source changed after labeling")
    return True


def labels(config, name):
    from .csm_st_data import write_rows
    from .csm_st_worker import dedup_identity
    from .local_global_team_u import LABEL, SCHEMA
    from .semantic_features import SentenceTransformerEmbedder

    if validate_labels(config, name):
        return
    rows, hashes = measured_lists(config, name)
    root = Path(config["output_dir"]) / name / "local_global_team_u"
    encoder = SentenceTransformerEmbedder(config["embedding_model"], cache_path=root / "dedup.sqlite")
    try:
        identity = dedup_identity(encoder, {"model": config["embedding_model"], "revision": None,
                                          "chunk_tokens": 192, "threshold": 0.98})
    finally:
        encoder.close()
    ce_identity = root.parent / "csm_st/dedup_identity.json"
    if ce_identity.exists() and json.loads(ce_identity.read_text()) != identity:
        raise ValueError("Team U semantic dedup model differs from the CE comparison")
    locked(root / "dedup_identity.json", identity)
    split = config["experiments"][name]["split"]
    save(root / "split.json", {"fit_task_ids": split["fit"], "dev_task_ids": split["dev"],
                               "final_test_ids": split["test"]})
    write_rows(root / "team_u_labels.jsonl", rows)
    distributions = {}
    for part in ("fit", "dev"):
        values = [v for r in rows if r["task_id"] in set(split[part]) for v in r["team_u"]]
        distributions[part] = {"candidates": len(values), "positive": sum(v > 0 for v in values),
            "zero": sum(v == 0 for v in values), "negative": sum(v < 0 for v in values),
            "minimum": min(values), "maximum": max(values)}
    save(root / "labels_manifest.json", {"schema": SCHEMA, "config_digest": digest(config),
        "label_type": LABEL, "metric": "success_explicit_won", "source_sha256": hashes,
        "fit_task_ids": split["fit"], "dev_task_ids": split["dev"], "dedup_identity": identity,
        "deduplication": True, "dedup_placement": "post_score", "training_candidate_deduplication": False,
        "recombination": False, "new_team_rollouts": 0, "ce_forwards": 0, "expert_targets_used": False,
        "team_u_distribution": distributions, "artifact_sha256": {f: sha(root / f) for f in
            ("team_u_labels.jsonl", "split.json", "dedup_identity.json")}})


def train(config, name):
    from .csm_st_data import read_rows, write_rows
    from .csm_st_model import CSMConfig
    from .csm_st_training import train as fit
    from .local_global_team_u import LABEL, SCHEMA, checkpoint_spec, deployment_dedup

    if not validate_labels(config, name):
        raise FileNotFoundError("run labels_team_u before train_team_u")
    root = Path(config["output_dir"]) / name / "local_global_team_u"
    provenance = json.loads((root / "labels_manifest.json").read_text())
    checkpoint = root / "checkpoint.json"
    if checkpoint.exists():
        raw = json.loads(checkpoint.read_text())
        if "local_global_team_u" in raw:
            payload, spec, _ = checkpoint_spec(checkpoint)
            if (payload["training_metadata"]["config_digest"] != digest(config)
                    or spec["dedup"] != deployment_dedup(provenance)):
                raise ValueError("Team U checkpoint belongs to a different experiment/dedup policy")
            return
        if (raw.get("training_metadata", {}).get("config_digest") != digest(config)
                or raw.get("training_metadata", {}).get("estimator", {}).get("type") != "local_global_team_u"):
            raise ValueError("unexpected Team U intermediate checkpoint")
    rows = [{**r, "ci": r["team_u"]} for r in read_rows(root / "team_u_labels.jsonl")]
    options = training_options(config, team_u=True)
    payload = fit(rows, root, model_config=CSMConfig(encoder=config["encoder_model"]), options=options,
        fit_ids=set(provenance["fit_task_ids"]), dev_ids=set(provenance["dev_task_ids"]),
        metadata={"dataset": name, "config_digest": digest(config), "metric": "success",
            "label_type": LABEL, "candidate_kinds": ["trajectory", "insight"],
            "estimator": {"type": "local_global_team_u"}, "source_diagnostic_rewards_used": True,
            "training_source_sha256": provenance["source_sha256"]},
        dedup_config=deployment_dedup(provenance), label_provenance=provenance)
    spec = payload.pop("csm_st")
    spec.update(schema=SCHEMA, label_type=LABEL, target_range=[-1, 1], output_activation="unbounded")
    payload["local_global_team_u"] = spec
    payload["training_metadata"]["policy_calibration"].update(
        selection_rule="keep_iff_team_u_gt_zero_and_not_duplicate",
        note="dev selects epoch only; full-list scoring then semantic dedup; unlimited drops")
    for epoch in payload["training_diagnostics"]["loss_history"]:
        epoch["dev_team_u_sign_accuracy"] = epoch.pop("dev_ci_sign_accuracy")
    write_rows(root / "dev_predictions.jsonl", [{"id": r["id"], "team_u": r["ci"],
        "predicted_team_u": r["predicted_ci"]} for r in read_rows(root / "dev_predictions.jsonl")])
    save(root / "loss_history.json", payload["training_diagnostics"]["loss_history"])
    save(root / "training_summary.json", {"schema": SCHEMA, "options": options,
        "diagnostics": payload["training_diagnostics"], "label_provenance": provenance,
        "training_metadata": payload["training_metadata"]})
    save(checkpoint, payload)


def select(gate, baseline):
    """Use the QA gate's own decisions after scoring the complete original list."""
    from .interactive_runtime import event_from_row
    from .types import ExposureAction

    events = [event_from_row(baseline, i) for i in range(len(baseline["candidates"]))]
    texts = [{"id": e.memory.memory_id, "kind": e.memory.memory_type, "text": e.memory.content} for e in events]
    values = gate.predictor.predict(baseline["query"], baseline["state"], texts) if events else []
    if len(values) != len(events) or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid Team U predictions")
    _, duplicates = gate.deduplicator.indices([e.memory.content for e in events])
    gate._scores = {e.memory.memory_id: v for e, v in zip(events, values)}
    gate._duplicates = {events[i].memory.memory_id: events[j].memory.memory_id for i, j in duplicates.items()}
    decisions = [gate._decide(e) for e in events]
    return sorted(d.memory_id for d in decisions if d.action == ExposureAction.DROP), [d.to_dict() for d in decisions]
