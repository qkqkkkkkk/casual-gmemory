"""Local--Global architecture with measured Team U targets, without CE scoring.

Reuse paired USE ALL / SINGLE DROP rewards on the ORIGINAL candidate lists.
Predict on that full list, then remove duplicate memories with the CE method's
frozen semantic deduplicator. This preserves the measured target's conditioning
set; it is not a claim to have Team U labels for deduplicated training lists.
Cross-task recombination is disabled. No Qwen weights are loaded here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import importlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

from tqdm import tqdm

from .compare_gate_runs import file_sha256
from .csm_st_data import digest, extract_lists, read_rows, save_json, split_ids, write_rows


SCHEMA = "local-global-team-u-v1"
LABEL = "mean_team_reward_use_minus_single_drop"


def load_team_u_lists(data):
    """Stream original branch rewards; pair by run hash, sample seed and repeat.

    The internal training-loop name ``ci`` is introduced only during fitting.
    Persisted labels are called ``team_u`` and retain both measured means.
    """
    rows = extract_lists(data["sources"], data["diagnostic_ids"],
                         forbidden_ids=[*data["support_ids"], *data["test_ids"]])
    indexed = {row["task_id"]: row for row in rows}
    metric = data["training_metric"]
    with tempfile.TemporaryDirectory(prefix="local-global-team-u-") as scratch:
        with closing(sqlite3.connect(str(Path(scratch) / "pairs.sqlite"))) as db:
            db.execute("CREATE TABLE contexts(task TEXT PRIMARY KEY, digest TEXT)")
            db.execute("CREATE TABLE rewards(task TEXT, candidate TEXT, rollout TEXT, condition TEXT, "
                       "reward REAL, PRIMARY KEY(task,candidate,rollout,condition))")
            for source in data["sources"]:
                path = Path(source)
                manifest = json.loads((path.parent / "run_manifest.json").read_text())
                design, run = manifest["design"], manifest["run"]
                if design["snapshot_design_hash"] != data["context"]["snapshot_design_hash"]:
                    raise ValueError("Team U branches belong to a different frozen snapshot")
                repeats, seed_base = design["repeats"], run["sample_seed_base"]
                expected = {str(v) for v in design["evaluation_ids"]}
                observed, cells = set(), Counter()
                with path.open(encoding="utf-8") as handle:
                    for line in tqdm(handle, desc="Read measured Team U", unit="branch"):
                        if not line.strip():
                            continue
                        branch = json.loads(line)
                        condition = branch["condition"]
                        if condition not in {"use_all", "global_drop"}:
                            continue
                        task, candidate = str(branch["task_id"]), branch["candidate"]
                        if task not in indexed or task not in expected:
                            raise ValueError(f"unexpected Team U task: {task}")
                        matches = [c for c in branch["candidate_set"]
                                   if c["candidate_id"] == candidate["candidate_id"]]
                        if len(matches) != 1 or matches[0] != candidate:
                            raise ValueError("Team U target differs from its original candidate set")
                        repeat, seed = branch["repeat_index"], branch["sample_seed"]
                        if (type(repeat) is not int or not 0 <= repeat < repeats
                                or seed != seed_base + repeat or branch["run_hash"] != manifest["run_hash"]):
                            raise ValueError("Team U branch repeat/seed/run identity changed")
                        context = digest({key: branch.get(key) for key in
                            ("task", "candidate_set", "retrieval_sizes", "exposure_count")})
                        previous = db.execute("SELECT digest FROM contexts WHERE task=?", (task,)).fetchone()
                        if previous is not None and previous[0] != context:
                            raise ValueError(f"Team U branch context differs for task {task}")
                        db.execute("INSERT OR IGNORE INTO contexts VALUES (?,?)", (task, context))
                        reward = float(branch["outcome"][metric])
                        if not math.isfinite(reward) or not 0 <= reward <= 1:
                            raise ValueError("Team U needs finite rewards in [0,1]")
                        rollout = json.dumps([branch["run_hash"], seed, repeat])
                        candidate_id = str(candidate["candidate_id"])
                        try:
                            db.execute("INSERT INTO rewards VALUES (?,?,?,?,?)",
                                       (task, candidate_id, rollout, condition, reward))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError("duplicate Team U counterfactual branch") from exc
                        observed.add(task)
                        cells[(task, candidate_id, condition)] += 1
                if observed != expected:
                    raise ValueError("Team U source does not cover its declared diagnostic tasks")
                for task in observed:
                    for candidate in indexed[task]["candidates"]:
                        if any(cells[(task, candidate["id"], condition)] != repeats
                               for condition in ("use_all", "global_drop")):
                            raise ValueError("Team U is missing declared candidate USE/DROP repeats")
                db.commit()
            for row in rows:
                row.pop("answers")  # Gold answers are not model inputs or Team U targets.
                row.update(team_u=[], q_use=[], q_drop=[], paired_samples_per_candidate=[], label_type=LABEL)
                for candidate in row["candidates"]:
                    values = {condition: dict(db.execute(
                        "SELECT rollout,reward FROM rewards WHERE task=? AND candidate=? AND condition=?",
                        (row["task_id"], candidate["id"], condition)))
                        for condition in ("use_all", "global_drop")}
                    use, drop = values["use_all"], values["global_drop"]
                    if not use or use.keys() != drop.keys():
                        raise ValueError("Team U USE/DROP branches are not paired")
                    q_use, q_drop = sum(use.values()) / len(use), sum(drop.values()) / len(drop)
                    row["q_use"].append(q_use)
                    row["q_drop"].append(q_drop)
                    row["team_u"].append(q_use - q_drop)
                    row["paired_samples_per_candidate"].append(len(use))
    return rows


def validate_labels(root, config):
    path = root / "labels_manifest.json"
    if not path.is_file():
        return False
    spec = json.loads(path.read_text())
    if spec.get("schema") != SCHEMA or spec.get("config_digest") != digest(config):
        raise ValueError("Team U labels belong to another experiment/configuration")
    for name, expected in spec["artifact_sha256"].items():
        target = root / name
        if root.resolve() not in target.resolve().parents or file_sha256(target) != expected:
            raise ValueError("Team U label artifacts changed")
    return True


def label_stage(config, dataset):
    from .csm_st_worker import dedup_identity
    from .semantic_features import SentenceTransformerEmbedder

    data, root = config["experiments"][dataset], Path(config["output_dir"]) / dataset
    rows = load_team_u_lists(data)
    fit_ids, dev_ids = split_ids(data["diagnostic_ids"], data["dev_fraction"], 71)
    # Freeze the same semantic model/threshold used by CE, without removing any
    # item from the list on which the already-measured Team U is conditioned.
    dedup = data["dedup"]
    encoder = SentenceTransformerEmbedder(dedup["model"], revision=dedup.get("revision"),
        chunk_tokens=dedup["chunk_tokens"], cache_path=root / "dedup.sqlite", device="cpu")
    try:
        identity = dedup_identity(encoder, dedup)
    finally:
        encoder.close()
    identity_path = root / "dedup_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Team U runtime dedup model changed; use the original model or a new output")
    save_json(identity_path, identity)
    write_rows(root / "team_u_labels.jsonl", rows)
    save_json(root / "split.json", {"fit_task_ids": sorted(fit_ids), "dev_task_ids": sorted(dev_ids),
        "final_test_ids": data["test_ids"], "split_seed": 71})
    distributions = {}
    for name, ids in (("fit", fit_ids), ("dev", dev_ids)):
        values = [v for row in rows if row["task_id"] in ids for v in row["team_u"]]
        distributions[name] = {"candidates": len(values), "negative": sum(v < -1e-8 for v in values),
            "near_zero": sum(abs(v) <= 1e-8 for v in values), "positive": sum(v > 1e-8 for v in values),
            "minimum": min(values), "maximum": max(values)}
    save_json(root / "labels_manifest.json", {"schema": SCHEMA, "config_digest": digest(config),
        "label_type": LABEL, "metric": data["training_metric"], "source_sha256": data["input_sha256"],
        "fit_task_ids": sorted(fit_ids), "dev_task_ids": sorted(dev_ids),
        "deduplication": True, "dedup_placement": "post_score", "dedup_identity": identity,
        "training_candidate_deduplication": False, "recombination": False,
        "new_team_rollouts": 0, "ce_forwards": 0,
        "team_u_distribution": distributions, "artifact_sha256": {name: file_sha256(root / name)
            for name in ("team_u_labels.jsonl", "split.json", "dedup_identity.json")}})


def deployment_dedup(provenance):
    """Resolved immutable dedup configuration; older checkpoints can disable it."""
    if not provenance.get("deduplication"):
        return {"enabled": False}
    spec = {**provenance["dedup_identity"], "enabled": True, "placement": "post_score"}
    if spec.get("resolved_commit"):
        spec["revision"] = spec["resolved_commit"]
    return spec


def train_stage(config, dataset):
    from .csm_st_model import CSMConfig
    from .csm_st_training import train

    data, root = config["experiments"][dataset], Path(config["output_dir"]) / dataset
    if not validate_labels(root, config):
        raise FileNotFoundError("extract measured Team U labels before training")
    provenance = json.loads((root / "labels_manifest.json").read_text())
    # ``ci`` is the existing generic list-regression loop's internal target name.
    # No CE scorer or generator is involved, and exported targets keep Team U names.
    rows = [{**row, "ci": row["team_u"]} for row in read_rows(root / "team_u_labels.jsonl")]
    metadata = {"training_contexts": [data["context"]], "candidate_kinds": data["candidate_kinds"],
        "candidate_ranks": data["candidate_ranks"], "candidate_kind_ranks": data["candidate_kind_ranks"],
        "training_source_sha256": data["input_sha256"], "estimator": {"type": "local_global_team_u"},
        "source_diagnostic_rewards_used": True, "config_digest": digest(config),
        "label_metric": data["training_metric"], "label_type": LABEL}
    payload = train(rows, root, model_config=CSMConfig(**config["settings"]["model_config"]),
        options=config["settings"]["training"], fit_ids=set(provenance["fit_task_ids"]),
        dev_ids=set(provenance["dev_task_ids"]), metadata=metadata,
        dedup_config=deployment_dedup(provenance), label_provenance=provenance)
    spec = payload.pop("csm_st")
    spec.update(schema=SCHEMA, label_type=LABEL, target_range=[-1, 1], output_activation="unbounded")
    payload["local_global_team_u"] = spec
    payload["training_metadata"]["policy_calibration"].update(
        selection_rule="keep_iff_team_u_gt_zero_and_not_duplicate",
        note="dev selects epoch only; full-list scoring then semantic dedup; unlimited drops")
    history = payload["training_diagnostics"]["loss_history"]
    for epoch in history:
        epoch["dev_team_u_sign_accuracy"] = epoch.pop("dev_ci_sign_accuracy")
    write_rows(root / "dev_predictions.jsonl", [{"id": row["id"], "team_u": row["ci"],
        "predicted_team_u": row["predicted_ci"]} for row in read_rows(root / "dev_predictions.jsonl")])
    save_json(root / "loss_history.json", history)
    save_json(root / "training_summary.json", {"schema": SCHEMA,
        "options": config["settings"]["training"], "diagnostics": payload["training_diagnostics"],
        "label_provenance": provenance, "training_metadata": payload["training_metadata"]})
    save_json(root / "checkpoint.json", payload)
    return payload


def checkpoint_spec(checkpoint):
    from .runtime_gate import CHECKPOINT_SCHEMA

    checkpoint = Path(checkpoint)
    payload = json.loads(checkpoint.read_text())
    spec = payload.get("local_global_team_u", {})
    dedup = spec.get("dedup", {})
    valid_dedup = (dedup == {"enabled": False} or
        dedup.get("enabled") is True and dedup.get("placement") == "post_score"
        and isinstance(dedup.get("model"), str) and bool(dedup["model"])
        and isinstance(dedup.get("threshold"), (int, float)) and 0 < dedup["threshold"] <= 1
        and isinstance(dedup.get("chunk_tokens"), int) and dedup["chunk_tokens"] >= 8)
    if (payload.get("schema") != CHECKPOINT_SCHEMA or spec.get("schema") != SCHEMA
            or spec.get("label_type") != LABEL or not valid_dedup):
        raise ValueError("expected a Local--Global Team U checkpoint; CE/old ST checkpoints are incompatible")
    artifact = checkpoint.parent / spec["artifact_dir"]
    if artifact.resolve().parent != checkpoint.parent.resolve() or "weights.pt" not in spec["artifact_sha256"]:
        raise ValueError("invalid Team U model artifact directory/weight identity")
    for name, expected in spec["artifact_sha256"].items():
        path = artifact / name
        if artifact.resolve() not in path.resolve().parents or file_sha256(path) != expected:
            raise ValueError("Team U model artifact changed")
    return payload, spec, artifact


def load_predictor(checkpoint, device):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .csm_st_model import CSMConfig, CSMPredictor, LocalGlobalCSM

    _, spec, artifact = checkpoint_spec(checkpoint)
    with torch.random.fork_rng(devices=[]):
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(artifact / "encoder", local_files_only=True),
                                        trust_remote_code=False)
        network = LocalGlobalCSM(CSMConfig(**spec["config"]), encoder)
        network.load_state_dict(torch.load(artifact / "weights.pt", map_location="cpu", weights_only=True), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(artifact / "tokenizer", local_files_only=True, trust_remote_code=False)
    return CSMPredictor(network, tokenizer, device)


def gate_class(device="cpu"):
    """Build the scoped list gate without importing Torch in labels/plan-only."""
    from .runtime_gate import ExposureDecision, GMemoryExposureGate
    from .types import ExposureAction, PotentialOutcomePrediction

    class LocalGlobalTeamUGate(GMemoryExposureGate):
        def __init__(self, *, predictor, deduplicator=None, **kwargs):
            kwargs.setdefault("max_drops", None)
            kwargs.setdefault("kappa", 0.0)
            kwargs.setdefault("delta", 0.0)
            if (kwargs.get("mode", "learned") != "learned" or kwargs["max_drops"] is not None
                    or kwargs["kappa"] != 0 or kwargs["delta"] != 0):
                raise ValueError("Local--Global Team U keeps U>0 with kappa=delta=0 and unlimited drops")
            super().__init__(estimator=SimpleNamespace(fitted=True), **kwargs)
            self.predictor, self._scores = predictor, {}
            self.deduplicator, self._duplicates = deduplicator, {}

        @classmethod
        def from_checkpoint(cls, checkpoint_path, **kwargs):
            from .csm_st_data import Deduplicator
            from .semantic_features import SentenceTransformerEmbedder

            payload, spec, _ = checkpoint_spec(checkpoint_path)
            kwargs.pop("semantic_cache_path", None)
            kwargs.pop("jev_cache_path", None)
            dedup, deduplicator = spec["dedup"], None
            if dedup["enabled"]:
                for name, expected in dedup.get("local_model_sha256", {}).items():
                    if file_sha256(Path(dedup["model"]) / name) != expected:
                        raise ValueError("local semantic dedup model changed since Team U training")
                encoder = SentenceTransformerEmbedder(dedup["model"], revision=dedup.get("revision"),
                    chunk_tokens=dedup["chunk_tokens"],
                    cache_path=Path(checkpoint_path).parent / "dedup_runtime.sqlite")
                deduplicator = Deduplicator(encoder, dedup["threshold"])
            try:
                return cls(predictor=load_predictor(checkpoint_path, device), deduplicator=deduplicator,
                           checkpoint_metadata=payload["training_metadata"], **kwargs)
            except BaseException:
                if deduplicator is not None:
                    deduplicator.encoder.close()
                raise

        def filter_retrieval(self, result, *, query_task, task_state="", **kwargs):
            retrieval = self.adapter.adapt(result)
            if retrieval.failed_count or any(c.memory_type not in self.candidate_kinds for c in retrieval.candidates):
                raise ValueError("Team U requires the complete original trajectory/insight list and failed_topk=0")
            candidates = [{"id": c.memory_id, "kind": c.memory_type, "text": c.content}
                          for c in retrieval.candidates]
            values = self.predictor.predict(query_task, task_state or query_task, candidates)
            if len(values) != len(candidates) or not all(math.isfinite(v) for v in values):
                raise ValueError("invalid Team U predictions")
            self._scores = {c["id"]: v for c, v in zip(candidates, values)}
            self._duplicates = {}
            if self.deduplicator is not None:
                _, duplicates = self.deduplicator.indices([c["text"] for c in candidates])
                self._duplicates = {candidates[i]["id"]: candidates[j]["id"] for i, j in duplicates.items()}
            return super().filter_retrieval(result, query_task=query_task, task_state=task_state, **kwargs)

        def _decide(self, event):
            value = self._scores[event.memory.memory_id]
            duplicate = self._duplicates.get(event.memory.memory_id)
            keep = duplicate is None and value > 0
            prediction = PotentialOutcomePrediction(0.0, 0.0, value,
                "positive" if value > 0 else "nonpositive", 0.0, "local_global_team_u", False,
                self.checkpoint_metadata.get("training_samples", 0))
            return ExposureDecision(event.event_id, event.memory.memory_id, event.memory.memory_type,
                ExposureAction.KEEP if keep else ExposureAction.DROP, value,
                "team_u_duplicate" if duplicate is not None else "team_u_positive" if keep else "team_u_nonpositive", prediction,
                policy_scores={"predicted_team_u": value, "duplicate_of": duplicate,
                    "selection_rule": ("keep_iff_team_u_gt_zero_and_not_duplicate" if self.deduplicator is not None
                                       else "keep_iff_team_u_gt_zero"),
                    "q_use_q_drop_are_placeholders": True, "deduplication": self.deduplicator is not None,
                    "dedup_placement": "post_score" if self.deduplicator is not None else None})

        def close(self):
            if self.deduplicator is not None and self.deduplicator.encoder is not None:
                self.deduplicator.encoder.close()
            super().close()

    return LocalGlobalTeamUGate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", choices=("fever", "hotpotqa"), required=True)
    parser.add_argument("--stage", choices=("labels", "train", "evaluate"), required=True)
    parser.add_argument("--device", default="cpu")
    args, remaining = parser.parse_known_args(argv)
    if args.stage == "evaluate":
        host = importlib.import_module(f"causal_memory_control.{args.dataset}_experiment")
        original = host.GMemoryExposureGate
        try:
            host.GMemoryExposureGate = gate_class(args.device)
            return host.main(remaining)
        finally:
            host.GMemoryExposureGate = original
    if args.config is None or remaining:
        parser.error("labels/train require --config and accept no host evaluation arguments")
    config = json.loads(args.config.read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("expected independent Local--Global Team U configuration")
    for path, expected in config["experiments"][args.dataset]["input_sha256"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError("original Team U source artifacts changed")
    return (label_stage if args.stage == "labels" else train_stage)(config, args.dataset)


if __name__ == "__main__":
    main()
