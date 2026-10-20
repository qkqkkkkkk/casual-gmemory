"""Isolated CE/fit stages, releasing GPU allocations between processes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .compare_gate_runs import file_sha256
from .csm_st_data import (Deduplicator, digest, extract_lists, intervention_plan,
                          read_rows, save_json, split_ids, write_rows)


def artifact_identity(paths):
    return {str(p): file_sha256(Path(p)) for p in paths}


def dedup_identity(encoder, spec):
    identity = dict(spec)
    if hasattr(encoder, "_load"):
        actual = encoder._load()
        module = actual[0]
        config = getattr(getattr(module, "auto_model", None), "config", None)
        identity["resolved_commit"] = getattr(config, "_commit_hash", None)
    model_path = Path(spec["model"])
    if model_path.is_dir():
        identity["local_model_sha256"] = {str(p.relative_to(model_path)): file_sha256(p)
            for p in sorted(model_path.rglob("*")) if p.is_file() and p.suffix in (
                ".json", ".safetensors", ".bin", ".model", ".txt")
            and ".cache" not in p.parts and ".git" not in p.parts}
    return identity


def labels(config, name):
    from .csm_st_scoring import CachedCEScorer, HuggingFaceCEScorer, label_lists
    from .semantic_features import SentenceTransformerEmbedder

    data, settings = config["experiments"][name], config["settings"]
    root = Path(config["output_dir"]) / name
    root.mkdir(parents=True, exist_ok=True)
    lists = extract_lists(data["sources"], data["diagnostic_ids"],
        forbidden_ids=[*data["support_ids"], *data["test_ids"]])
    original_candidates = sum(len(r["candidates"]) for r in lists)
    fit_ids, dev_ids = split_ids(data["diagnostic_ids"], data["dev_fraction"], 71)
    dedup = data["dedup"]
    encoder = SentenceTransformerEmbedder(dedup["model"], revision=dedup.get("revision"),
        chunk_tokens=dedup["chunk_tokens"], cache_path=root / "dedup.sqlite", device="cpu")
    try:
        frozen_identity = dedup_identity(encoder, dedup)
        frozen_path = root / "dedup_identity.json"
        if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen_identity:
            raise ValueError("frozen semantic dedup model changed; refusing mixed labels")
        save_json(frozen_path, frozen_identity)
        deduplicator = Deduplicator(encoder, dedup["threshold"])
        lists = [deduplicator.apply(r) for r in lists]
        write_rows(root / "lists.jsonl", lists)
        save_json(root / "split.json", {"fit_task_ids": sorted(fit_ids), "dev_task_ids": sorted(dev_ids),
            "final_test_ids": data["test_ids"], "split_seed": 71})
        scorer = HuggingFaceCEScorer(settings["generator_model"], revision=settings["generator_revision"],
            device=settings["ce_device"], dtype=settings["ce_dtype"],
            max_length=settings["ce_max_length"], reduction=settings["ce_reduction"],
            local_files_only=settings["local_files_only"])
        identity_path = root / "generator_identity.json"
        identity = scorer.identity
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            scorer.close()
            raise ValueError("CE generator identity changed; do not mix labels from different weights/tokenizers")
        save_json(identity_path, identity)
        cached = CachedCEScorer(scorer, root / "ce_cache.sqlite")
        try:
            originals = label_lists(lists, root / "ce_labels.jsonl", name, cached, root / "ce_progress.json")
            fit = [r for r in originals if r["task_id"] in fit_ids]
            options = settings["training"]
            augmented = intervention_plan(fit, encoder, seed=options["seed"],
                alpha=options["rarity_alpha"], hard_threshold=options["hard_threshold"],
                gamma=settings["intervention_gamma"], query_max_cosine=settings["query_max_cosine"],
                per_hard=settings["interventions_per_hard"])
            proposed = len(augmented)
            seen = {digest([r["task_id"], sorted((c["kind"], c["text"]) for c in r["candidates"])])
                    for r in fit}
            distinct = []
            for row in augmented:
                row = deduplicator.apply(row)
                key = digest([row["task_id"], sorted((c["kind"], c["text"]) for c in row["candidates"])])
                if key not in seen:
                    distinct.append(row)
                    seen.add(key)
            augmented = distinct
            write_rows(root / "augmentation_plan.jsonl", augmented)
            augmented = label_lists(augmented, root / "augmented_labels.jsonl", name, cached,
                                    root / "augmentation_progress.json")
            summary = {"schema": "csm-st-ce-labels-v1", "config_digest": digest(config),
                "input_sha256": data["input_sha256"], "generator_identity_sha256": file_sha256(identity_path),
                "dedup_identity": frozen_identity,
                "generator_role": identity["generator_role"], "proxy_acknowledged": True,
                "source_model": data["context"]["model"], "original_lists": len(originals),
                "augmented_lists": len(augmented), "proposed_augmented_lists": proposed,
                "original_candidate_count": original_candidates,
                "deduplicated_candidate_count": sum(len(r["ci"]) for r in originals),
                "new_forwards_this_invocation": cached.calls,
                "cache_hits_this_invocation": cached.hits, "fit_task_ids": sorted(fit_ids),
                "dev_task_ids": sorted(dev_ids), "final_test_boundary_unchanged": True,
                "ce_reduction": settings["ce_reduction"], "no_silent_generator_truncation": True,
                "note": "Full CSM-ST components adapted to GMemory; single-generator CE proxy, not team-reward CI or original FlashRAG reproduction."}
            import numpy as np

            summary["ci_distribution"] = {}
            for partition, group in (("fit", fit), ("dev", [r for r in originals if r["task_id"] in dev_ids]),
                                     ("augmented_fit", augmented)):
                values = [v for row in group for v in row["ci"]]
                summary["ci_distribution"][partition] = {"candidates": len(values),
                    "negative": sum(v < -1e-8 for v in values), "near_zero": sum(abs(v) <= 1e-8 for v in values),
                    "positive": sum(v > 1e-8 for v in values),
                    "quantiles_0_25_50_75_100": np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else []}
            summary["artifact_sha256"] = artifact_identity(root / filename for filename in (
                "lists.jsonl", "ce_labels.jsonl", "augmented_labels.jsonl", "augmentation_plan.jsonl", "split.json",
                "dedup_identity.json"))
            save_json(root / "labels_manifest.json", summary)
        finally:
            cached.close()
    finally:
        encoder.close()


def train(config, name):
    from .csm_st_model import CSMConfig
    from .csm_st_training import train as fit

    data = config["experiments"][name]
    root = Path(config["output_dir"]) / name
    provenance = json.loads((root / "labels_manifest.json").read_text())
    if provenance["config_digest"] != digest(config):
        raise ValueError("CE labels configuration changed")
    for path, expected in provenance["artifact_sha256"].items():
        if file_sha256(Path(path)) != expected:
            raise ValueError("CE label artifact changed")
    rows = read_rows(root / "ce_labels.jsonl") + read_rows(root / "augmented_labels.jsonl")
    fit_ids, dev_ids = set(provenance["fit_task_ids"]), set(provenance["dev_task_ids"])
    metadata = {"training_contexts": [data["context"]], "candidate_kinds": data["candidate_kinds"],
        "candidate_ranks": data["candidate_ranks"], "candidate_kind_ranks": data["candidate_kind_ranks"],
        "training_source_sha256": data["input_sha256"], "estimator": {"type": "csm_st_local_global_ce"},
        "source_diagnostic_rewards_used": False, "config_digest": digest(config),
        "label_metric": "teacher_forced_gold_ce_difference", "generator_role": provenance["generator_role"]}
    dedup_config = dict(data["dedup"])
    if provenance["dedup_identity"].get("resolved_commit"):
        dedup_config["revision"] = provenance["dedup_identity"]["resolved_commit"]
    dedup_config["local_model_sha256"] = provenance["dedup_identity"].get("local_model_sha256", {})
    return fit(rows, root, model_config=CSMConfig(**config["settings"]["model_config"]),
        options=config["settings"]["training"], fit_ids=fit_ids, dev_ids=dev_ids,
        metadata=metadata, dedup_config=dedup_config, label_provenance=provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("fever", "hotpotqa"), required=True)
    parser.add_argument("--stage", choices=("labels", "train"), required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    return (labels if args.stage == "labels" else train)(config, args.dataset)


if __name__ == "__main__":
    main()
