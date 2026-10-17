"""Import the exact dataset/split settings of a recorded full benchmark."""

from __future__ import annotations

import copy
from pathlib import Path

from . import run_full_gate_benchmark as full
from . import run_hotpotqa_gate_comparison as base
from .dataset_adapters import load_dataset_adapter


PATH_FIELDS = ("data", "memory_dir", "diagnostic_results", "checkpoint", "output_dir",
               "semantic_cache", "baseline_cache_seed_from")


def import_source_dataset(suite, name):
    manifest = base._load_json(suite.source_root / "benchmark_manifest.json")
    if manifest is None and (suite.source_root / "benchmark_manifest.json").exists():
        raise ValueError("unreadable source benchmark_manifest.json; do not silently replace its configuration")
    if manifest is None:
        child = copy.copy(suite)
        child.output_dir = suite.source_root
        adapter, args, _ = full.prepare_dataset(child, name)
    else:
        if manifest.get("schema") != full.SCHEMA or name not in manifest.get("experiments", {}):
            raise ValueError(f"source manifest does not register full {name}: {suite.source_root}")
        if suite.shared_argv:
            raise ValueError("source manifest found: endpoint/model/split are imported exactly; omit shared overrides")
        adapter = load_dataset_adapter(name)
        args = adapter.parse_args([])
        stored = manifest["experiments"][name]
        for key, value in stored.items():
            if key not in ("pipeline_schema", "experiment_plan", "source_md5", "api_key"):
                setattr(args, key, value)
        for key, value in manifest.get("semantic", {}).get(name, {}).items():
            setattr(args, key, value)
        for key in PATH_FIELDS:
            value = getattr(args, key, None)
            setattr(args, key, Path(value).resolve() if value is not None else None)
        if args.output_dir != suite.source_root / name:
            raise ValueError("source artifact paths moved; restore original full-run directory layout")
        adapter.resolve_plan(args)
        adapter.validate_args(args)
        current = adapter.configuration(args)
        for key in ("source_md5", "training_claims", "final_claims", "valid_source_questions", "evaluation_count"):
            if stored.get(key) != current.get(key):
                raise ValueError(f"source {name} data/split changed: {key}")
    if not args.use_all_data:
        raise ValueError("full-data extension requires an all-data source, not a 100-question pilot")
    return adapter, args
