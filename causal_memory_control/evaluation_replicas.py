"""Explicit, auditable endpoint substitution for an existing paired baseline.

Live Ollama metadata matching is not proof of historical model identity or
bitwise-equivalent generation. The original baseline and actual gate endpoint
remain unchanged in their logs; only comparison's endpoint check is relaxed.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit


SCHEMA = "verified-paired-evaluation-replicas-v1"
LIMITATION = (
    "Live installed model digest, template/settings and Ollama version match. "
    "Legacy baseline has no recorded historical weight digest; live probes cannot "
    "prove historical identity, server context/environment equivalence or bitwise "
    "generation equivalence across GPUs. No-DROP outcomes must still match exactly."
)
IDENTITY_FIELDS = ("model", "digest", "ollama_version", "model_settings_sha256")


def validate_endpoint(value):
    endpoint = value.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path != "/v1"):
        raise ValueError("evaluation endpoint must be a credential-free HTTP(S) /v1 base URL")
    return endpoint


def build_manifest(baseline, *, source_endpoint, target_endpoint, model):
    from .compare_gate_runs import file_sha256
    from .parallel_collection import verify_replicas

    baseline = Path(baseline).resolve()
    probes = verify_replicas([source_endpoint, target_endpoint], model)
    return {"schema": SCHEMA, "source_endpoint": source_endpoint,
        "target_endpoint": target_endpoint, "model": model, "replicas": probes,
        "baseline_files": {str(baseline / name): file_sha256(baseline / name)
            for name in ("memory_gate.jsonl", "run_manifest.json", "llm_cache.sqlite")},
        "limitation": LIMITATION}


def validate_manifest(path, baseline_path, *, source_endpoint=None, target_endpoint=None, model=None):
    """Validate saved evidence before allowing exactly one endpoint pair."""
    from .compare_gate_runs import file_sha256

    path, baseline_path = Path(path).resolve(), Path(baseline_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("limitation") != LIMITATION:
        raise ValueError("invalid evaluation replica manifest")
    for key, expected in (("source_endpoint", source_endpoint), ("target_endpoint", target_endpoint), ("model", model)):
        if expected is not None and payload.get(key) != expected:
            raise ValueError(f"evaluation replica {key} changed")
    source, target = payload.get("source_endpoint"), payload.get("target_endpoint")
    if not isinstance(source, str) or not isinstance(target, str) or source == target:
        raise ValueError("evaluation replica manifest requires distinct source/target endpoints")
    validate_endpoint(source)
    validate_endpoint(target)
    probes = payload.get("replicas", [])
    if (len(probes) != 2 or [p.get("endpoint") for p in probes] != [source, target]
            or not isinstance(payload.get("model"), str) or not payload["model"]
            or any(not all(isinstance(p.get(key), str) and p[key] for key in IDENTITY_FIELDS) for p in probes)
            or any(p["model"] != payload["model"] for p in probes)
            or any(probes[0][key] != probes[1][key] for key in IDENTITY_FIELDS)):
        raise ValueError("evaluation replicas have inconsistent model identities")
    files = payload.get("baseline_files", {})
    expected_files = {str(baseline_path.parent / name) for name in (
        "memory_gate.jsonl", "run_manifest.json", "llm_cache.sqlite")}
    if baseline_path.name != "memory_gate.jsonl" or set(files) != expected_files:
        raise ValueError("evaluation replica manifest belongs to a different baseline")
    for name, digest in files.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"baseline artifact changed after replica verification: {name}")
    baseline_manifest = json.loads((baseline_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    if any(baseline_manifest.get("design", {}).get(key) != value
            for key, value in (("endpoint", source), ("model", payload["model"]))):
        raise ValueError("replica source does not match the actual baseline design")
    return payload


def paired_metadata(baseline, gate, proof):
    """Normalize only the verified endpoints; do not erase actual log metadata."""
    baseline, gate = dict(baseline), dict(gate)
    if (baseline.get("endpoint") != proof["source_endpoint"]
            or gate.get("endpoint") != proof["target_endpoint"]
            or baseline.get("model") != proof["model"] or gate.get("model") != proof["model"]):
        raise ValueError("paired endpoints/model disagree with evaluation replica evidence")
    baseline.pop("endpoint")
    gate.pop("endpoint")
    return baseline, gate
