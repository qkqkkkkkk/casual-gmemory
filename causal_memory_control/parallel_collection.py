"""Task-sharded counterfactual collection over independently served replicas.

Workers keep their own manifests, original run hashes, branch logs and SQLite
caches. Training consumes their validated union, never relabelled/fabricated
single-endpoint branch records. No server/GPU processes are started or stopped.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from tqdm import tqdm

from . import run_hotpotqa_gate_comparison as base


SCHEMA = "task-sharded-seed0-collection-v1"


def parse_endpoints(value):
    endpoints = [part.strip().rstrip("/") for part in value.split(",") if part.strip()]
    if len(endpoints) < 2 or len(endpoints) != len(set(endpoints)):
        raise ValueError("--collection-endpoints needs at least two distinct replica URLs")
    for endpoint in endpoints:
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path != "/v1"):
            raise ValueError("replica endpoints must be credential-free HTTP(S) /v1 base URLs")
    return endpoints


def shard_ranges(total, workers):
    if not 1 <= workers <= total:
        raise ValueError("each worker needs at least one task")
    offset, shards = 0, []
    for index in range(workers):
        size = total // workers + int(index < total % workers)
        shards.append((offset, size))
        offset += size
    return shards


def validate_union(shards, expected):
    observed = [str(t) for shard in shards for t in shard]
    if len(observed) != len(set(observed)) or set(observed) != {str(t) for t in expected}:
        raise ValueError("worker task shards overlap or do not exactly cover the registered diagnostic subset")


@contextmanager
def exclusive_run(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".seed0_parallel.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another parallel matrix controller is using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def probe_replica(endpoint, model):
    """Ollama-only preflight: require matching weights, template and runtime.

    This cannot prove GPU pinning or bitwise reproducibility. Hardware/server
    context settings still need to be matched by the experiment operator.
    """
    parsed = urlsplit(endpoint)
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    try:
        models = _request_json(origin + "/api/tags")["models"]
        match = next((item for item in models if model in (item.get("name"), item.get("model"))), None)
        if match is None or not isinstance(match.get("digest"), str) or not match["digest"]:
            raise ValueError(f"model {model!r} is not installed with a readable digest")
        show = _request_json(origin + "/api/show", {"model": model})
        version = _request_json(origin + "/api/version").get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("missing Ollama version")
        settings = {k: show.get(k) for k in ("parameters", "template", "system", "model_info", "details", "capabilities")}
        if not show.get("model_info") or not show.get("template"):
            raise ValueError("model metadata/template is incomplete")
        fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        return {"endpoint": endpoint, "model": model, "digest": match["digest"],
            "ollama_version": version, "model_settings_sha256": fingerprint}
    except Exception as exc:
        raise RuntimeError(f"replica preflight failed for {endpoint}: {exc}; check the service/model, do not silently mix replicas") from exc


def verify_replicas(endpoints, model):
    probes = [probe_replica(endpoint, model) for endpoint in endpoints]
    keys = ("model", "digest", "ollama_version", "model_settings_sha256")
    for probe in probes[1:]:
        mismatches = [k for k in keys if probe[k] != probes[0][k]]
        if mismatches:
            raise ValueError(f"Ollama replicas differ in {mismatches}: {probe['endpoint']}; use identical installed models and runtime versions")
    return probes


def validate_manifest(adapter, args):
    from . import run_seed0_gate_matrix as runner
    path = args.diagnostic_results / "parallel_collection_manifest.json"
    payload = base._load_json(path)
    workers = runner.collection_workers(args)
    expected = {"schema": SCHEMA,
        "snapshot_sha256": runner.file_sha256(args.memory_dir / "causal_snapshot_manifest.json"),
        "workers": [{"endpoint": child.endpoint, "offset": child.diagnostic_task_offset,
            "task_ids": runner.diagnostic_ids(child), "repeats": child.repeats,
            "sample_seed_base": child.sample_seed_base, "source": str(runner.source_path(child))}
            for child in workers]}
    if payload is None:
        raise ValueError("parallel collection is missing its verified model-replica manifest")
    base._require_fields("parallel replica provenance", payload, expected)
    probes = payload.get("replicas", [])
    if (len(probes) != len(workers) or [p.get("endpoint") for p in probes] != list(args.collection_endpoints)
            or any(p.get("model") != args.model or not all(p.get(k) for k in (
                "digest", "ollama_version", "model_settings_sha256")) for p in probes)):
        raise ValueError("invalid verified replica provenance")
    for probe in probes[1:]:
        if any(probe[k] != probes[0][k] for k in ("model", "digest", "ollama_version", "model_settings_sha256")):
            raise ValueError("recorded model replicas are inconsistent")


def _stop_workers(active):
    """Signal only worker processes launched here, never Ollama or other jobs."""
    for job in active:
        if job["process"].poll() is None:
            job["process"].send_signal(signal.SIGINT)
    deadline = time.monotonic() + 15
    for job in active:
        process = job["process"]
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()
    for job in active:
        process = job["process"]
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def collect_parallel(adapter, args, original):
    from . import run_seed0_gate_matrix as runner
    from . import run_full_gate_benchmark as full

    root = args.diagnostic_results
    state_path = root / "parallel_collection_progress.json"
    manifest_path = root / "parallel_collection_manifest.json"
    workers = runner.collection_workers(args)
    runner.diagnostic_ids(args)  # fail before launching if the frozen snapshot is missing
    validate_union([runner.diagnostic_ids(child) for child in workers], runner.diagnostic_ids(args))
    if runner.collection_complete(args):
        runner.validate_sources(adapter, args)
        previous = base._load_json(state_path) or {"schema": SCHEMA, "workers": {}}
        previous.update(status="completed", reused=True, error=None)
        for index, child in enumerate(workers):
            label = f"worker_{index:03d}"
            previous.setdefault("workers", {}).setdefault(label, {}).update(status="reused",
                source=str(runner.source_path(child)))
        full._save(state_path, previous)
        return
    state = {"schema": SCHEMA, "status": "running", "workers": {}, "error": None}
    active = []
    try:
        state["phase"] = "verifying_model_replicas"
        full._save(state_path, state)
        probes = verify_replicas(args.collection_endpoints, args.model)
        identity = {"schema": SCHEMA, "replicas": probes,
            "snapshot_sha256": runner.file_sha256(args.memory_dir / "causal_snapshot_manifest.json"),
            "workers": [{"endpoint": child.endpoint, "offset": child.diagnostic_task_offset,
                "task_ids": runner.diagnostic_ids(child), "repeats": child.repeats,
                "sample_seed_base": child.sample_seed_base, "source": str(runner.source_path(child))}
                for child in workers]}
        full.save_plan(manifest_path, identity)
        state["phase"] = "collecting"
        with ExitStack() as stack:
            for index, child in enumerate(workers):
                label = f"worker_{index:03d}"
                source = runner.source_path(child)
                if runner.collection_complete(child):
                    runner.validate_seed0(adapter, child)
                    state["workers"][label] = {"status": "reused", "source": str(source)}
                    continue
                command = runner.collection_command(adapter, child, original, verified_replica_cache=True)
                log_path = root / "logs" / (label + ".log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(log_path.open("ab"))
                handle.write((f"\n{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {label}\n"
                    f"command: {shlex.join(command)}\n").encode())
                handle.flush()
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                if args.api_key is not None:
                    environment["OPENAI_API_KEY"] = args.api_key
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
                active.append({"label": label, "process": process, "args": child, "log": log_path,
                    "command": command, "validated": False})
                state["workers"][label] = {"status": "running", "pid": process.pid,
                    "endpoint": child.endpoint, "log_path": str(log_path), "source": str(source)}
            full._save(state_path, state)
            per_task = (args.successful_topk + args.insights_topk) * args.repeats * 2
            total = runner.diagnostic_count(args) * per_task
            with tqdm(total=total, desc=f"{adapter.display_name} parallel seed0", unit="branch", mininterval=1) as progress:
                while True:
                    resolved = 0
                    for index, child in enumerate(workers):
                        payload = base._load_json(runner.source_path(child).parent / "collection_progress.json") or {}
                        resolved += int(payload.get("persisted_branches", 0))
                        label = f"worker_{index:03d}"
                        state["workers"][label]["progress"] = payload
                    progress.update(max(0, min(total, resolved) - progress.n))
                    for job in active:
                        code = job["process"].poll()
                        if code is None or job["validated"]:
                            continue
                        record = state["workers"][job["label"]]
                        record["exit_code"] = code
                        if code != 0:
                            record.update(status="failed", log_tail=base._tail(job["log"]))
                            raise base.StageFailure(job["label"], code, job["command"], job["log"], record["log_tail"])
                        runner.validate_seed0(adapter, job["args"])
                        job["validated"] = True
                        record["status"] = "completed"
                    state["resolved_branches"] = resolved
                    full._save(state_path, state)
                    if all(job["validated"] for job in active):
                        break
                    time.sleep(1)
                runner.validate_sources(adapter, args)
                progress.update(max(0, total - progress.n))
        state.update(status="completed", error=None)
        full._save(state_path, state)
    except BaseException as exc:
        _stop_workers(active)
        for job in active:
            if state["workers"][job["label"]]["status"] == "running":
                state["workers"][job["label"]].update(status="stopped", exit_code=job["process"].poll())
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error={"type": type(exc).__name__, "message": str(exc),
                "exit_code": getattr(exc, "returncode", None), "log_path": str(getattr(exc, "log_path", ""))})
        full._save(state_path, state)
        raise
