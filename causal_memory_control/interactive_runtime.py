"""Native MacNet execution and resumable, matched episode/action storage."""
from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time
import uuid

from .interactive_data import digest, save


ACTION_OUTPUT_INSTRUCTION = (
    "For this environment step, output exactly ONE action command on a single line. "
    "Start directly with the command in the task's action syntax. "
    "Do not output a heading, Markdown, code fences, an action plan, or an explanation."
)


class ActionFormatError(ValueError):
    """A node returned presentation text instead of a single action line."""


def validate_action_response(messages, answer):
    # Memory summarization/insight requests use the same client and may return
    # Markdown. Only node requests carry this exact system-level contract.
    for message in messages:
        role = message.role if hasattr(message, "role") else message["role"]
        content = message.content if hasattr(message, "content") else message["content"]
        if role == "system" and ACTION_OUTPUT_INSTRUCTION in content:
            text = answer.strip()
            if len(text.splitlines()) != 1 or text.startswith(("#", "```", "~~~")):
                raise ActionFormatError(
                    f"expected one action command, received heading/Markdown or multiple lines: {answer[:500]!r}")
            break


def configure_action_nodes(mas, task_instruction, temperature):
    # Native agents_team contains only solvers. The decision node submits the
    # actual environment action and needs the same domain and output rules.
    instruction = task_instruction + "\n" + ACTION_OUTPUT_INSTRUCTION
    for node in (*mas._agent_nodes.values(), mas._decision_node):
        node.reasoning_config.temperature = temperature
        node._agent.add_task_instruction(instruction)


def request_error_details(error):
    """Describe the original SDK/HTTP exception without logging request headers."""
    result, seen = [], set()
    while error is not None and id(error) not in seen and len(result) < 8:
        seen.add(id(error))
        item = {"type": type(error).__name__, "message": str(error)[:4000]}
        status = getattr(error, "status_code", None)
        if status is not None:
            item["http_status"] = status
        result.append(item)
        error = error.__cause__ or error.__context__
    return result


def clone_memory(source, destination):
    """Copy a consistent SQLite snapshot, including committed WAL contents."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*-wal", "*-shm"))
    for database in source.rglob("*.sqlite3"):
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as src:
            with sqlite3.connect(destination / database.relative_to(source)) as dest:
                src.backup(dest)


def candidate_rows(retrieval):
    return [{"memory_id": c.memory_id, "memory_type": c.memory_type,
             "content": c.content, "metadata": dict(c.metadata)} for c in retrieval.candidates]


def retrieval_check(adapter, result, reference, *, query_task, task_state, root, task_id, seed, drops):
    """Restore a measured candidate order only when every item still matches.

    A new native retrieval still runs in each episode. Only a permutation of
    identical candidates may be aligned to the original prompt; changes in
    membership, content, non-rank metadata or public context remain fatal.
    """
    if reference is None:
        return result
    from collections import Counter, defaultdict, deque

    retrieval = adapter.adapt(result)
    observed = candidate_rows(retrieval)
    expected = reference["candidates"]
    context_matches = query_task == reference["query"] and task_state == reference["state"]
    if context_matches and observed == expected:
        return result

    def item_key(row):
        return digest({**row, "metadata": {k: v for k, v in row["metadata"].items()
                                           if k != "retrieval_index"}})

    aligned = None
    if (context_matches and not result[1]
            and Counter(map(item_key, observed)) == Counter(map(item_key, expected))):
        objects = defaultdict(deque)
        for candidate, raw in zip(observed, [*result[0], *result[2]]):
            objects[item_key(candidate)].append(raw)
        successful, insights = [], []
        for candidate in expected:
            raw = objects[item_key(candidate)].popleft()
            (successful if candidate["memory_type"] == "trajectory" else insights).append(raw)
        proposed = (successful, list(result[1]), insights)
        if candidate_rows(adapter.adapt(proposed)) == expected:
            aligned = proposed

    classification = ("order_aligned_to_baseline" if aligned is not None else
                      "public_context_changed" if not context_matches else "candidate_data_changed")
    details = {"schema": "interactive-retrieval-check-v1", "task_id": task_id, "seed": seed,
        "drop_ids": sorted(drops), "classification": classification,
        "expected": {"query": reference["query"], "state": reference["state"], "candidates": expected},
        "observed": {"query": query_task, "state": task_state, "candidates": observed},
        "expected_ids": [c["memory_id"] for c in expected],
        "observed_ids": [c["memory_id"] for c in observed]}
    path = Path(root) / "retrieval_checks" / (digest([task_id, seed, sorted(drops)]) +
                                              "-" + uuid.uuid4().hex[:8] + ".json")
    save(path, details)
    if aligned is not None:
        print(f"[检索对齐] {task_id}：候选内容一致，恢复已测基线顺序；详情：{path}", flush=True)
        return aligned
    raise ValueError(f"original retrieval changed before exposure: {classification}; "
                     f"task={task_id}; details={path}")


def event_from_row(row, index):
    from .types import MemoryCandidate, RetrievalMetadata, TeamMemoryExposureEvent
    from .features import task_context_metadata

    candidates = tuple(MemoryCandidate(c["memory_id"], c["memory_type"], c["content"], c["metadata"])
                       for c in row["candidates"])
    candidate = candidates[index]
    return TeamMemoryExposureEvent(event_id=digest([row["task_id"], candidate.memory_id]),
        query=row["query"], task_state=row["state"], memory=candidate, candidate_set=candidates,
        exposure_agent_ids=tuple(f"host-recipient-{i}" for i in range(row["exposure_count"])),
        retrieval=RetrievalMetadata(candidate_id=candidate.memory_id, source="g-memory",
            rank=int(candidate.metadata["retrieval_index"]) + 1),
        task_metadata=task_context_metadata(row["task_config"]))


class Observer:
    def __init__(self):
        self.messages = []

    def log(self, message):
        self.messages.append(str(message))


class ActionStore:
    """A measured outcome is unique for (task, seed, candidate set, drop set).

    Task-sized atomic files preserve completed branches even if the process
    stops before an aggregate log is written. No lucky branch selection.
    """
    def __init__(self, root, identity):
        from .interactive_data import locked
        self.root = Path(root)
        locked(self.root / "identity.json", identity)
        self.identity = identity

    def key(self, task_id, seed, baseline, drops):
        return digest([task_id, seed, baseline["initial_fingerprint"],
                       baseline["retrieval_fingerprint"], sorted(drops)])

    def get(self, task_id, seed, baseline, drops):
        key = self.key(task_id, seed, baseline, drops)
        path = self.root / "parts" / (key + ".json")
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record.get("record_sha256") != digest({k: v for k, v in record.items() if k != "record_sha256"}):
            raise ValueError("measured action record changed")
        if record["action_key"] != key or record["input_identity"] != digest(self.identity):
            raise ValueError("action-store identity changed")
        # All lookups also audit the baseline, including a recovered no-DROP.
        audit(record, baseline, drops)
        return record

    def put(self, row):
        key = self.key(row["task_id"], row["seed"], row, row["drop_ids"])
        row = {**row, "action_key": key, "input_identity": digest(self.identity)}
        row["record_sha256"] = digest({k: v for k, v in row.items() if k != "record_sha256"})
        path = self.root / "parts" / (key + ".json")
        if path.exists() and json.loads(path.read_text()) != row:
            raise ValueError("conflicting outcomes for one matched action")
        save(path, row)
        return row


def audit(row, baseline, drops):
    if (row["task_id"] != baseline["task_id"] or row["seed"] != baseline["seed"]
            or row["initial_fingerprint"] != baseline["initial_fingerprint"]
            or row["retrieval_fingerprint"] != baseline["retrieval_fingerprint"]
            or row["candidates"] != baseline["candidates"]):
        raise ValueError("matched episode/retrieval/seed changed")
    if set(row["drop_ids"]) != set(drops):
        raise ValueError("requested drop set was not executed")
    if not drops and any(row[k] != baseline[k] for k in ("outcome", "trajectory", "query", "state")):
        raise ValueError("no-DROP parity failed")


class NativeRunner:
    def __init__(self, config, dataset_root, memory_dir, *, writable=False):
        from .interactive_envs import quiet_native_imports
        quiet_native_imports()
        from causal_diagnostic.oracle_recipient.seeded_client import SeededCachedChat
        from mas.memory.mas_memory.GMemory import GMemory
        from mas.utils import EmbeddingFunc

        self.config, self.root = config, Path(dataset_root)
        self._retrieval_references = {}
        self.memory_dir = Path(memory_dir)
        if not writable:
            # Chroma may update SQLite bookkeeping even on read-only logical
            # access. The registered physical snapshot is never opened by it.
            private = self.root / "runtime_memory" / uuid.uuid4().hex / "g-memory"
            clone_memory(self.memory_dir, private)
            self.memory_dir = private
        class StrictChat(SeededCachedChat):
            def __call__(client, *args, **kwargs):
                try:
                    answer = super().__call__(*args, **kwargs)
                    if not answer.strip():
                        raise ValueError("empty model answer; refusing unbounded native retry loop")
                    try:
                        validate_action_response(args[0] if args else kwargs.get("messages", ()), answer)
                    except ActionFormatError as exc:
                        # A model's formatting mistake is an ordinary output.
                        # Preserve it for native node collaboration/environment
                        # feedback; do not retry until a better answer appears.
                        print(json.dumps({"event": "interactive_action_format_warning",
                            "model": config["model"], "endpoint": config["endpoint"],
                            "message": str(exc), "handling": "native_action_processing"},
                            ensure_ascii=False), file=sys.stderr, flush=True)
                    return answer
                except Exception as exc:
                    client.branch_errors += 1
                    client.last_branch_error = exc
                    print(json.dumps({"event": "interactive_llm_request_failed",
                        "model": config["model"], "endpoint": config["endpoint"],
                        "error_chain": request_error_details(exc)}, ensure_ascii=False),
                        file=sys.stderr, flush=True)
                    raise

        self.client = StrictChat(config["model"], self.root / "responses.sqlite",
            experiment_seed=0, api_base=config["endpoint"])
        self.client.branch_errors = 0
        self.client.last_branch_error = None
        self.memory = GMemory(namespace="g-memory", global_config={"working_dir": str(self.memory_dir.parent),
            "persist_dir": str(self.memory_dir), "read_only": not writable, "hop": 1},
            llm_model=self.client, embedding_func=EmbeddingFunc(config["embedding_model"]))

    def run(self, task, seed, drops=(), *, expected=None):
        import numpy as np
        from mas.reasoning import ReasoningIO
        from tasks.mas_workflow.macnet.graph_mas import MacNet
        from tasks.prompts import get_dataset_system_prompt, get_task_few_shots
        from .interactive_envs import Episode
        from .runtime_gate import GMemoryExposureGate
        from .task_state import reset_task_execution_state
        from .types import ExposureAction
        from dataclasses import replace

        if expected is not None:
            self.remember_retrieval(expected)
        reference = self._retrieval_references.get(task["id"])
        random.seed(seed)
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.client.experiment_seed = seed
        calls, hits = self.client.calls, self.client.cache_hits
        errors = self.client.branch_errors
        self.client.last_branch_error = None
        self.memory.insights_cache = []
        self.memory.current_task_context = None

        class FixedGate(GMemoryExposureGate):
            def _decide(gate, event):
                decision = super()._decide(event)
                return replace(decision, action=ExposureAction.DROP if event.memory.memory_id in drops
                               else ExposureAction.KEEP, reason="registered_fixed_action")

            def filter_retrieval(gate, result, **kwargs):
                if getattr(gate, "retrieval_calls", 0):
                    raise ValueError("only one retrieval per episode is registered")
                gate.retrieval_calls = 1
                result = retrieval_check(gate.adapter, result, reference,
                    query_task=kwargs["query_task"], task_state=kwargs.get("task_state", ""),
                    root=self.root, task_id=task["id"], seed=seed, drops=drops)
                gate.raw_candidates = candidate_rows(gate.adapter.adapt(result))
                filtered = super().filter_retrieval(result, **kwargs)
                if reference and gate._retrieval_fingerprint != reference["retrieval_fingerprint"]:
                    raise ValueError(f"original retrieval fingerprint changed before exposure: task={task['id']}")
                if not set(drops) <= {r["memory_id"] for r in gate.raw_candidates}:
                    raise ValueError("DROP refers to an unavailable memory")
                return filtered

        episode = Episode(task, self.config["repo"], self.config["max_steps"])
        try:
            if reference and reference["initial_fingerprint"] != episode.initial_fingerprint:
                raise ValueError("initial environment differs from matched baseline")
            task_config = {**task["config"], "task_main": episode.task_main,
                "task_description": episode.task_description,
                "few_shots": get_task_few_shots(task["dataset"], task["config"], self.config["few_shots"])}
            gate = FixedGate(mode="always_keep", memory_dir=self.memory_dir, max_drops=None)
            gate.begin_task(task["id"], task_config)
            self.memory.set_retrieval_gate(gate)
            observer = Observer()
            mas = MacNet()
            mas.add_observer(observer)
            mas.build_system(ReasoningIO(llm_model=self.client), self.memory, episode,
                {"graph_type": "Chain", "node_num": 3, "use_critic": False,
                 "successful_topk": self.config["successful_topk"], "failed_topk": 0,
                 "insights_topk": self.config["insights_topk"], "threshold": 0.0, "use_projector": False})
            reset_task_execution_state(mas)
            configure_action_nodes(mas,
                get_dataset_system_prompt(task["dataset"], task["config"]), self.config["temperature"])
            gate.set_exposure_agent_ids([*mas.agents_team, mas._decision_node.id])
            started = time.monotonic()
            try:
                reward, success = mas.schedule(task_config)
            except Exception:
                if self.client.branch_errors != errors:
                    self._raise_request_failure(task["id"])
                raise
            if self.client.branch_errors != errors:
                self._raise_request_failure(task["id"])
            out = episode.result()
            if reward != out["success"] or bool(success) != bool(out["won"]):
                raise ValueError("native MAS did not use the registered success metric")
            # Agent graph keeps complete system/user/reply text for each step.
            context = self.memory.current_task_context
            row = {"schema": "interactive-episode-v1", "task_id": task["id"], "seed": seed,
                "task_config": {k: v for k, v in task_config.items() if k != "few_shots"},
                "query": episode.task_main, "state": episode.task_description,
                "initial_fingerprint": episode.initial_fingerprint, "reset_fingerprints": episode.resets,
                "retrieval_fingerprint": gate._retrieval_fingerprint, "candidates": gate.raw_candidates,
                "exposure_count": len(gate.exposure_agent_ids), "drop_ids": sorted(drops),
                "outcome": out, "reward": out["success"], "done": out["done"],
                "trajectory": episode.trace, "agent_transcript": observer.messages,
                "message_graph": type(context).to_dict(context),
                "llm_calls": self.client.calls - calls, "cache_hits": self.client.cache_hits - hits,
                "elapsed_seconds": time.monotonic() - started,
                "task_state_policy": "isolated_episode_rng_environment_memory_nodes_v1"}
            if expected:
                audit(row, expected, drops)
            self.remember_retrieval(row)
            return row
        finally:
            episode.close()
            self.memory.set_retrieval_gate(None)

    def close(self):
        self.client.close()

    def remember_retrieval(self, row):
        """Pin repeats to the first measured list, including after cache resume."""
        reference = {k: row[k] for k in ("query", "state", "candidates", "retrieval_fingerprint",
                                         "initial_fingerprint")}
        prior = self._retrieval_references.setdefault(row["task_id"], reference)
        if prior != reference:
            raise ValueError(f"cached baseline retrieval changed across repeats: task={row['task_id']}")

    def _raise_request_failure(self, task_id):
        error = self.client.last_branch_error
        details = json.dumps(request_error_details(error), ensure_ascii=False)
        reason = "LLM action format failed" if isinstance(error, ActionFormatError) else "LLM request failed"
        raise RuntimeError(f"{reason}; incomplete episode was not saved; task={task_id}; "
                           f"model={self.config['model']}; endpoint={self.config['endpoint']}; "
                           f"original_errors={details}") from error


def execute(runner, store, task, seed, drops, baseline):
    existing = store.get(task["id"], seed, baseline, drops)
    if existing is not None:
        return existing
    row = runner.run(task, seed, drops, expected=baseline)
    return store.put(row)


def baseline_for(runner, store, task, seed):
    # Index keeps the original baseline fingerprint, avoiding any model call on
    # resume. A missing action record after an interrupted index write is safe.
    path = store.root / "baselines" / (digest([task["id"], seed]) + ".json")
    if path.exists():
        baseline = json.loads(path.read_text())
        row = store.get(task["id"], seed, baseline, [])
        if row is None:
            raise ValueError("baseline index points to a missing measured outcome")
        if hasattr(runner, "remember_retrieval"):
            runner.remember_retrieval(row)
        return row
    row = store.put(runner.run(task, seed))
    save(path, row)
    if hasattr(runner, "remember_retrieval"):
        runner.remember_retrieval(row)
    return row


def model_identity(endpoint, model):
    """Read Ollama metadata; never called by --plan-only."""
    import urllib.request

    base = endpoint.rstrip("/").removesuffix("/v1")
    with urllib.request.urlopen(base + "/api/tags", timeout=20) as response:
        tags = json.load(response)
    match = next((r for r in tags["models"] if r.get("name") == model or r.get("model") == model), None)
    if match is None or not match.get("digest"):
        raise ValueError("cannot lock Ollama model digest; check the model name and endpoint")
    request = urllib.request.Request(base + "/api/show", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        info = json.load(response)
    return {"model": model, "digest": match["digest"],
            "show_sha256": digest({k: info.get(k) for k in ("template", "parameters", "model_info", "details", "system")})}
