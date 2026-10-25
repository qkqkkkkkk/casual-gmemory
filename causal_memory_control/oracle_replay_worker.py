"""Scoped HotpotQA host wrapper with request traces and cache-only enforcement.

All writes belong to a new replay evaluation. Old clients/evaluators/ST are not
edited. Task bundles are persisted before the derived legacy-format task log,
so an interrupted append can be recovered without generating another answer.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

from causal_diagnostic.oracle_recipient.seeded_client import SeededCachedChat
from .csm_st_data import save_json
from .jsonl import read_jsonl
from .runtime_gate import GMemoryExposureGate


SCHEMA = "oracle-canonical-task-bundle-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


class CacheOnlyMiss(BaseException):
    """BaseException prevents the legacy HTTP retry loop from retrying a miss."""

    def __init__(self, key, table="responses"):
        self.key, self.table = key, table
        super().__init__(f"cache-only replay needs an uncached request: {table}/{key}")


class NoNetworkClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.reject))

    @staticmethod
    def reject(**payload):
        raise CacheOnlyMiss("unexpected-network-attempt")

    def close(self):
        pass


def append_json(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def recover_task_log(output):
    ledger = output / "task_request_ledger.jsonl"
    bundles = read_jsonl(ledger, repair_tail=True)
    rows = read_jsonl(output / "memory_gate.jsonl", repair_tail=True)
    indexed = {}
    for bundle in bundles:
        if bundle.get("schema") != SCHEMA:
            raise ValueError(f"wrong replay ledger schema: {ledger}")
        row = bundle["row"]
        task_id = str(row["task_id"])
        if task_id in indexed or bundle["row_sha256"] != digest(row):
            raise ValueError(f"duplicate/changed task bundle: {ledger}, {task_id}")
        indexed[task_id] = row
    seen = set()
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in seen or task_id not in indexed or digest(row) != digest(indexed[task_id]):
            raise ValueError(f"task log disagrees with replay ledger: {output}, {task_id}")
        seen.add(task_id)
    for task_id, row in indexed.items():
        if task_id not in seen:
            append_json(output / "memory_gate.jsonl", row)
    return indexed


def validate_ledger(output):
    """Read-only completion check; recovery itself belongs to the worker."""
    bundles = read_jsonl(output / "task_request_ledger.jsonl")
    rows = read_jsonl(output / "memory_gate.jsonl")
    indexed = {}
    for bundle in bundles:
        row = bundle["row"]
        task_id = str(row["task_id"])
        if (bundle.get("schema") != SCHEMA or task_id in indexed
                or bundle["row_sha256"] != digest(row)):
            raise ValueError(f"invalid replay ledger: {output}")
        indexed[task_id] = bundle
    if not rows or len(rows) != len(indexed):
        raise ValueError(f"incomplete replay ledger: {output}")
    seen = set()
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in seen or indexed.get(task_id, {}).get("row_sha256") != digest(row):
            raise ValueError(f"task log/ledger mismatch: {output}")
        seen.add(task_id)
    requests = [request for bundle in bundles for request in
                (*bundle.get("initialization_requests", []), *bundle["requests"])]
    if requests:
        path = (output / "llm_cache.sqlite").resolve()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            verified = {}
            for request in requests:
                table, key = request["table"], request["key"]
                if table not in {"responses", "binary_scores"} or request["source"] not in {"cache", "generated"}:
                    raise ValueError(f"invalid traced request: {output}")
                if (table, key) not in verified:
                    column = "response" if table == "responses" else "payload_json"
                    cached = db.execute(f"SELECT {column} FROM {table} WHERE key=?", (key,)).fetchone()
                    verified[(table, key)] = None if cached is None else hashlib.sha256(cached[0].encode("utf-8")).hexdigest()
                if verified[(table, key)] != request["response_sha256"]:
                    raise ValueError(f"traced response/cache mismatch: {output}, {table}/{key}")
    return indexed


class TaskRecorder:
    def __init__(self, output):
        self.output = output
        self.task_id = None
        self.requests = []
        self.initialization_requests = []

    def begin(self, task_id):
        self.task_id, self.requests = str(task_id), []

    def request(self, table, key, raw_value, cache_hit):
        record = {"table": table, "key": key,
                  "response_sha256": hashlib.sha256(raw_value.encode("utf-8")).hexdigest(),
                  "source": "cache" if cache_hit else "generated"}
        (self.initialization_requests if self.task_id is None else self.requests).append(record)

    def commit(self, row, log_path):
        if str(row["task_id"]) != self.task_id or Path(log_path).resolve() != self.output / "memory_gate.jsonl":
            raise ValueError("replay task context/log path changed")
        append_json(self.output / "task_request_ledger.jsonl", {
            "schema": SCHEMA, "row_sha256": digest(row), "row": row,
            "initialization_requests": self.initialization_requests,
            "requests": self.requests,
            "drop_memory_ids": sorted(d["memory_id"] for d in row["decisions"] if d["action"] == "drop"),
        })
        append_json(Path(log_path), row)
        self.task_id, self.requests = None, []


def tracking_chat(recorder, cache_only):
    class TrackedChat(SeededCachedChat):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if cache_only:
                self.client.close()
                self.client = NoNetworkClient()

        def tracked(self, table, key, callback):
            column = "response" if table == "responses" else "payload_json"
            before = self.db.execute(f"SELECT {column} FROM {table} WHERE key=?", (key,)).fetchone()
            if before is None and cache_only:
                raise CacheOnlyMiss(key, table)
            result = callback()
            after = self.db.execute(f"SELECT {column} FROM {table} WHERE key=?", (key,)).fetchone()
            if after is None or before is not None and before != after:
                raise ValueError("legacy request key changed or an existing canonical response was overwritten")
            stored_result = after[0] if table == "responses" else json.loads(after[0])
            if result != stored_result:
                raise ValueError("legacy returned answer does not match the traced cache key")
            recorder.request(table, key, str(after[0]), before is not None)
            return result

        def __call__(self, messages, temperature=0.1, max_tokens=512, stop_strs=None, num_comps=1):
            # Exact legacy payload. A post-call DB check catches key-version drift;
            # NoNetworkClient also guards against unexpected parent cache misses.
            rendered = self._messages(messages)
            payload = {"model": self.model_name, "messages": rendered,
                "temperature": 0.1 if temperature is None else float(temperature),
                "max_tokens": 512 if max_tokens is None else int(max_tokens),
                "stop": list(stop_strs) if stop_strs else None,
                "n": 1 if num_comps is None else int(num_comps), "seed": self._request_seed(rendered)}
            return self.tracked("responses", digest(payload), lambda: super(TrackedChat, self).__call__(
                messages, temperature, max_tokens, stop_strs, num_comps))

        def binary_label_probabilities(self, messages, *, top_logprobs=5):
            if top_logprobs < 2:
                raise ValueError("top_logprobs must be at least 2")
            rendered = self._messages(messages)
            payload = {"model": self.model_name, "messages": rendered, "temperature": 0.0,
                "max_tokens": 1, "n": 1, "seed": self._request_seed(rendered),
                "logprobs": True, "top_logprobs": int(top_logprobs)}
            key = digest({"schema": "binary-label-logprobs-v1", "payload": payload})
            return self.tracked("binary_scores", key, lambda: super(TrackedChat, self).binary_label_probabilities(
                messages, top_logprobs=top_logprobs))

    return TrackedChat


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-only", action="store_true")
    options, remaining = parser.parse_known_args(argv)
    from . import hotpotqa_experiment as host

    args = host.parse_args(remaining)
    if args.mode not in {"always_keep", "planned_drop"} or args.checkpoint is not None:
        parser.error("replay worker only accepts fixed actions, never trained policy evaluation")
    output = args.output_dir.resolve()
    recover_task_log(output)
    recorder = TaskRecorder(output)
    begin, end = GMemoryExposureGate.begin_task, GMemoryExposureGate.end_task

    def traced_begin(gate, task_id, task_config):
        begin(gate, task_id, task_config)
        recorder.begin(task_id)

    def traced_end(gate, *args, **kwargs):
        log_path = gate.log_path
        if log_path is None:
            raise ValueError("replay requires a task log")
        gate.log_path = None
        try:
            row = end(gate, *args, **kwargs)
        finally:
            gate.log_path = log_path
        recorder.commit(row, log_path)
        return row

    try:
        with patch.object(host, "SeededCachedChat", tracking_chat(recorder, options.cache_only)), \
                patch.object(GMemoryExposureGate, "begin_task", traced_begin), \
                patch.object(GMemoryExposureGate, "end_task", traced_end):
            result = host.main(remaining)
        validate_ledger(output)
        save_json(output / "replay_worker_status.json", {"status": "completed", "cache_only": options.cache_only})
        return result
    except CacheOnlyMiss as exc:
        save_json(output / "cache_only_miss.json", {"status": "blocked_on_uncached_request",
            "task_id": recorder.task_id, "table": exc.table, "key": exc.key,
            "network_calls_allowed": False, "message": str(exc)})
        print(str(exc), flush=True)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
