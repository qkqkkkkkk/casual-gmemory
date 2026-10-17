"""Execution-only optimizations shared by FEVER and HotpotQA collectors.

Logical branch rows, request seeds, prompts and collection identities are kept
unchanged. A USE ALL outcome is shared only within one task and sample seed.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import time


class BranchIndex:
    """Index once on resume, then update in O(1) per persisted branch."""

    def __init__(self, existing):
        self.by_spec = defaultdict(set)
        self.by_task = defaultdict(list)
        for key, row in existing.items():
            self.add(key, row)

    def add(self, key, row):
        candidate = row["candidate"]
        self.by_spec[(key[0], str(candidate.get("kind")), int(candidate.get("index", -1)))].add((key[2], key[3]))
        self.by_task[key[0]].append(row)

    def observed(self, task_id, kind, index):
        return self.by_spec.get((task_id, kind, index), set())


class SharedUseAll:
    """A task-local memo with strict context checks for pre-optimization rows."""

    def __init__(self, rows, *, run_hash, task, candidate_set, retrieval_sizes,
                 exposure_count, seed_base, repeats):
        self.samples = {}
        for row in rows:
            if row["condition"] != "use_all":
                continue
            repeat = int(row["repeat_index"])
            expected = {
                "run_hash": run_hash, "candidate_set": candidate_set,
                "retrieval_sizes": retrieval_sizes, "exposure_count": exposure_count,
                "sample_seed": seed_base + repeat,
            }
            if (not 0 <= repeat < repeats
                    or any(row.get(k) != v for k, v in expected.items())
                    or any(row.get("task", {}).get(k) != task[k]
                           for k in ("task_main", "task_description"))):
                raise ValueError("cannot share resumed USE ALL: task, retrieval or sample seed changed")
            self.remember(row)

    def remember(self, row):
        key = (int(row["repeat_index"]), int(row["sample_seed"]))
        if key in self.samples and self.samples[key]["outcome"] != row["outcome"]:
            raise ValueError("conflicting USE ALL outcomes for the same task/seed; inspect existing branches")
        self.samples.setdefault(key, row)

    def get(self, repeat_index, sample_seed):
        row = self.samples.get((repeat_index, sample_seed))
        if row is None:
            return None
        return copy.deepcopy(row["outcome"]), {
            "mode": "shared_use_all",
            "source_event_id": row["event_id"],
            "source_candidate_id": row["candidate"]["candidate_id"],
            "repeat_index": repeat_index, "sample_seed": sample_seed,
        }


class RetrievalSession:
    """One read-only native GMemory/client per collector process, lazy opened.

    Retrieval never receives the branch-local task history. Only the native
    per-query insight cache is reset; native ranking and seeded requests remain
    untouched. Explicit close/finally avoids leaking SQLite/HTTP connections.
    """

    def __init__(self, args, *, client_factory, memory_factory, config_factory, embedding_factory):
        self.args = args
        self.client_factory, self.memory_factory = client_factory, memory_factory
        self.config_factory, self.embedding_factory = config_factory, embedding_factory
        self.client = self.memory = None
        self.records = None
        self.stats = {"persistent_memory_loads": 0, "memory_initialization_seconds": 0.0,
                      "native_retrieval_seconds": 0.0}

    def __enter__(self):
        return self

    def retrieve(self, *, query_task, cache_path, memory_records):
        from .memory import FrozenRetrieval

        args = self.args
        if self.memory is None:
            started = time.perf_counter()
            self.client = self.client_factory(args.model, cache_path,
                experiment_seed=args.selection_seed, api_base=args.endpoint, api_key=args.api_key)
            self.memory = self.memory_factory(namespace=args.memory_dir.name,
                global_config=self.config_factory(args.memory_dir), llm_model=self.client,
                embedding_func=self.embedding_factory(args.embedding_model))
            self.records = int(memory_records)
            if self.memory.memory_size != self.records:
                raise RuntimeError("frozen GMemory size differs from snapshot manifest")
            self.stats["persistent_memory_loads"] += 1
            self.stats["memory_initialization_seconds"] += time.perf_counter() - started
        elif self.records != int(memory_records):
            raise RuntimeError("frozen snapshot record count changed during collection")
        self.memory.insights_cache = []
        self.memory.current_task_context = None
        before = self.client.calls, self.client.cache_hits
        started = time.perf_counter()
        result = FrozenRetrieval.from_result(self.memory.retrieve_memory(
            query_task=query_task, successful_topk=args.successful_topk,
            failed_topk=args.failed_topk, insight_topk=args.insights_topk,
            threshold=args.threshold))
        self.stats["native_retrieval_seconds"] += time.perf_counter() - started
        return result, self.client.calls - before[0], self.client.cache_hits - before[1]

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None and self.memory is not None and self.memory.memory_size != self.records:
                raise RuntimeError("read-only GMemory changed persistent memory size")
        finally:
            if self.client is not None:
                self.client.close()
        return False
