"""Copy verified, committed v9 support to a new root after the format warning fix.

This one-time migration accepts exactly two known runtime versions. Completed
v9 episodes could not contain a format violation: its StrictChat aborted the
episode on every such violation. The new runtime keeps prompts, seeds, native
action processing and successful-response caching identical. Consequently the
committed prefix is reusable. No diagnostics, training or evaluation migrate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile

from .interactive_data import digest, files_identity, save, sha
from .run_interactive_gate_benchmark import dataset_jobs


RUNTIME = "causal_memory_control/interactive_runtime.py"
OLD_RUNTIME_SHA = "325b057a0e5b10bbc604fa75fb2f416ba4df3e636a1eedb984d0b440fcf9f9a4"
NEW_RUNTIME_SHA = "d984bfc1fa332500053214d42703ac3af8c10672b28fb195d129202d68321d0a"


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def experiment_lock(root):
    # Same sidecar lock as run_interactive_expanded_v8.sh. No live SQLite or
    # support transaction is read while an authorized launcher holds this lock.
    with Path(str(root) + ".lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"experiment is still running/locked: {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def committed_prefix(config, name):
    root = Path(config["output_dir"]) / name
    for entry in ("diagnostic", "experts", "csm_st", "local_global_team_u", "semantic_linear",
                  "semantic_mlp", "semantic_st", "evaluation"):
        if (root / entry).exists():
            raise ValueError(f"support-only migration refuses later-stage artifacts: {root / entry}")
    ids = config["experiments"][name]["split"]["support"]
    markers = sorted((root / "support/committed").glob("*.json"))
    if len(markers) > len(ids) or [p.name for p in markers] != [f"{i:04d}.json" for i in range(len(markers))]:
        raise ValueError(f"support commits must be a contiguous registered prefix: {root}")
    commits = []
    for index, marker in enumerate(markers):
        commit = read(marker)
        relative = Path(commit["directory"])
        if (relative.is_absolute() or len(relative.parts) != 4 or relative.parts[:2] != ("support", "attempts")
                or relative.parts[-1] != "committed" or ".." in relative.parts
                or not (root / relative).resolve().is_relative_to(root.resolve())):
            raise ValueError(f"invalid committed support path: {marker}")
        episode = root / relative.parent / "episode.json"
        row = read(episode)
        if (commit["task_id"] != ids[index] or row["task_id"] != ids[index]
                or row["seed"] != 10000 + index or commit["config_digest"] != digest(config)
                or sha(episode) != commit["episode_sha256"]
                or files_identity(root / relative / "g-memory") != commit["files_sha256"]):
            raise ValueError(f"support episode/commit/memory identity changed: {marker}")
        commits.append((marker, commit))
    snapshot = root / "snapshot.json"
    if snapshot.exists():
        state = read(snapshot)
        expected_memory = str(Path(commits[-1][1]["directory"]) / "g-memory") if commits else None
        if (len(commits) != len(ids) or state["config_digest"] != digest(config)
                or state["support_ids"] != ids or state["memory_dir"] != expected_memory
                or state["files_sha256"] != commits[-1][1]["files_sha256"]
                or state["support_episode_sha256"] != {p.name: sha(p) for p, _ in commits}):
            raise ValueError(f"frozen support snapshot identity changed: {snapshot}")
    return commits


def migrate(source, destination, *, apply=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if (source == destination or destination.is_relative_to(source) or source.is_relative_to(destination)
            or destination.exists()):
        raise ValueError("choose a new independent, nonexistent destination; source is always preserved")
    if not destination.parent.is_dir():
        raise ValueError("destination parent directory must exist")
    with experiment_lock(source), experiment_lock(destination):
        old = read(source / "benchmark_manifest.json")
        if (Path(old["output_dir"]).resolve() != source or not old.get("exclude_ce")
                or not old.get("include_local_global_team_u")
                or old["code_sha256"].get(RUNTIME) != OLD_RUNTIME_SHA):
            raise ValueError("only the known v9 CE-excluded action-format runtime can be migrated")
        repo = Path(old["repo"])
        if repo.resolve() != Path(__file__).resolve().parents[1]:
            raise ValueError("run migration from the source manifest's repository")
        for path, expected in old["code_sha256"].items():
            if sha(repo / path) != (NEW_RUNTIME_SHA if path == RUNTIME else expected):
                raise ValueError(f"unexpected implementation change; migration refused: {path}")
        for data in old["experiments"].values():
            for task in data["tasks"]:
                for path, expected in task["files"].items():
                    if sha(repo / path) != expected:
                        raise ValueError(f"registered task data changed: {path}")
        new = deepcopy(old)
        new["output_dir"] = str(destination)
        new["code_sha256"][RUNTIME] = NEW_RUNTIME_SHA
        jobs, counts = [], {}
        for name in old["experiments"]:
            for (label, previous), (_, updated) in zip(dataset_jobs(old, name), dataset_jobs(new, name)):
                manifest = Path(previous["output_dir"]) / "benchmark_manifest.json"
                if manifest.exists() and read(manifest) != previous:
                    raise ValueError(f"registered fold manifest changed: {manifest}")
                commits = committed_prefix(previous, name)
                if commits and not manifest.exists():
                    raise ValueError(f"support commits lack their registered manifest: {manifest}")
                jobs.append((name, label, previous, updated, commits))
                counts[label] = len(commits)
                print(f"[迁移检查] {label}: 可复用 {len(commits)} 个已提交 support", flush=True)
        audit = {"schema": "interactive-support-format-migration-v1", "source": str(source),
            "destination": str(destination), "source_config_digest": digest(old), "config_digest": digest(new),
            "old_runtime_sha256": OLD_RUNTIME_SHA, "new_runtime_sha256": NEW_RUNTIME_SHA,
            "committed_counts": counts, "apply": apply,
            "reason": "format violations become native outputs; completed old episodes had no format violations"}
        if not apply:
            print("[迁移检查] 通过；未复制结果，使用 --apply 执行。", flush=True)
            return audit
        temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".migrating-", dir=destination.parent))
        try:
            save(temporary / "benchmark_manifest.json", new)
            save(temporary / "support_migration/source_manifest.json", old)
            for name, label, previous, updated, commits in jobs:
                job_relative = Path(updated["output_dir"]).relative_to(destination)
                target = temporary / job_relative
                save(target / "benchmark_manifest.json", updated)
                data_root = target / name
                old_root = Path(previous["output_dir"]) / name
                for marker, commit in commits:
                    relative = Path(commit["directory"])
                    stable = data_root / relative / "g-memory"
                    shutil.copytree(old_root / relative / "g-memory", stable,
                                    ignore=shutil.ignore_patterns("*-wal", "*-shm"))
                    if files_identity(stable) != commit["files_sha256"]:
                        raise ValueError(f"copied support memory identity mismatch: {label}/{marker.name}")
                    shutil.copy2(old_root / relative.parent / "episode.json", stable.parent.parent / "episode.json")
                    migrated = {**commit, "config_digest": digest(updated)}
                    save(data_root / "support/committed" / marker.name, migrated)
                    save(target / "support_migration" / name / marker.name, commit)
                snapshot = old_root / "snapshot.json"
                if snapshot.exists():
                    state = read(snapshot)
                    state.update(config_digest=digest(updated), support_episode_sha256={
                        marker.name: sha(data_root / "support/committed" / marker.name) for marker, _ in commits})
                    save(data_root / "snapshot.json", state)
                cache = old_root / "support/responses.sqlite"
                if cache.exists():
                    output_cache = data_root / "support/responses.sqlite"
                    output_cache.parent.mkdir(parents=True, exist_ok=True)
                    with sqlite3.connect(cache.as_uri() + "?mode=ro", uri=True) as src:
                        with sqlite3.connect(output_cache) as dst:
                            src.backup(dst)
                print(f"[迁移完成] {label}: {len(commits)} 个 support；模型回复缓存已保留", flush=True)
            save(temporary / "support_migration.json", audit)
            temporary.rename(destination)
        except BaseException:
            shutil.rmtree(temporary)
            raise
        print(f"[迁移完成] 新目录：{destination}；原目录未修改。", flush=True)
        return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Copy validated support; default is validation only")
    args = parser.parse_args(argv)
    migrate(args.source, args.output_dir, apply=args.apply)


if __name__ == "__main__":
    main()
