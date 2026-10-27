"""Read-only terminal progress bars for existing PDDL/ALFWorld runs.

Run this file directly: it uses only the standard library and never imports the
benchmark, a model, or an environment. Its filename deliberately falls outside
the benchmark's interactive_*.py implementation fingerprint.
"""
from __future__ import annotations

# Direct script execution puts causal_memory_control/ ahead of the stdlib.
# Remove that directory before argparse imports enum -> types; this package
# contains its own types.py (and logging.py), which must not shadow the stdlib.
import sys
if __name__ == "__main__" and not __package__:
    import os
    _script_directory = os.path.realpath(os.path.dirname(__file__))
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.curdir) != _script_directory]

import argparse
import hashlib
import json
import math
from pathlib import Path
import os
import time


class Reader:
    def __init__(self):
        self.cache = {}

    def json(self, path):
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if path not in self.cache or self.cache[path][0] != stamp:
                self.cache[path] = (stamp, json.loads(path.read_text()))
            return self.cache[path][1]
        except FileNotFoundError:
            return None


def tail(path, limit=32768):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
    except FileNotFoundError:
        return []


def jobs(root):
    result = [(name, root / name) for name in ("pddl", "alfworld")]
    result += [("pddl/" + fold.name, fold / "pddl")
               for fold in sorted((root / "pddl_folds").glob("fold_*"))]
    return result


def active_job(root, pipeline, reader):
    candidates = jobs(root)
    for label, data in candidates:
        if label == pipeline.get("dataset"):
            return label, data
    # The existing parent drops dataset/command when recording a failure.
    # Recover the failed child from its own progress rather than showing '-'.
    matching = []
    for label, data in candidates:
        path = data / "progress.json"
        state = reader.json(path) or {}
        if state.get("stage") == pipeline.get("stage") and state.get("status") == "failed":
            matching.append((path.stat().st_mtime_ns, label, data))
    if matching:
        _, label, data = max(matching)
        return label, data
    return None, None


def task_digest(task):
    payload = json.dumps(task, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def saved_tasks(directory, ids):
    expected = {task_digest(task) for task in ids}
    observed = {p.stem for p in directory.glob("*.json")}
    return len(expected & observed), len(expected)


def count_progress(data, stage, state, reader):
    config = reader.json(data.parent / "benchmark_manifest.json") or {}
    split = config.get("experiments", {}).get(data.name, {}).get("split", {})
    diagnostic = split.get("fit", []) + split.get("dev", [])
    if split and stage in ("experts", "collect", "evaluate"):
        directory, ids = {
            "experts": ("experts", diagnostic),
            "collect": ("diagnostic/tasks", diagnostic),
            "evaluate": ("evaluation/policy_parts", split.get("test", [])),
        }[stage]
        done, total = saved_tasks(data / directory, ids)
        return "已保存任务", done, total
    if split and stage == "support":
        total = len(split.get("support", []))
        done = sum((data / "support/committed" / f"{i:04d}.json").exists() for i in range(total))
        return "已提交建库任务", done, total
    if stage == "labels":
        for name in ("augmented_parts", "original_parts"):
            progress = reader.json(data / "csm_st" / (name + "_progress.json"))
            if progress:
                done = progress.get("completed", 0)
                if progress.get("status") == "completed":
                    done = progress["total"]
                return "CE " + name, done, progress["total"]
    if stage == "train_reward":
        methods = ("semantic_linear", "semantic_mlp", "semantic_st")
        return "已保存模型", sum((data / m / "checkpoint.json").exists() for m in methods), len(methods)
    if state.get("stage") == stage:
        done = state.get("completed_tasks", state.get("completed"))
        total = state.get("total_tasks", state.get("total"))
        if done is not None and total is not None:
            return "阶段任务", done, total
    return "阶段进度", None, None


def bar(title, done, total, width=28):
    if done is None or total is None:
        return f"{title} [{'?' * width}] 当前阶段未记录可用计数"
    fraction = done / total if total else 1.0
    filled = max(0, min(width, int(width * fraction)))
    return f"{title} [{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({fraction:.1%})"


def render(root, reader, log_lines=4, full_error=False):
    pipeline = reader.json(root / "pipeline_progress.json") or {}
    if not pipeline:
        return "等待 pipeline_progress.json；查看器只读取文件，不启动实验。", None
    status, stage = pipeline.get("status", "unknown"), pipeline.get("stage", "-")
    label, data = active_job(root, pipeline, reader)
    lines = ["PDDL / ALFWorld 实时阶段进度", f"记录状态: {status} | 阶段: {stage} | 数据集/折: {label or '-'}"]
    if status == "completed" and pipeline.get("stages"):
        stages = pipeline["stages"]
        lines += [bar("本次命令已完成阶段", len(stages), len(stages)), ", ".join(stages)]
    child_error = False
    if data:
        state = reader.json(data / "progress.json") or {}
        lines.append(bar(*count_progress(data, stage, state, reader)))
        if state.get("stage") == stage:
            if state.get("current_task"):
                lines.append("当前任务: " + str(state["current_task"]))
            for key in ("repeat", "single_drop_index"):
                if key in state:
                    lines.append(f"{key} (从 0 开始): {state[key]}")
        log = data / "logs" / (stage + ".log")
        recent = [line for line in tail(log) if line.strip()]
        for line in reversed(recent):
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if "expert_phase" in event:
                    lines.append(f"最后专家事件: {event['task_id']} / {event['expert_phase']} "
                                 f"(事件时已用 {event['elapsed_seconds']}s)")
                    break
        if stage in ("train_csm", "train_team_u"):
            method = "csm_st" if stage == "train_csm" else "local_global_team_u"
            history = reader.json(data / method / "loss_history.json")
            if isinstance(history, list) and history:
                lines.append("最近 epoch: " + json.dumps(history[-1], ensure_ascii=False))
        if status == "failed":
            errors = list((data / "errors").glob(stage + "-*.json"))
            if errors:
                error = max(errors, key=lambda p: p.stat().st_mtime_ns)
                detail = reader.json(error) or {}
                trace = str(detail.get("traceback", "")).strip()
                lines += ["错误文件: " + str(error), trace if full_error else trace.splitlines()[-1] if trace else ""]
                child_error = bool(trace)
        lines.append("阶段日志: " + str(log))
        if log_lines:
            lines += recent[-log_lines:]
    if pipeline.get("error") and not child_error:
        lines.append("父进程错误: " + str(pipeline["error"]))
    if status == "failed":
        lines.append("该阶段已失败；上面的计数是已保存工作，不表示仍在执行。")
    lines.append("Ctrl+C 退出查看。状态来自保存的文件；手动终止后 running 记录可能滞后。")
    return "\n".join(lines), status


class LogCursor:
    """Read new log bytes, with a small initial tail when attaching to a run."""
    def __init__(self):
        self.path, self.inode, self.offset, self.pending = None, None, 0, ""

    def lines(self, path):
        try:
            stat = path.stat()
            if self.path != path or self.inode != stat.st_ino or self.offset > stat.st_size:
                self.path, self.inode = path, stat.st_ino
                self.offset, self.pending = max(0, stat.st_size - 8192), ""
                skip_partial = self.offset > 0
            else:
                skip_partial = False
            with path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read(65536)
                self.offset = stream.tell()
            text = self.pending + chunk.decode("utf-8", errors="replace").replace("\r", "\n")
            parts = text.split("\n")
            self.pending = parts.pop()
            if skip_partial and parts:
                parts.pop(0)
            return [line for line in parts if line.strip()]
        except FileNotFoundError:
            return []


def run_pid(root):
    """Linux liveness check for an attached run; None means unavailable."""
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / "cmdline").read_bytes().decode(errors="replace").split(chr(0))
            if "causal_memory_control.run_interactive_gate_benchmark" in args and "--output-dir" in args:
                output = args[args.index("--output-dir") + 1]
                if Path(output).resolve() == root:
                    return int(entry.name)
        except (OSError, IndexError):
            continue
    return 0


def stream_updates(root, reader, interval, *, parent_pid=None, started_at=0):
    cursor, last_stage, last_progress, last_heartbeat = LogCursor(), None, None, 0.0
    owner = run_pid(root) if parent_pid is None else parent_pid
    last_read_error = None
    interrupt = "Ctrl+C 会中断前台实验。" if parent_pid is not None else "Ctrl+C 仅退出查看，实验继续运行。"
    print("实时输出阶段、任务进度和新增阶段日志；" + interrupt, flush=True)
    while True:
        try:
            pipeline_file = root / "pipeline_progress.json"
            pipeline = reader.json(pipeline_file) or {}
            if pipeline and started_at and pipeline_file.stat().st_mtime < started_at:
                pipeline = {}
            if not pipeline:
                if last_stage != "initializing":
                    print("[初始化] 等待本次阶段记录；启动信息和异常也会输出到当前终端。", flush=True)
                    last_stage = "initializing"
            else:
                status, stage = pipeline.get("status", "unknown"), pipeline.get("stage", "-")
                label, data = active_job(root, pipeline, reader)
                marker = (stage, label, status)
                if marker != last_stage:
                    print(f"\n[阶段] {stage} | 数据集/折: {label or '-'} | 状态: {status}", flush=True)
                    last_stage = marker
                if data:
                    state = reader.json(data / "progress.json") or {}
                    count = count_progress(data, stage, state, reader)
                    task = state.get("current_task") if state.get("stage") == stage else None
                    fields = tuple((k, state[k]) for k in ("repeat", "single_drop_index")
                                   if state.get("stage") == stage and k in state)
                    current = (marker, count, task, fields)
                    now = time.monotonic()
                    if current != last_progress or now - last_heartbeat >= 30:
                        suffix = f" | 当前任务: {task}" if task else ""
                        if fields:
                            suffix += " | " + ", ".join(f"{k}={v}" for k, v in fields)
                        print("[进度] " + bar(*count) + suffix, flush=True)
                        last_progress, last_heartbeat = current, now
                    for line in cursor.lines(data / "logs" / (stage + ".log")):
                        # The complete raw output remains in the stage log.
                        if line.strip() in ("0.0", "1.0"):
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            event = None
                        if isinstance(event, dict) and "expert_phase" in event:
                            details = {k: v for k, v in event.items()
                                       if k not in ("expert_phase", "task_id", "elapsed_seconds")}
                            suffix = " " + json.dumps(details, ensure_ascii=False) if details else ""
                            print(f"[专家] {event['task_id']} -> {event['expert_phase']} "
                                  f"({event['elapsed_seconds']}s){suffix}", flush=True)
                        else:
                            print(line, flush=True)
                if status in ("completed", "failed"):
                    if status == "failed":
                        print(render(root, reader, log_lines=0, full_error=True)[0], flush=True)
                    else:
                        print("[完成] 本次阶段: " + ", ".join(pipeline.get("stages", [])), flush=True)
                    return 1 if status == "failed" else 0
            last_read_error = None
        except (OSError, ValueError, KeyError) as exc:
            # Progress writes may be in flight on a shared filesystem. A display
            # read failure must never terminate the experiment or its monitor.
            message = f"[读取进度暂不可用，将重试] {exc}"
            if message != last_read_error:
                print(message, flush=True)
                last_read_error = message
        # A monitor launched by the shell keeps the same parent PID across
        # exec env -> flock -F -> Python, and exits after that parent exits.
        if parent_pid is not None:
            alive = os.getppid() == parent_pid
        elif owner is not None:
            if not owner:
                owner = run_pid(root)
            try:
                alive = bool(owner) and (os.kill(owner, 0) is None)
            except ProcessLookupError:
                alive = False
        else:
            alive = True
        if not alive:
            print("[已停止] 主进程已退出；以上是最后保存的进度，请查看终端中的启动/异常输出。", flush=True)
            return 1
        time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--log-lines", type=int, default=4)
    parser.add_argument("--once", action="store_true", help="Print one snapshot, including the saved failure traceback")
    parser.add_argument("--stream", action="store_true", help="Append stage/task progress and new worker logs to this terminal")
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--started-at", type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 0.2 or args.log_lines < 0:
        parser.error("interval must be >=0.2 seconds and log-lines must be >=0")
    reader = Reader()
    try:
        if args.stream:
            return stream_updates(args.output_dir.resolve(), reader, args.interval,
                                  parent_pid=args.parent_pid, started_at=args.started_at)
        while True:
            try:
                content, status = render(args.output_dir.resolve(), reader, args.log_lines, full_error=args.once)
            except (OSError, ValueError, KeyError) as exc:
                content, status = f"读取进度失败: {exc}", None
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(content, flush=True)
            if args.once:
                return 1 if status == "failed" else 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已退出进度查看。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
