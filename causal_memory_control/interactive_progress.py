"""Terminal progress and lossless worker-log forwarding; no model imports."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time


STAGE_NAMES = {
    "experts": "专家计划验证", "support": "建立冻结记忆库", "collect": "诊断采集",
    "labels": "CE 标签", "labels_team_u": "Team U 标签", "train_reward": "Reward Gate 训练",
    "train_csm": "CSM CE 训练", "train_team_u": "Local–Global Team U 训练",
    "evaluate": "开始测评", "report": "汇总结果表", "smoke": "环境与流程检查",
}


def show_progress(title, completed=None, total=None, detail=""):
    if completed is None or total is None:
        bar = "[计数待初始化]"
    else:
        fraction = completed / total if total else 1.0
        fill = max(0, min(24, int(24 * fraction)))
        bar = f"[{'#' * fill}{'-' * (24 - fill)}] {completed}/{total} {fraction:.1%}"
    print(f"[进度] {title} {bar}" + (" | " + detail if detail else ""), flush=True)


class InventoryProgress:
    """Limit inventory output while keeping first/final counts visible."""
    def __init__(self):
        self.last = {}

    def __call__(self, title, completed, total, detail=""):
        now = time.monotonic()
        if title not in self.last or completed == total or now - self.last[title] >= 1:
            show_progress(title, completed, total, detail)
            self.last[title] = now


def run_logged(command, path, *, cwd):
    """Forward worker stdout/stderr (including tqdm CRs) to terminal and log.

    Termination of the foreground launcher also stops its isolated worker group,
    including expert grandchildren. It never targets another experiment.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    process = None
    try:
        with path.open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    block = os.read(process.stdout.fileno(), 8192)
                    if not block:
                        break
                    log.write(block)
                    if hasattr(sys.stdout, "buffer"):
                        sys.stdout.buffer.write(block)
                        sys.stdout.buffer.flush()
                    else:
                        sys.stdout.write(block.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                process.stdout.close()
    finally:
        try:
            if process is not None:
                # A completed worker can still leave expert descendants after
                # an exception; the dedicated process group owns only this job.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                # The immediate worker may exit before a descendant that
                # ignores SIGTERM. Do not leave those descendants running.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            signal.signal(signal.SIGTERM, previous)
