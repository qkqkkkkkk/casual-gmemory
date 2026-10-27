"""Check actual terminal/log behavior using harmless subprocesses only."""
from contextlib import redirect_stdout
import io
import fcntl
from pathlib import Path
import os
import selectors
import subprocess
import sys
import tempfile
import time
import unittest

from causal_memory_control.interactive_data import inventory
from causal_memory_control.interactive_progress import run_logged


REPO = Path(__file__).resolve().parents[2]


class TerminalProgressTests(unittest.TestCase):
    def test_worker_output_arrives_before_exit_and_is_preserved_in_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            worker = "import sys,time; print('first',flush=True); time.sleep(1); sys.stderr.write('bar\\rfinished\\n')"
            parent = "from causal_memory_control.interactive_progress import run_logged; import sys; run_logged([sys.executable,'-u','-c',sys.argv[2]],sys.argv[1],cwd=sys.argv[3])"
            with subprocess.Popen([sys.executable, "-u", "-c", parent, str(log), worker, str(REPO)],
                                  cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    self.assertTrue(selector.select(timeout=5), "no live terminal output")
                self.assertEqual(process.stdout.readline(), b"first\n")
                self.assertIsNone(process.poll(), "first output was buffered until exit")
                out, err = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, err)
            self.assertEqual(out, b"bar\rfinished\n")
            self.assertEqual(log.read_bytes(), b"first\nbar\rfinished\n")

    def test_worker_failure_preserves_output_and_return_code(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            log = Path(directory) / "worker.log"
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                run_logged([sys.executable, "-u", "-c", "print('failure detail'); raise SystemExit(7)"], log, cwd=REPO)
            self.assertEqual(raised.exception.returncode, 7)
            self.assertEqual(output.getvalue(), "failure detail\n")
            self.assertEqual(log.read_text(), output.getvalue())

    def test_terminating_launcher_stops_its_worker_and_expert_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "worker.log"
            lease = Path(directory) / "descendant.lock"
            descendant = ("import fcntl,os,signal,time; lease=open(" + repr(str(lease)) + ",'w'); "
                          "fcntl.flock(lease,fcntl.LOCK_EX); signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                          "print('ready',os.getpid(),flush=True); time.sleep(30)")
            worker = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-u','-c',sys.argv[1]]); time.sleep(30)"
            parent = "from causal_memory_control.interactive_progress import run_logged; import sys; run_logged([sys.executable,'-u','-c',sys.argv[2],sys.argv[3]],sys.argv[1],cwd=sys.argv[4])"
            with subprocess.Popen([sys.executable, "-u", "-c", parent, str(log), worker, descendant, str(REPO)],
                                  cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    self.assertTrue(selector.select(timeout=5), "worker did not start")
                line = process.stdout.readline()
                self.assertTrue(line.startswith(b"ready "), line)
                descendant_pid = int(line.split()[1])
                try:
                    process.terminate()
                    _, err = process.communicate(timeout=8)
                    self.assertEqual(process.returncode, 143, err)
                    with lease.open() as handle:
                        for _ in range(20):
                            try:
                                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except BlockingIOError:
                                time.sleep(.1)
                        else:
                            self.fail("expert descendant still holds its lock after launcher exit")
                finally:
                    try:
                        os.kill(descendant_pid, 9)
                    except ProcessLookupError:
                        pass

    def test_inventory_progress_does_not_change_task_selection(self):
        events = []
        plain = inventory(REPO, ["pddl"], pddl_folds=5)
        visible = inventory(REPO, ["pddl"], pddl_folds=5, progress=lambda *args: events.append(args))
        self.assertEqual(plain, visible)
        scan = [e for e in events if e[0] == "数据采样 / PDDL 扫描"]
        self.assertEqual(scan[0][1:3], (0, 60))
        self.assertEqual(scan[-1][1:3], (60, 60))
        self.assertIn("5 折", events[-1][3])

    def test_direct_viewer_entry_does_not_shadow_stdlib_types(self):
        result = subprocess.run([sys.executable, str(REPO / "causal_memory_control/watch_interactive_progress.py"), "--help"],
                                cwd=REPO, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
