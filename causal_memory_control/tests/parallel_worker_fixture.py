"""Subprocess-only test fixture: real collector/MAS/SQLite, mocked model/backend.

Never used by production commands. Test outputs live only in temporary dirs.
"""

from contextlib import ExitStack
import importlib
import os
from pathlib import Path
import sys
import time
from unittest import mock

from causal_memory_control.tests.test_seed0_subset import CollectionMemory
from causal_memory_control.tests.test_collection_runtime import Backend


def main():
    dataset, argv = sys.argv[1], sys.argv[2:]
    collector = importlib.import_module(f"causal_diagnostic.{dataset}_oracle_recipient.run_experiment")
    args = collector.parse_args(argv)
    ready = Path(os.environ["GMEMORY_TEST_WORKER_READY"])
    ready.mkdir(parents=True, exist_ok=True)
    (ready / str(args.evaluation_offset)).write_text(str(os.getpid()))
    deadline = time.monotonic() + 30
    while len(list(ready.iterdir())) < 2:
        if time.monotonic() >= deadline:
            raise RuntimeError("test requires workers to run concurrently, not sequentially")
        time.sleep(0.05)
    branch, calls = collector._run_branch, [0]
    failed = ready.parent / "failure-injected"

    def interrupt(*values, **kwargs):
        calls[0] += 1
        if (os.environ.get("GMEMORY_TEST_FAIL_OFFSET") == str(args.evaluation_offset)
                and calls[0] == 3 and not failed.exists()):
            failed.touch()
            raise RuntimeError("simulated parallel worker timeout")
        return branch(*values, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(collector, "GMemory", CollectionMemory))
        stack.enter_context(mock.patch.object(collector, "EmbeddingFunc", return_value=None))
        stack.enter_context(mock.patch.object(collector, "_run_branch", side_effect=interrupt))
        stack.enter_context(mock.patch("causal_diagnostic.oracle_recipient.seeded_client.OpenAI", return_value=Backend(dataset)))
        try:
            collector.main(argv)
        except BaseException as exc:
            collector._record_uncaught_error(exc)
            raise


if __name__ == "__main__":
    main()
