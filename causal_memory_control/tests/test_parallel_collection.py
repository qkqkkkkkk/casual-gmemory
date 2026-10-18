from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from causal_memory_control import parallel_collection as parallel
from causal_memory_control import run_seed0_gate_matrix as runner, train_gate
from causal_memory_control import fever_experiment, hotpotqa_experiment
from causal_memory_control.tests.test_seed0_subset import options
from causal_memory_control.tests.test_seed0_gate_matrix import prepare_source
from causal_memory_control.tests.test_semantic_jev import TinyEncoder
from causal_memory_control.tests.test_st_gate import _STMemory, _FeverSTMacNet
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests


def probes(endpoints, model):
    return [{"endpoint": endpoint, "model": model, "digest": "fixed-test-weights",
        "ollama_version": "test-version", "model_settings_sha256": "fixed-test-settings"}
        for endpoint in endpoints]


class ParallelCollectionTests(unittest.TestCase):
    def test_shards_endpoint_validation_and_replica_identity_fail_closed(self):
        self.assertEqual(parallel.shard_ranges(1000, 2), [(0, 500), (500, 500)])
        self.assertEqual(parallel.shard_ranges(11, 2), [(0, 6), (6, 5)])
        parallel.validate_union([[1, 2], [3, 4]], [1, 2, 3, 4])
        for shards in ([[1, 2], [2, 3]], [[1], [2]]):
            with self.assertRaises(ValueError):
                parallel.validate_union(shards, [1, 2, 3, 4])
        endpoints = ["http://127.0.0.1:11434/v1", "http://127.0.0.1:11438/v1"]
        self.assertEqual(parallel.parse_endpoints(",".join(endpoints)), endpoints)
        for value in (endpoints[0], ",".join([endpoints[0]] * 2),
                "http://user:secret@host/v1,http://other/v1", "http://host/not-v1,http://other/v1"):
            with self.assertRaises(ValueError):
                parallel.parse_endpoints(value)
        calls = []

        def request(url, payload=None):
            calls.append((url, payload))
            if url.endswith("/api/tags"):
                return {"models": [{"name": "test-model", "digest": "same-weights"}]}
            if url.endswith("/api/show"):
                return {"template": "same template", "model_info": {"context_length": 32768}, "parameters": "same"}
            return {"version": "same-version"}

        with mock.patch.object(parallel, "_request_json", side_effect=request):
            results = parallel.verify_replicas(endpoints, "test-model")
        self.assertEqual(len(calls), 6)
        self.assertEqual(results[0]["digest"], results[1]["digest"])
        for field in ("digest", "ollama_version", "model_settings_sha256"):
            values = probes(endpoints, "test-model")
            values[1][field] = "different"
            with mock.patch.object(parallel, "probe_replica", side_effect=values):
                with self.assertRaisesRegex(ValueError, field):
                    parallel.verify_replicas(endpoints, "test-model")
        with mock.patch.object(parallel, "_request_json", return_value={"models": []}):
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                parallel.probe_replica(endpoints[0], "test-model")

    def test_plan_only_and_output_lock(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            _, original = prepare_source(root, "hotpotqa")
            endpoints = original.endpoint + ",http://127.0.0.1:11438/v1"
            argv = [*options(root, "hotpotqa"), "--collection-endpoints", endpoints]
            with mock.patch.object(parallel, "verify_replicas") as verify:
                plan = runner.main([*argv, "--plan-only"])["hotpotqa"]
                verify.assert_not_called()
            self.assertEqual([(w["offset"], w["tasks"]) for w in plan["parallel_workers"]], [(0, 5), (5, 5)])
            self.assertEqual(len(plan["training_sources"]), 2)
            self.assertFalse((root / "quick").exists())
            with parallel.exclusive_run(root / "locked"):
                with self.assertRaisesRegex(RuntimeError, "another parallel"):
                    with parallel.exclusive_run(root / "locked"):
                        self.fail("second controller acquired the same output")
            with parallel.exclusive_run(root / "locked"):
                pass

    def test_real_subprocess_workers_union_training_and_resume_for_both_datasets(self):
        for dataset, inject_failure in (("fever", False), ("hotpotqa", True)):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                adapter, original = prepare_source(root, dataset)
                cache(runner.source_path(original).parent / "llm_cache.sqlite")
                override = root / "serial-progress" / "llm_cache.sqlite"
                if dataset == "hotpotqa":
                    override.parent.mkdir()
                    shutil.copy2(runner.source_path(original).parent / "llm_cache.sqlite", override)
                    shutil.copy2(runner.source_path(original).parent / "run_manifest.json", override.parent / "run_manifest.json")
                original_bytes = {p: p.read_bytes() for p in (root / "source").rglob("*") if p.is_file()}
                evaluation = fever_experiment if dataset == "fever" else hotpotqa_experiment
                if dataset == "fever":
                    for patch in fever_tests.FeverRunnerTests._patches():
                        stack.enter_context(patch)
                    stack.enter_context(mock.patch.object(evaluation, "MacNet", _FeverSTMacNet))
                else:
                    hotpot_tests.HotpotRunnerTests.patches(stack)
                stack.enter_context(mock.patch.object(evaluation, "GMemory", _STMemory))
                stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=TinyEncoder()))
                stack.enter_context(mock.patch.object(parallel, "verify_replicas", side_effect=probes))
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                real_popen = subprocess.Popen
                worker_commands, evaluation_commands = [], []

                def spawn(command, **kwargs):
                    worker_commands.append(command)
                    kwargs["env"] = {**kwargs["env"], "GMEMORY_TEST_WORKER_READY": str(root / "ready"),
                        "OPENAI_API_KEY": "EMPTY"}
                    if inject_failure:
                        kwargs["env"]["GMEMORY_TEST_FAIL_OFFSET"] = "5"
                    # Separate Python processes exercise global RNG/progress
                    # isolation and concurrent native collector scheduling.
                    transformed = [sys.executable, "-u", "-m",
                        "causal_memory_control.tests.parallel_worker_fixture", dataset, *command[4:]]
                    return real_popen(transformed, **kwargs)

                def execute(label, command, log, **_):
                    evaluation_commands.append(label)
                    if label.endswith(":train"):
                        self.assertEqual(command[command.index("--input") + 3], "--metric")
                        train_gate.main(command[4:])
                    elif label.endswith(":baseline") or label.endswith(":evaluate"):
                        self.assertEqual(command[command.index("--endpoint") + 1], original.endpoint)
                        self.assertEqual(command[command.index("--evaluation-offset") + 1], "16")
                        evaluation.main(command[4:])
                        if label.endswith(":baseline"):
                            cache(Path(command[command.index("--output-dir") + 1]) / "llm_cache.sqlite")
                    else:
                        self.fail(f"unexpected recollection/snapshot/retest: {label}")

                stack.enter_context(mock.patch.object(parallel.subprocess, "Popen", side_effect=spawn))
                stack.enter_context(mock.patch.object(runner.base, "_run_command", side_effect=execute))
                argv = [*options(root, dataset), "--collection-endpoints", original.endpoint + ",http://127.0.0.1:11438/v1"]
                if dataset == "hotpotqa":
                    argv += ["--collection-cache-seed-from", str(override)]
                prefix = {}
                if inject_failure:
                    with self.assertRaises(runner.base.StageFailure):
                        runner.main(argv)
                    state = runner.base._load_json(root / "quick" / "artifacts" / dataset / "diagnostic" / "parallel_collection_progress.json")
                    self.assertEqual(state["status"], "failed")
                    self.assertIn("worker_001", state["error"]["message"])
                    self.assertIn("simulated parallel worker timeout", state["workers"]["worker_001"]["log_tail"])
                    for p in (root / "quick" / "artifacts").rglob("branches.jsonl"):
                        prefix[p] = p.read_bytes()
                result_path = runner.main(argv)
                self.assertEqual(len(list((root / "ready").iterdir())), 2)
                sources = list((root / "quick" / "artifacts").rglob("branches.jsonl"))
                self.assertEqual(len(sources), 2)
                task_sets = []
                for source in sorted(sources):
                    rows = [json.loads(line) for line in source.read_text().splitlines()]
                    self.assertEqual(len(rows), 40)
                    self.assertEqual({r["sample_seed"] for r in rows}, {0, 1})
                    task_sets.append({r["task_id"] for r in rows})
                    manifest = runner.base._load_json(source.parent / "run_manifest.json")
                    self.assertTrue(all(r["run_hash"] == manifest["run_hash"] for r in rows))
                    expected_cache = override if dataset == "hotpotqa" else runner.source_path(original).parent / "llm_cache.sqlite"
                    self.assertEqual(manifest["run"]["cache_seed_from"], str(expected_cache.resolve()))
                self.assertFalse(task_sets[0] & task_sets[1])
                snapshot = runner.base._load_json(original.memory_dir / "causal_snapshot_manifest.json")
                self.assertEqual(task_sets[0] | task_sets[1], set(snapshot["evaluation_ids"][:10]))
                for path, contents in prefix.items():
                    self.assertTrue(path.read_bytes().startswith(contents))
                results = json.loads(result_path.read_text())
                self.assertEqual(len(results["results"][dataset]), 5)
                self.assertTrue(all(c["paired_control_valid"] for c in results["details"][dataset].values()))
                for arm in runner.matrix.DEFAULT_ARMS:
                    payload = runner.base._load_json(root / "quick" / dataset / "checkpoints" / (arm + ".json"))
                    self.assertEqual(len(payload["training_metadata"]["training_sources"]), 2)
                    self.assertEqual(set(payload["training_metadata"]["training_task_ids"]), task_sets[0] | task_sets[1])
                for path, contents in original_bytes.items():
                    self.assertEqual(path.read_bytes(), contents, str(path))
                worker_commands.clear()
                evaluation_commands.clear()
                runner.main(argv)
                self.assertFalse(worker_commands or evaluation_commands)
                # Completed data cannot be used with edited shard provenance.
                manifest_path = root / "quick" / "artifacts" / dataset / "diagnostic" / "parallel_collection_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["workers"][1]["offset"] = 0
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "parallel replica provenance"):
                    runner.main(argv)


if __name__ == "__main__":
    unittest.main()
