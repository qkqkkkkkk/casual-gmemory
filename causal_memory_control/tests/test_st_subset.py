from __future__ import annotations

import copy
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control import hotpotqa_experiment, parallel_collection, train_gate
from causal_memory_control import run_seed0_gate_matrix as seed0
from causal_memory_control import run_st_gate_benchmark as st
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_parallel_collection import probes
from causal_memory_control.tests.test_seed0_gate_matrix import prepare_source
from causal_memory_control.tests.test_semantic_jev import TinyEncoder
from causal_memory_control.tests.test_st_gate import _STMemory


def previous_subset(root, dataset, *, parallel=False, repeats=2):
    """Register measured fixture branches as a completed old subset matrix."""
    adapter, original = prepare_source(root, dataset)
    snapshot = st.base._load_json(original.memory_dir / "causal_snapshot_manifest.json")
    rows = [json.loads(line) for line in seed0.source_path(original).read_text().splitlines()]
    recorded = copy.copy(original)
    recorded.output_dir = root / "previous" / dataset
    recorded.diagnostic_task_limit = 10
    recorded.diagnostic_results = root / "previous" / "artifacts" / dataset / "diagnostic"
    recorded.repeats = repeats
    if parallel:
        recorded.collection_endpoints = [original.endpoint, "http://127.0.0.1:11438/v1"]
    for worker in seed0.collection_workers(recorded):
        ids = seed0.diagnostic_ids(worker)
        design = {**adapter.checkpoint_context(worker, snapshot),
            "candidate_scope": "all_retrieved", "collection_scope": "gate_training",
            "candidate_specs": [{"kind": "trajectory", "index": 0}, {"kind": "insight", "index": 0}],
            "evaluation_ids": ids, "evaluation_offset": getattr(worker, "diagnostic_task_offset", 0),
            **{k: getattr(worker, k) for k in ("repeats", "temperature", "selection_seed")}}
        run = {**design, "endpoint": worker.endpoint, "sample_seed_base": worker.sample_seed_base}
        manifest = {"design": design, "run": run,
            "design_hash": seed0.stable_hash(design), "run_hash": seed0.stable_hash(run)}
        path = seed0.source_path(worker)
        st.full._save(path.parent / "run_manifest.json", manifest)
        st.full._save(path.parent / "collection_progress.json", {"status": "completed"})
        selected = [{**r, "design_hash": manifest["design_hash"], "run_hash": manifest["run_hash"]}
            for r in rows if r["task_id"] in ids and r["repeat_index"] < repeats]
        path.write_text("".join(json.dumps(r) + "\n" for r in selected))
    if parallel:
        st.full._save(recorded.diagnostic_results / "parallel_collection_manifest.json", {
            "schema": parallel_collection.SCHEMA,
            "snapshot_sha256": st.file_sha256(original.memory_dir / "causal_snapshot_manifest.json"),
            "replicas": probes(recorded.collection_endpoints, original.model),
            "workers": [{"endpoint": w.endpoint, "offset": w.diagnostic_task_offset,
                "task_ids": seed0.diagnostic_ids(w), "repeats": w.repeats,
                "sample_seed_base": w.sample_seed_base, "source": str(seed0.source_path(w))}
                for w in seed0.collection_workers(recorded)]})
    st.full._save(root / "previous" / "seed0_manifest.json", {
        "schema": seed0.SCHEMA, "source_root": str(root / "source"),
        "experiments": {dataset: adapter.configuration(recorded)}})
    st.full._save(recorded.output_dir / "baseline_selection.json",
        {"path": str(recorded.output_dir / "native_gmemory")})
    return adapter, original, recorded


def options(root, dataset):
    return ["--source-root", str(root / "source"), "--seed0-results-root", str(root / "previous"),
        "--output-dir", str(root / "st"), "--datasets", dataset, "--seed0-only",
        "--seed", "23", "--st-beta", "0.01", "--ensemble-size", "2",
        "--st-hidden-sizes", "8,4", "--st-epochs", "2"]


class STSubsetTests(unittest.TestCase):
    def test_serial_and_sharded_plans_preserve_original_test_boundary_without_writes(self):
        for dataset, parallel in (("fever", False), ("hotpotqa", False), ("hotpotqa", True)):
            with self.subTest(dataset=dataset, parallel=parallel), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    _, _, recorded = previous_subset(root, dataset, parallel=parallel, repeats=1)
                    with mock.patch.object(st.base, "_run_command") as run:
                        plan = st.main([*options(root, dataset), "--plan-only"])[dataset]
                    run.assert_not_called()
                self.assertEqual(plan["gate_training_questions"], 10)
                self.assertEqual(plan["original_diagnostic_partition_questions"], 16)
                self.assertEqual(plan["unused_diagnostic_questions"], 6)
                self.assertEqual(plan["diagnostic_branches_per_seed"], 40)
                self.assertEqual(plan["new_diagnostic_branches"], 0)
                self.assertEqual(plan["final_test_questions"], 4)
                self.assertEqual(plan["final_test_offset"], 16)
                self.assertEqual(plan["sample_seeds"], [0])
                self.assertEqual(plan["training_sources"], [str(p) for p in seed0.training_sources(recorded)])
                self.assertFalse((root / "st").exists())

    def test_hotpot_only_trains_st_reuses_subset_baseline_and_old_controls_and_resumes(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory).resolve()
                adapter, original, recorded = previous_subset(root, "hotpotqa", parallel=parallel)
                hotpot_tests.HotpotRunnerTests.patches(stack)
                stack.enter_context(mock.patch.object(hotpotqa_experiment, "GMemory", _STMemory))
                stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=TinyEncoder()))
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                commands = []

                def execute(label, command, path, **_):
                    commands.append((label, command))
                    if command[3] == "causal_memory_control.train_gate":
                        train_gate.main(command[4:])
                    elif command[3] == "causal_memory_control.hotpotqa_experiment":
                        hotpotqa_experiment.main(command[4:])
                        if label.endswith(":baseline"):
                            cache(Path(command[command.index("--output-dir") + 1]) / "llm_cache.sqlite")
                    else:
                        self.fail(f"unexpected collection/FEVER/old-method command: {command}")

                stack.enter_context(mock.patch.object(st.base, "_run_command", side_effect=execute))
                # Finish the fixture's old matrix so all report rows have real,
                # audited logs, including the no-memory control.
                old_options = ["--source-root", str(root / "source"), "--output-dir", str(root / "previous"),
                    "--datasets", "hotpotqa", "--diagnostic-tasks", "10", "--collection-repeats", "2",
                    "--include-always-global-drop"]
                if parallel:
                    old_options += ["--collection-endpoints", ",".join(recorded.collection_endpoints)]
                # Match the matrix's immutable configuration for this fixture.
                seed0.full._save(root / "previous" / "seed0_manifest.json", {
                    "schema": seed0.SCHEMA, "source_root": str(root / "source"), "datasets": ["hotpotqa"],
                    "experiments": {"hotpotqa": adapter.configuration(self._matrix_args(recorded))},
                    "training_protocol": "single-group-streamed-v1", "arms": seed0.matrix.DEFAULT_ARMS})
                seed0.main(old_options)
                before = {p: p.read_bytes() for folder in (root / "source", root / "previous")
                    for p in folder.rglob("*") if p.is_file()}
                # Missing original full collection must not trigger recollection.
                seed0.source_path(original).unlink()
                before.pop(seed0.source_path(original))
                commands.clear()
                report = st.base._load_json(st.main(options(root, "hotpotqa")))
                self.assertEqual([label for label, _ in commands], ["hotpotqa:train_st", "hotpotqa:evaluate_st"])
                training, evaluation = [c for _, c in commands]
                sources = training[training.index("--input") + 1:training.index("--output")]
                self.assertEqual(sources, [str(p) for p in seed0.training_sources(recorded)])
                self.assertEqual(evaluation[evaluation.index("--evaluation-offset") + 1], "16")
                self.assertEqual(evaluation[evaluation.index("--claims") + 1], "4")
                payload = st.base._load_json(root / "st" / "hotpotqa" / "checkpoint.json")
                metadata = payload["training_metadata"]
                old = st.base._load_json(root / "previous" / "hotpotqa" / "checkpoints" / "semantic_mlp.json")
                self.assertEqual(metadata["fit_task_ids"], old["training_metadata"]["fit_task_ids"])
                self.assertEqual(metadata["policy_calibration"]["calibration_task_ids"],
                    old["training_metadata"]["policy_calibration"]["calibration_task_ids"])
                self.assertEqual(payload["estimator"]["config"]["beta"], 0.01)
                self.assertEqual(len(metadata["training_task_ids"]), 10)
                protocol = report["data_protocol"]["hotpotqa"]
                self.assertEqual((protocol["diagnostic_tasks"], protocol["final_test_offset"], protocol["final_test_tasks"]), (10, 16, 4))
                self.assertTrue(report["details"]["hotpotqa"]["semantic_st"]["result"]["paired_control_valid"])
                self.assertTrue(report["details"]["hotpotqa"]["always_global_drop"]["reused"])
                metrics = st.base._load_json(root / "st" / "hotpotqa" / "st_metrics.json")
                self.assertEqual(len(metrics["results"]), 6)
                self.assertIn("answer_em", metrics["results"][-1])
                self.assertIn("answer_f1", metrics["results"][-1])
                self.assertIn("| F1 | EM |", (root / "st" / "hotpotqa" / "st_metrics.md").read_text())
                commands.clear()
                st.main(options(root, "hotpotqa"))
                self.assertFalse(commands)
                for path, contents in before.items():
                    self.assertEqual(path.read_bytes(), contents, str(path))

    @staticmethod
    def _matrix_args(recorded):
        args = copy.copy(recorded)
        args.checkpoint = args.output_dir / "unused.json"
        args.semantic_cache = args.output_dir / "semantic_cache.sqlite"
        args.kappa = "auto"
        return args

    def test_partial_workers_and_tampered_provenance_never_collect_or_train(self):
        for damage in ("branch", "replica", "test_offset", "location", "count"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    _, _, recorded = previous_subset(root, "hotpotqa", parallel=True)
                    if damage == "branch":
                        path = seed0.training_sources(recorded)[1]
                        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
                    elif damage == "replica":
                        path = recorded.diagnostic_results / "parallel_collection_manifest.json"
                        payload = st.base._load_json(path)
                        payload["replicas"][1]["digest"] = "different"
                        st.full._save(path, payload)
                    else:
                        path = root / "previous" / "seed0_manifest.json"
                        payload = st.base._load_json(path)
                        key, value = {"test_offset": ("training_claims", 10),
                            "location": ("diagnostic_results", str(root / "unregistered")),
                            "count": ("diagnostic_task_limit", 17)}[damage]
                        payload["experiments"]["hotpotqa"][key] = value
                        st.full._save(path, payload)
                    with mock.patch.object(st.base, "_run_command") as run:
                        with self.assertRaises((ValueError, RuntimeError)):
                            st.main(options(root, "hotpotqa"))
                        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
