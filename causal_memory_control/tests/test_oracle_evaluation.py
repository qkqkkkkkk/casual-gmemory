from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control import fever_experiment, hotpotqa_experiment
from causal_memory_control import run_full_gate_benchmark as full
from causal_memory_control import run_oracle_evaluation as oracle
from causal_memory_control import run_trained_gate_evaluation as trained
from causal_memory_control.planned_drop import build_plan, indexed_rows
from causal_memory_control.tests.test_seed0_gate_matrix import prepare_source
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_st_gate import _STMemory, _FeverSTMacNet
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests


class OracleEvaluationTests(unittest.TestCase):
    def test_latest_test_slice_only_resume_audit_and_sources_unchanged(self):
        for dataset, global_exists in (("fever", True), ("hotpotqa", False)):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory).resolve()
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                adapter, args = prepare_source(root, dataset)
                runner = fever_experiment if dataset == "fever" else hotpotqa_experiment
                if dataset == "fever":
                    for patch in fever_tests.FeverRunnerTests._patches():
                        stack.enter_context(patch)
                    parent = _FeverSTMacNet
                else:
                    hotpot_tests.HotpotRunnerTests.patches(stack)
                    parent = hotpot_tests._HotpotMacNet
                stack.enter_context(mock.patch.object(runner, "GMemory", _STMemory))
                stack.enter_context(mock.patch.object(runner, "MacNet", parent))
                args.output_dir = root / "latest" / dataset
                args.baseline_cache_seed_from = None
                args.gate_evaluation_seed = 2111
                args.kappa, args.max_drops = 0.0, -1
                native = args.output_dir / "native_gmemory"
                runner.main(adapter.evaluation_command(args, mode="always_keep", output=native, resume=False)[4:])
                cache(native / "llm_cache.sqlite")
                methods = ["native_gmemory"]
                if global_exists:
                    plan = args.output_dir / "plans" / "always_global_drop.json"
                    full.save_plan(plan, build_plan(native / "memory_gate.jsonl", policy="global_all"))
                    runner.main(full.planned_command(adapter, args, args.output_dir / "always_global_drop",
                        plan, native / "llm_cache.sqlite")[4:])
                    methods.append("always_global_drop")
                full._save(root / "trained" / "csm_st_manifest.json", {"source_root": str(root / "source")})
                full._save(root / "latest" / "reevaluation_manifest.json", {"schema": trained.SCHEMA,
                    "trained_root": str(root / "trained"), "datasets": {dataset: {
                        "methods": methods, "test_tasks": args.final_claims, "test_offset": args.training_claims,
                        "endpoint": args.endpoint, "model": args.model, "seed": 2111}}})
                originals = {p: p.read_bytes() for name in ("source", "trained", "latest")
                             for p in (root / name).rglob("*") if p.is_file()}
                commands, interrupted = [], [False]

                def execute(label, command, path, **kwargs):
                    commands.append(command)
                    self.assertEqual(command[3], f"causal_memory_control.{dataset}_experiment")
                    self.assertEqual(command[command.index("--mode") + 1], "planned_drop")
                    self.assertEqual(command[command.index("--seed") + 1], "2111")
                    self.assertEqual(command[command.index("--evaluation-offset") + 1], "16")
                    if interrupted[0] and label.endswith("drop_001"):
                        interrupted[0] = False
                        raise KeyboardInterrupt()
                    runner.main(command[4:])

                stack.enter_context(mock.patch.object(full.base, "_run_command", side_effect=execute))
                argv = ["--evaluation-root", str(root / "latest"), "--output-dir", str(root / "oracle"),
                        "--datasets", dataset]
                plan = oracle.main([*argv, "--plan-only"])[dataset]
                self.assertEqual(plan["single_drop_branches_per_task"], 2)
                self.assertEqual(plan["maximum_new_team_inferences"], 4 * (2 + int(not global_exists)))
                self.assertFalse((root / "oracle").exists())
                self.assertEqual(commands, [])
                interrupted[0] = True
                with self.assertRaises(KeyboardInterrupt):
                    oracle.main(argv)
                self.assertEqual(json.loads((root / "oracle" / "oracle_progress.json").read_text())["status"], "interrupted")
                commands.clear()
                report = json.loads(oracle.main(argv).read_text())
                self.assertEqual(len(commands), 1)
                result = report["details"][dataset][oracle.ORACLE]
                self.assertEqual(result["paired_tasks"], 4)
                self.assertTrue(result["paired_control_valid"])
                self.assertGreaterEqual(result["gate_mean_reward"], result["baseline_mean_reward"])
                rows = indexed_rows(root / "oracle" / dataset / oracle.ORACLE / "memory_gate.jsonl")
                self.assertTrue(all(r["oracle"]["uses_test_outcomes"] and not r["oracle"]["deployable"] for r in rows.values()))
                for row in rows.values():
                    self.assertEqual(row["reward"], max(v["reward"] for v in row["oracle"]["choices"]))
                commands.clear()
                oracle.main(argv)
                self.assertEqual(commands, [])
                for path, value in originals.items():
                    self.assertEqual(path.read_bytes(), value, str(path))
                broken_progress = native / "progress.json"
                full._save(broken_progress, {"status": "interrupted"})
                with self.assertRaisesRegex(ValueError, "finish the latest reevaluation first"):
                    oracle.main([*argv, "--output-dir", str(root / "bad"), "--plan-only"])
                self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
