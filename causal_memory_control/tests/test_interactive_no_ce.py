"""CE-excluded runs must retain Team U and never require expert/CE artifacts."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from causal_memory_control import interactive_team_u as team
from causal_memory_control import interactive_training as training
from causal_memory_control import interactive_worker as worker
from causal_memory_control import run_interactive_gate_benchmark as cli
from causal_memory_control.interactive_data import save
from causal_memory_control.interactive_protocol import learned_methods, policies, stages
from causal_memory_control.interactive_report import report
from causal_memory_control.tests.test_interactive_team_u import fixture, collect


class ExcludeCETests(unittest.TestCase):
    def test_plan_has_eight_policies_no_ce_stages_or_expert_budget(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as out:
            target = Path(directory) / "not-created"
            cli.main(["--datasets", "pddl", "--pddl-folds", "5", "--include-local-global-team-u",
                      "--exclude-ce", "--output-dir", str(target), "--plan-only"])
            plan = json.loads(out.getvalue())
            self.assertEqual(len(plan["policies"]), 8)
            self.assertNotIn("csm_st", plan["policies"])
            self.assertIn("local_global_team_u", plan["policies"])
            self.assertEqual(plan["stages"], ["support", "collect", "labels_team_u", "train_reward",
                                             "train_team_u", "evaluate", "report"])
            self.assertFalse(plan["ce_enabled"])
            self.assertNotIn("generator_model", plan["local_model_paths_exist"])
            self.assertNotIn("fast_downward", plan["missing_dependencies"])
            for key in ("expert_episodes", "ce_original_forwards_upper_bound", "ce_augmented_forwards_upper_bound"):
                self.assertEqual(plan["workload"]["pddl"][key], 0)
            self.assertEqual(plan["workload"]["pddl"]["test_schedules_upper_bound_shared_actions"], 540)
            self.assertFalse(target.exists())

    def test_collect_team_u_evaluate_report_and_resume_need_no_ce_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             mock.patch("causal_memory_control.csm_st_scoring.HuggingFaceCEScorer", side_effect=AssertionError("no CE")), \
             mock.patch.object(worker, "run_expert_task", side_effect=AssertionError("no expert")):
            root, config, runner, calls = fixture(directory)
            config["exclude_ce"] = True
            collect(config, "pddl", runner)
            rows, _ = team.measured_lists(config, "pddl")
            self.assertEqual(len(rows), 3)
            self.assertIn("team_u", rows[0])
            methods = learned_methods(config)
            self.assertEqual(methods, ("semantic_linear", "semantic_mlp", "semantic_st", "local_global_team_u"))
            for method in methods:
                save(root / "pddl" / method / "checkpoint.json", {"mock": method})
            gates = {m: SimpleNamespace(method=m, close=lambda: None) for m in methods}
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", runner), \
                 mock.patch.object(worker, "snapshot", return_value=root / "unused"), \
                 mock.patch.object(training, "learned_gates", return_value=gates), \
                 mock.patch.object(training, "select", return_value=(["bad"], [])):
                worker.evaluate(config, "pddl")
                summary = report(config)["pddl"]
                self.assertEqual(set(summary["policies"]), {m for m, _ in policies(config)} |
                                 {"random_drop_seed11", "random_drop_seed23", "random_drop_seed47"})
                self.assertNotIn("csm_st", summary["checkpoints_sha256"])
                self.assertFalse(summary["ce_enabled"])
                table = (root / "matrix_results.md").read_text()
                self.assertIn("Local–Global + Team U", table)
                self.assertNotIn("CE proxy", table)
                before = len(calls)
                worker.evaluate(config, "pddl")
                self.assertEqual(len(calls), before)
            self.assertFalse((root / "pddl/experts").exists())
            self.assertFalse((root / "pddl/csm_st").exists())

    def test_smoke_skips_expert_and_still_checks_environment_and_no_drop_parity(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root, config, runner, _ = fixture(directory)
            config.update(exclude_ce=True, max_steps=2)
            episode = SimpleNamespace(reset=lambda: None, close=lambda: None, resets=2)
            with mock.patch("causal_memory_control.interactive_envs.Episode", return_value=episode), \
                 mock.patch("causal_memory_control.interactive_runtime.NativeRunner", runner), \
                 mock.patch.object(worker, "run_expert_task", side_effect=AssertionError("expert must be skipped")):
                worker.smoke(config, "pddl")
            state = json.loads((root / "pddl/smoke/summary.json").read_text())
            self.assertEqual(state["status"], "passed")
            self.assertTrue(state["no_drop_parity"])
            self.assertIsNone(state["expert_success"])
            self.assertEqual(state["expert_check"], "skipped_ce_excluded")
            self.assertFalse((root / "pddl/smoke/expert.json").exists())

    def test_explicit_ce_stages_are_rejected_instead_of_accidentally_running(self):
        for stage in ("experts", "labels", "train_csm"):
            with self.subTest(stage=stage), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(["--exclude-ce", "--stage", stage])
        config = {"exclude_ce": True}
        for function in (worker.experts, training.ce_labels, training.csm_train):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(ValueError, "disabled"):
                function(config, "pddl")
        self.assertEqual(len(policies({})), 8)
        self.assertEqual(len(policies({"include_local_global_team_u": True})), 9)
        self.assertIn("experts", stages({}))


if __name__ == "__main__":
    unittest.main()
