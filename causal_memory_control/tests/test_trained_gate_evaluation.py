from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from transformers import BertConfig, BertModel

from causal_memory_control import csm_st_evaluate, csm_st_worker, train_gate
from causal_memory_control import fever_experiment, hotpotqa_experiment
from causal_memory_control import run_csm_st_benchmark as csm
from causal_memory_control import run_seed0_gate_matrix as seed0
from causal_memory_control import run_st_gate_benchmark as st
from causal_memory_control import run_trained_gate_evaluation as reevaluate
from causal_memory_control import run_oracle_evaluation as oracle
from causal_memory_control.csm_st_model import LocalGlobalCSM
from causal_memory_control.tests.test_csm_st import FakeCE, tiny_components
from causal_memory_control.tests.test_csm_st_benchmark import options
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_seed0_gate_matrix import prepare_source
from causal_memory_control.tests.test_semantic_jev import TinyEncoder
from causal_memory_control.tests.test_st_gate import _STMemory, _FeverSTMacNet
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests


class TrainedEvaluationTests(unittest.TestCase):
    def test_explicit_old_st_options_are_exclusive_and_calibration_is_required(self):
        for dataset in ("fever", "hotpotqa"):
            with self.subTest(dataset=dataset), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    reevaluate.parse_args(["--trained-root", "trained", "--output-dir", "out",
                        f"--{dataset}-old-st-root", "old", f"--{dataset}-old-st-checkpoint", "direct.json"])
        args = SimpleNamespace(checkpoint=Path("raw/checkpoint.json"))
        policy = {"enabled": True, "recommended_kappa": 1.96, "delta": 0.2, "recommended_max_drops": 0}
        reevaluate.calibrated_st_policy(args, {"training_metadata": {"policy_calibration": policy}})
        self.assertEqual((args.kappa, args.delta, args.max_drops), (1.96, 0.2, 0))
        for update in ({"enabled": False}, {"recommended_kappa": None}, {"recommended_kappa": float("nan")},
                       {"delta": -0.1}, {"recommended_max_drops": 2}, {"recommended_max_drops": True}):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, "saved calibrated"):
                reevaluate.calibrated_st_policy(args, {"training_metadata": {"policy_calibration": {**policy, **update}}})

    def test_both_datasets_fresh_paired_evaluation_resume_no_training_or_old_cache(self):
        for dataset in ("fever", "hotpotqa"):
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
                stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder",
                    return_value=TinyEncoder()))
                _, tokenizer, _ = tiny_components(root)
                evaluation_only = [False]
                commands = []
                interrupt = [False]

                def execute(label, command, path, **kwargs):
                    commands.append((label, command))
                    if evaluation_only[0]:
                        self.assertNotIn("causal_memory_control.csm_st_worker", command)
                        self.assertNotIn("causal_memory_control.train_gate", command)
                        if interrupt[0] and label == dataset + ":linear":
                            interrupt[0] = False
                            raise KeyboardInterrupt()
                    if command[3] == "causal_memory_control.train_gate":
                        train_gate.main(command[4:])
                    elif command[3] == "causal_memory_control.csm_st_worker":
                        if label.endswith(":labels"):
                            with mock.patch("causal_memory_control.csm_st_scoring.HuggingFaceCEScorer", return_value=FakeCE()):
                                csm_st_worker.main(command[4:])
                        else:
                            def network(config, **kwargs):
                                encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=8,
                                    num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                                    max_position_embeddings=64))
                                return LocalGlobalCSM(config, encoder)
                            with mock.patch("causal_memory_control.csm_st_model.LocalGlobalCSM", side_effect=network), \
                                 mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                                csm_st_worker.main(command[4:])
                    elif command[3] == "causal_memory_control.csm_st_evaluate":
                        csm_st_evaluate.main(command[4:])
                    else:
                        runner.main(command[4:])
                    if "--mode" in command and command[command.index("--mode") + 1] == "always_keep":
                        destination = Path(command[command.index("--output-dir") + 1]) / "llm_cache.sqlite"
                        if not destination.exists():
                            cache(destination)

                stack.enter_context(mock.patch.object(seed0.base, "_run_command", side_effect=execute))
                seed0.main(["--source-root", str(root / "source"), "--output-dir", str(root / "old"), "--datasets", dataset])
                csm.main([*options(root, dataset), f"--{dataset}-results-root", str(root / "old")])
                st_suite = st.parse_args(["--source-root", str(root / "source"), "--output-dir", str(root / "old_st"),
                    "--datasets", dataset, "--seed0-only", "--seed0-results-root", str(root / "old"),
                    "--seed", "23", "--st-beta", "0.01", "--ensemble-size", "2", "--st-epochs", "1",
                    "--st-hidden-sizes", "8,4"])
                _, st_args = st.prepare_dataset(st_suite, dataset)
                if dataset == "fever":
                    st_args.checkpoint = root / "ablation" / "beta_0.01" / "checkpoint.json"
                train_gate.main(st.train_command(st_suite, adapter, st_args)[4:])
                saved_policy = json.loads(st_args.checkpoint.read_text())["training_metadata"]["policy_calibration"]
                original = {p: p.read_bytes() for folder in ("source", "old", "csm", "old_st", "ablation")
                            for p in (root / folder).rglob("*") if p.is_file()}
                evaluation_only[0] = True
                commands.clear()
                argv = ["--trained-root", str(root / "csm"), "--output-dir", str(root / "retest"),
                        "--datasets", dataset, "--csm-device", "cpu", "--seed", "2111"]
                plan = reevaluate.main([*argv, "--plan-only"])[dataset]
                self.assertFalse((root / "retest").exists())
                self.assertEqual(commands, [])
                self.assertEqual(plan["test_tasks"], 4)
                self.assertEqual(plan["test_offset"], 16)
                self.assertEqual(plan["unavailable"], ["semantic_st"])
                self.assertEqual(plan["new_training_runs"], 0)
                interrupt[0] = True
                with self.assertRaises(KeyboardInterrupt):
                    reevaluate.main(argv)
                self.assertEqual(json.loads((root / "retest" / "reevaluation_progress.json").read_text())["status"], "interrupted")
                commands.clear()
                result = json.loads(reevaluate.main(argv).read_text())
                self.assertEqual([label for label, _ in commands], [f"{dataset}:{m}"
                    for m in ("linear", "semantic_linear", "semantic_mlp", "csm_st")])
                new_baseline = root / "retest" / dataset / "native_gmemory"
                baseline_run = json.loads((new_baseline / "run_manifest.json").read_text())["run"]
                self.assertIsNone(baseline_run["cache_seed_from"])
                self.assertEqual(baseline_run["seed"], 2111)
                for method, comparison in result["details"][dataset].items():
                    if method == "native_gmemory":
                        continue
                    self.assertEqual(comparison["paired_tasks"], 4)
                    self.assertTrue(comparison["paired_control_valid"])
                    run = json.loads((root / "retest" / dataset / method / "run_manifest.json").read_text())["run"]
                    self.assertEqual(run["cache_seed_from"], str(new_baseline / "llm_cache.sqlite"))
                global_run = json.loads((root / "retest" / dataset / "always_global_drop" / "summary.json").read_text())
                self.assertEqual(global_run["dropped_candidates"], global_run["retrieved_candidates"])
                commands.clear()
                reevaluate.main(argv)
                self.assertEqual(commands, [])
                for path, value in original.items():
                    self.assertEqual(path.read_bytes(), value, str(path))
                with self.assertRaisesRegex(ValueError, "configuration/inputs changed"):
                    reevaluate.main([*argv, "--seed", "2112"])
                commands.clear()
                with_st = [*argv, "--output-dir", str(root / "retest_with_st"),
                           f"--{dataset}-old-st-checkpoint", str(st_args.checkpoint)]
                plan = reevaluate.main([*with_st, "--plan-only"])[dataset]
                self.assertEqual(plan["unavailable"], [])
                self.assertEqual(plan["checkpoints"]["semantic_st"], str(st_args.checkpoint))
                self.assertEqual(plan["old_st_settings"], {"beta": 0.01, "training_seed": 23,
                    "kappa": saved_policy["recommended_kappa"], "delta": saved_policy["delta"],
                    "max_drops": saved_policy["recommended_max_drops"]})
                self.assertEqual(commands, [])
                self.assertFalse((root / "retest_with_st").exists())
                result = json.loads(reevaluate.main(with_st).read_text())
                self.assertEqual([label for label, _ in commands], [f"{dataset}:{m}"
                    for m, _ in reevaluate.POLICIES])
                self.assertTrue(result["details"][dataset]["semantic_st"]["paired_control_valid"])
                run = json.loads((root / "retest_with_st" / dataset / "semantic_st" / "run_manifest.json").read_text())["run"]
                self.assertEqual((run["kappa"], run["delta"], run["max_drops"]),
                    (saved_policy["recommended_kappa"], saved_policy["delta"], saved_policy["recommended_max_drops"]))
                commands.clear()
                reevaluate.main(with_st)
                self.assertEqual(commands, [])
                oracle_argv = ["--evaluation-root", str(root / "retest_with_st"),
                               "--output-dir", str(root / "latest_oracle"), "--datasets", dataset]
                oracle_report = json.loads(oracle.main(oracle_argv).read_text())
                self.assertEqual(len(commands), 2)
                self.assertTrue(all("oracle_branches" in command[command.index("--output-dir") + 1]
                                    for _, command in commands))
                self.assertTrue(oracle_report["details"][dataset][oracle.ORACLE]["paired_control_valid"])
                self.assertTrue(all(method in oracle_report["details"][dataset] for method, _ in reevaluate.POLICIES))
                commands.clear()
                oracle.main(oracle_argv)
                self.assertEqual(commands, [])
                if dataset == "hotpotqa":
                    root_plan = reevaluate.main([*argv, "--output-dir", str(root / "root_plan"),
                        "--hotpotqa-old-st-root", str(root / "old_st"), "--plan-only"])[dataset]
                    self.assertEqual(root_plan["unavailable"], [])
                with self.assertRaisesRegex(ValueError, "expected an old Semantic ST"):
                    reevaluate.main([*argv, "--output-dir", str(root / "bad_st"), f"--{dataset}-old-st-checkpoint",
                        str(root / "old" / dataset / "checkpoints" / "semantic_mlp.json"), "--plan-only"])
                self.assertFalse((root / "bad_st").exists())
                for path, value in original.items():
                    self.assertEqual(path.read_bytes(), value, str(path))
                checkpoint = root / "csm" / dataset / "model" / "weights.pt"
                checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "CSM artifact changed"):
                    reevaluate.main([*argv, "--output-dir", str(root / "bad")])
                self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
