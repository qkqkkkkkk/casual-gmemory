from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from transformers import BertConfig, BertModel

from causal_memory_control import csm_st_worker, csm_st_evaluate, fever_experiment, hotpotqa_experiment
from causal_memory_control import run_csm_st_benchmark as csm
from causal_memory_control.csm_st_model import LocalGlobalCSM
from causal_memory_control.tests.test_csm_st import FakeCE, tiny_components
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_seed0_gate_matrix import prepare_source
from causal_memory_control.tests.test_st_subset import previous_subset
from causal_memory_control.tests.test_semantic_jev import TinyEncoder
from causal_memory_control.tests.test_st_gate import _STMemory, _FeverSTMacNet
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests


def options(root, dataset):
    return ["--source-root", str(root / "source"), "--output-dir", str(root / "csm"),
        "--datasets", dataset, "--generator-model", "fake-offline", "--encoder-model", "tiny-offline",
        "--max-pair-tokens", "32", "--epochs", "1", "--batch-tasks", "4", "--trivial-keep-fraction", "1",
        "--ce-device", "cpu", "--ce-dtype", "float32", "--train-device", "cpu", "--acknowledge-ce-proxy"]


class CSMBenchmarkTests(unittest.TestCase):
    def test_plan_only_full_and_sharded_subset_no_writes_no_downloads(self):
        for dataset, subset in (("fever", False), ("hotpotqa", True)):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    if subset:
                        _, _, recorded = previous_subset(root, dataset, parallel=True)
                    else:
                        prepare_source(root, dataset)
                    argv = options(root, dataset)
                    if subset:
                        argv += [f"--{dataset}-results-root", str(root / "previous")]
                    with mock.patch.object(csm.reuse.base, "_run_command") as execute:
                        plan = csm.main([*argv, "--plan-only"])[dataset]
                    execute.assert_not_called()
                self.assertEqual(plan["final_test_offset"], 16)
                self.assertEqual(plan["final_test_tasks"], 4)
                self.assertEqual(plan["diagnostic_tasks"], 10 if subset else 16)
                self.assertEqual(len(plan["sources"]), 2 if subset else 1)
                self.assertEqual(plan["other_methods_run"], [])
                self.assertFalse((root / "csm").exists())

    def test_both_hosts_new_pipeline_resume_and_old_sources_untouched(self):
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
                native = args.output_dir / "native_gmemory"
                runner.main(adapter.evaluation_command(args, mode="always_keep", output=native, resume=False)[4:])
                cache(native / "llm_cache.sqlite")
                before = {p: p.read_bytes() for p in (root / "source").rglob("*") if p.is_file()}
                _, tokenizer, _ = tiny_components(root)
                commands = []

                def execute(label, command, path, **kwargs):
                    commands.append((label, command))
                    if command[3] == "causal_memory_control.csm_st_worker":
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
                        self.fail(f"old method/collector was invoked: {command}")

                stack.enter_context(mock.patch.object(csm.reuse.base, "_run_command", side_effect=execute))
                report = json.loads(csm.main(options(root, dataset)).read_text())
                self.assertEqual([label for label, _ in commands], [f"{dataset}:labels", f"{dataset}:train", f"{dataset}:evaluate"])
                result = report["details"][dataset]["csm_st"]["result"]
                self.assertTrue(result["paired_control_valid"])
                self.assertEqual(result["paired_tasks"], 4)
                checkpoint = json.loads((root / "csm" / dataset / "checkpoint.json").read_text())
                self.assertEqual(checkpoint["training_metadata"]["source_diagnostic_rewards_used"], False)
                self.assertEqual(checkpoint["csm_st"]["config"]["global_layers"], 3)
                self.assertTrue((root / "csm" / "csm_st_results.csv").is_file())
                commands.clear()
                csm.main(options(root, dataset))
                self.assertEqual(commands, [])
                for path, contents in before.items():
                    self.assertEqual(path.read_bytes(), contents, str(path))
                with self.assertRaisesRegex(ValueError, "configuration.*changed"):
                    csm.main([*options(root, dataset), "--beta", "0.01"])

    def test_missing_baseline_and_old_output_collision_fail_before_model_work(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory).resolve()
            prepare_source(root, "fever")
            with mock.patch.object(csm.reuse.base, "_run_command") as execute:
                with self.assertRaisesRegex(FileNotFoundError, "baseline required"):
                    csm.main(options(root, "fever"))
                execute.assert_not_called()
            self.assertFalse((root / "csm").exists())
            with self.assertRaises(SystemExit):
                csm.parse_args([*options(root, "fever"), "--output-dir", str(root / "source" / "new")])
            with self.assertRaises(SystemExit):
                csm.parse_args([v for v in options(root, "fever") if v != "--acknowledge-ce-proxy"])


if __name__ == "__main__":
    unittest.main()
