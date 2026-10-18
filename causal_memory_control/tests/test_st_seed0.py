from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control import fever_experiment, train_gate, parallel_collection
from causal_memory_control import evaluation_replicas as replicas
from causal_memory_control.compare_gate_runs import compare
from causal_memory_control import run_seed0_gate_matrix as seed0
from causal_memory_control import run_st_gate_benchmark as st
from causal_memory_control.tests import test_fever_experiment as fever_tests
from causal_memory_control.tests.test_full_gate_benchmark import cache
from causal_memory_control.tests.test_seed0_gate_matrix import prepare_source
from causal_memory_control.tests.test_semantic_jev import TinyEncoder
from causal_memory_control.tests.test_st_gate import _STMemory, _FeverSTMacNet


def options(root, *, previous=False):
    argv = ["--source-root", str(root / "source"), "--output-dir", str(root / "st"),
        "--datasets", "fever", "--seed0-only", "--ensemble-size", "2",
        "--st-hidden-sizes", "8,4", "--st-epochs", "2"]
    if previous:
        argv += ["--seed0-results-root", str(root / "previous")]
    return argv


class STSeed0Tests(unittest.TestCase):
    def test_plan_only_single_source_original_test_boundary_and_no_writes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory).resolve()
            _, args = prepare_source(root, "fever")
            # An incomplete/missing seed1000 group must not be inspected.
            (args.diagnostic_results / "seed1000_retest" / "branches.jsonl").unlink()
            with mock.patch.object(st.base, "_run_command") as run:
                plan = st.main([*options(root), "--plan-only"])["fever"]
                run.assert_not_called()
            self.assertEqual(plan["collection_groups"], 1)
            self.assertEqual(plan["new_diagnostic_branches"], 0)
            self.assertEqual(plan["sample_seeds"], [0, 1])
            self.assertEqual(plan["training_sources"], [str(seed0.source_path(args))])
            self.assertEqual(plan["gate_training_questions"], 16)
            self.assertEqual(plan["final_test_questions"], 4)
            self.assertNotIn("diagnostic_branches_two_seeds", plan)
            self.assertFalse((root / "st").exists())

    def test_reject_partial_seed0_missing_baseline_and_unsafe_options_without_inference(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory).resolve()
            _, args = prepare_source(root, "fever")
            argv = options(root)
            with self.assertRaises(SystemExit):
                st.parse_args([*argv, "--prepare-missing"])
            with self.assertRaises(SystemExit):
                st.parse_args([v for v in options(root, previous=True) if v != "--seed0-only"])
            source = seed0.source_path(args)
            original = source.read_bytes()
            source.write_bytes(b"\n".join(original.splitlines()[:-1]) + b"\n")
            with mock.patch.object(st.base, "_run_command") as run:
                with self.assertRaisesRegex(ValueError, "incomplete seed0"):
                    st.main(argv)
                run.assert_not_called()
                source.write_bytes(original)
                with self.assertRaisesRegex(FileNotFoundError, "baseline"):
                    st.main(argv)
                run.assert_not_called()
            failure = st.base._load_json(root / "st" / "st_progress.json")
            self.assertEqual(failure["error"]["stage"], "fever:baseline")

    def test_existing_seed0_matrix_is_read_only_only_st_runs_and_resume_is_audited(self):
        self._existing_seed0_matrix(replica=False)

    def test_replica_uses_actual_new_endpoint_preserves_baseline_and_offline_completed_resume(self):
        self._existing_seed0_matrix(replica=True)

    def _existing_seed0_matrix(self, *, replica):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            _, args = prepare_source(root, "fever")
            for patch in fever_tests.FeverRunnerTests._patches():
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(fever_experiment, "GMemory", _STMemory))
            stack.enter_context(mock.patch.object(fever_experiment, "MacNet", _FeverSTMacNet))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=TinyEncoder()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            verify = stack.enter_context(mock.patch.object(parallel_collection, "verify_replicas", side_effect=lambda endpoints, model: [
                {"endpoint": endpoint, "model": model, "digest": "fixed-test-weights",
                 "ollama_version": "test-version", "model_settings_sha256": "fixed-test-settings"}
                for endpoint in endpoints]))
            commands = []
            interrupt_evaluation = [True]

            def execute(label, command, path, **_):
                commands.append((label, command))
                if command[3] == "causal_memory_control.train_gate":
                    train_gate.main(command[4:])
                elif command[3] == "causal_memory_control.fever_experiment":
                    if label.endswith(":evaluate_st") and interrupt_evaluation and interrupt_evaluation.pop():
                        raise RuntimeError("simulated ST evaluation interruption")
                    fever_experiment.main(command[4:])
                    if label.endswith(":baseline"):
                        cache(Path(command[command.index("--output-dir") + 1]) / "llm_cache.sqlite")
                else:
                    self.fail(f"unexpected collection or old-method invocation: {command}")

            stack.enter_context(mock.patch.object(st.base, "_run_command", side_effect=execute))
            seed0.main(["--source-root", str(root / "source"), "--output-dir", str(root / "previous"),
                "--datasets", "fever"])
            (args.diagnostic_results / "seed1000_retest" / "branches.jsonl").unlink()
            originals = {p: p.read_bytes() for folder in (root / "source", root / "previous")
                for p in folder.rglob("*") if p.is_file()}
            commands.clear()
            argv = options(root, previous=True)
            target = "http://127.0.0.1:11440/v1"
            if replica:
                argv += ["--evaluation-endpoint", target]
            with self.assertRaisesRegex(RuntimeError, "ST evaluation interruption"):
                st.main(argv)
            self.assertEqual([label for label, _ in commands], ["fever:train_st", "fever:evaluate_st"])
            training = commands[0][1]
            self.assertEqual(training[training.index("--input") + 1:training.index("--output")],
                [str(seed0.source_path(args))])
            evaluation = commands[-1][1]
            self.assertEqual(evaluation[evaluation.index("--endpoint") + 1], target if replica else args.endpoint)
            previous = root / "previous" / "fever"
            self.assertEqual(evaluation[evaluation.index("--cache-seed-from") + 1],
                str(previous / "native_gmemory" / "llm_cache.sqlite"))
            self.assertEqual(evaluation[evaluation.index("--evaluation-offset") + 1], "16")
            commands.clear()
            report = st.base._load_json(st.main(argv))
            self.assertEqual([label for label, _ in commands], ["fever:evaluate_st"])
            for arm in ("linear", "semantic_linear", "semantic_mlp"):
                self.assertTrue(report["details"]["fever"][arm]["reused"])
                self.assertEqual(report["details"]["fever"][arm]["path"], str(previous / arm))
            comparison = report["details"]["fever"]["semantic_st"]["result"]
            self.assertTrue(comparison["paired_control_valid"])
            self.assertTrue(comparison["policy_valid"])
            self.assertEqual(comparison["paired_tasks"], 4)
            self.assertEqual(report["data_protocol"]["fever"]["collection_groups"], 1)
            output = root / "st" / "fever" / "evaluation"
            if replica:
                proof_path = output.parent / "evaluation_replicas.json"
                proof = st.base._load_json(proof_path)
                self.assertEqual(proof["source_endpoint"], args.endpoint)
                self.assertEqual(proof["target_endpoint"], target)
                self.assertEqual(verify.call_count, 2)  # first run and unfinished resume
                self.assertEqual(comparison["evaluation_replica_provenance"]["manifest"], str(proof_path))
                actual = st.base._load_json(output / "run_manifest.json")
                self.assertEqual(actual["design"]["endpoint"], target)
                with self.assertRaisesRegex(ValueError, "run metadata differs"):
                    compare(previous / "native_gmemory" / "memory_gate.jsonl", output / "memory_gate.jsonl",
                        bootstrap_samples=10)
                verify.side_effect = RuntimeError("completed source service is offline")
            else:
                verify.assert_not_called()
            checkpoint = st.base._load_json(root / "st" / "fever" / "checkpoint.json")
            metadata = checkpoint["training_metadata"]
            self.assertEqual(metadata["training_source_sha256"].keys(), {str(seed0.source_path(args))})
            old = st.base._load_json(previous / "checkpoints" / "semantic_mlp.json")
            self.assertEqual(metadata["fit_task_ids"], old["training_metadata"]["fit_task_ids"])
            self.assertEqual(metadata["policy_calibration"]["calibration_task_ids"],
                old["training_metadata"]["policy_calibration"]["calibration_task_ids"])
            commands.clear()
            st.main(argv)
            self.assertFalse(commands)
            for path, data in originals.items():
                self.assertEqual(path.read_bytes(), data, str(path))
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                st.main([*argv, "--st-beta", "0.5"])
            # Changed source caches cannot silently change the paired baseline.
            with (previous / "native_gmemory" / "llm_cache.sqlite").open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "baseline artifact changed" if replica else "plan changed"):
                st.main(argv)

    def test_replica_mismatch_fails_before_training_and_plan_only_never_probes(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            adapter, args = prepare_source(root, "fever")
            for patch in fever_tests.FeverRunnerTests._patches():
                stack.enter_context(patch)
            stack.enter_context(mock.patch.object(fever_experiment, "GMemory", _STMemory))
            stack.enter_context(mock.patch.object(fever_experiment, "MacNet", _FeverSTMacNet))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            native = args.output_dir / "native_gmemory"
            fever_experiment.main(adapter.evaluation_command(args, mode="always_keep", output=native, resume=False)[4:])
            cache(native / "llm_cache.sqlite")
            argv = [*options(root), "--evaluation-endpoint", "http://127.0.0.1:11440/v1"]
            verify = stack.enter_context(mock.patch.object(parallel_collection, "verify_replicas",
                side_effect=ValueError("Ollama replicas differ in digest")))
            run = stack.enter_context(mock.patch.object(st.base, "_run_command"))
            plan = st.main([*argv, "--plan-only"])["fever"]
            self.assertEqual(plan["evaluation_endpoint"], "http://127.0.0.1:11440/v1")
            verify.assert_not_called()
            self.assertFalse((root / "st").exists())
            with self.assertRaisesRegex(ValueError, "digest"):
                st.main(argv)
            run.assert_not_called()
            failure = st.base._load_json(root / "st" / "st_progress.json")
            self.assertEqual(failure["error"]["stage"], "fever:verify_evaluation_replica")

    def test_endpoint_validation_and_evidence_never_mask_other_design_differences(self):
        for value in ("http://user:secret@host/v1", "http://host/invalid", "http://host/v1?key=secret", "file:///v1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replicas.validate_endpoint(value)
        proof = {"source_endpoint": "http://old/v1", "target_endpoint": "http://new/v1", "model": "test"}
        old = {"endpoint": "http://old/v1", "model": "test", "seed": 1}
        new = {"endpoint": "http://new/v1", "model": "test", "seed": 2}
        left, right = replicas.paired_metadata(old, new, proof)
        self.assertNotEqual(left, right)
        self.assertIn("endpoint", old)
        with self.assertRaisesRegex(ValueError, "disagree"):
            replicas.paired_metadata(old, {**new, "model": "other"}, proof)

    def test_replica_comparison_retains_seed_and_no_drop_checks_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            native = root / "native"
            gate = root / "gate.jsonl"
            source, target, model = "http://old/v1", "http://new/v1", "test-model"
            metadata = {"endpoint": source, "model": model, "seed": 1}
            baseline = {"task_id": "1", "run_metadata": metadata, "reward": 1, "done": True,
                "dropped_count": 0, "retrieval_fingerprint": "same", "outcome": {"prediction": "SUPPORTS"}}
            st.full._save(native / "run_manifest.json", {"design": metadata})
            log = native / "memory_gate.jsonl"
            log.write_text(json.dumps(baseline) + "\n")
            cache(native / "llm_cache.sqlite")
            probes = [{"endpoint": endpoint, "model": model, "digest": "same",
                "ollama_version": "same", "model_settings_sha256": "same"} for endpoint in (source, target)]
            with mock.patch.object(parallel_collection, "verify_replicas", return_value=probes):
                proof = replicas.build_manifest(native, source_endpoint=source, target_endpoint=target, model=model)
            path = root / "replicas.json"
            st.full._save(path, proof)
            changed = {**baseline, "run_metadata": {**metadata, "endpoint": target}}
            gate.write_text(json.dumps(changed) + "\n")
            self.assertTrue(compare(log, gate, bootstrap_samples=10, replica_manifest=path)["paired_control_valid"])
            gate.write_text(json.dumps({**changed, "run_metadata": {**changed["run_metadata"], "seed": 2}}) + "\n")
            with self.assertRaisesRegex(ValueError, "run metadata differs"):
                compare(log, gate, bootstrap_samples=10, replica_manifest=path)
            gate.write_text(json.dumps({**changed, "reward": 0, "done": False,
                "outcome": {"prediction": "REFUTES"}}) + "\n")
            result = compare(log, gate, bootstrap_samples=10, replica_manifest=path)
            self.assertFalse(result["paired_control_valid"])
            self.assertEqual(result["no_drop_mismatches"][0]["task_id"], "1")
            proof["replicas"][1]["digest"] = "different"
            st.full._save(path, proof)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                compare(log, gate, bootstrap_samples=10, replica_manifest=path)

    def test_reject_wrong_source_split_or_unregistered_baseline(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory).resolve()
            adapter, args = prepare_source(root, "fever")
            previous = root / "previous"
            manifest = {"schema": seed0.SCHEMA, "source_root": str(root / "source"),
                "experiments": {"fever": adapter.configuration(args)}}
            st.full._save(previous / "seed0_manifest.json", manifest)
            st.full._save(previous / "fever" / "baseline_selection.json", {"path": str(root / "unrelated")})
            with self.assertRaisesRegex(ValueError, "outside"):
                st.main([*options(root, previous=True), "--plan-only"])
            manifest["experiments"]["fever"]["repeats"] = 99
            st.full._save(previous / "seed0_manifest.json", manifest)
            with self.assertRaisesRegex(RuntimeError, "repeats"):
                st.main([*options(root, previous=True), "--plan-only"])


if __name__ == "__main__":
    unittest.main()
