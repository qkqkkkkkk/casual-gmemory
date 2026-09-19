from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from causal_memory_control import run_fever_gate_comparison as comparison


class _QuietProgress:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total")
        self.initial = kwargs.get("initial", 0)

    def set_postfix_str(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None

    def close(self):
        return None


class FeverGateComparisonTests(unittest.TestCase):
    def _args(self, root: Path, *extra: str):
        data = root / "fever.jsonl"
        data.write_text("{}\n", encoding="utf-8")
        return comparison.parse_args(
            [
                "--data",
                str(data),
                "--memory-dir",
                str(root / "memory"),
                "--diagnostic-results",
                str(root / "diagnostic"),
                "--checkpoint",
                str(root / "gate.json"),
                "--output-dir",
                str(root / "output"),
                *extra,
            ]
        )

    def test_default_design_reserves_sixty_claims_for_final_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            self.assertEqual(
                comparison._configuration(args)["final_evaluation_claims"],
                60,
            )

    def test_native_and_learned_commands_form_a_resumable_paired_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root, "--api-key", "must-not-be-logged")
            native = comparison._evaluation_command(
                args,
                mode="always_keep",
                output=args.output_dir / "native_gmemory",
                resume=False,
            )
            learned = comparison._evaluation_command(
                args,
                mode="learned",
                output=args.output_dir / "learned_gate",
                resume=True,
            )

            self.assertEqual(native[native.index("--mode") + 1], "always_keep")
            self.assertEqual(native[native.index("--claims") + 1], "60")
            self.assertNotIn("must-not-be-logged", native)
            self.assertEqual(learned[learned.index("--mode") + 1], "learned")
            self.assertEqual(
                learned[learned.index("--checkpoint") + 1],
                str(args.checkpoint),
            )
            self.assertEqual(
                learned[learned.index("--cache-seed-from") + 1],
                str(args.output_dir / "native_gmemory" / "llm_cache.sqlite"),
            )
            self.assertIn("--resume", learned)

    def test_evaluation_resume_rejects_ambiguous_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            output.mkdir()
            self.assertFalse(comparison._evaluation_resume(output))
            (output / "run_manifest.json.tmp").write_text("partial", encoding="utf-8")
            self.assertFalse(comparison._evaluation_resume(output))
            (output / "unexpected.txt").touch()
            with self.assertRaisesRegex(RuntimeError, "run_manifest.json is missing"):
                comparison._evaluation_resume(output)

    def test_pipeline_manifest_rejects_changed_rerun_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            comparison._prepare_manifest(args)
            changed = self._args(root, "--model", "different-model")
            with self.assertRaisesRegex(SystemExit, "different arguments"):
                comparison._prepare_manifest(changed)

    def test_stage_failure_persists_exact_cause_and_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            argv = [
                "--data",
                str(args.data),
                "--memory-dir",
                str(args.memory_dir),
                "--diagnostic-results",
                str(args.diagnostic_results),
                "--checkpoint",
                str(args.checkpoint),
                "--output-dir",
                str(args.output_dir),
            ]
            incomplete = {stage: False for stage in comparison.STAGES}

            def fail(stage, command, log_path, *, api_key=None):
                raise comparison.StageFailure(
                    stage,
                    7,
                    command,
                    log_path,
                    "specific model-server failure",
                )

            with mock.patch.object(comparison, "_status", return_value=incomplete), mock.patch.object(
                comparison, "_run_command", side_effect=fail
            ), mock.patch.object(comparison, "tqdm", _QuietProgress):
                with self.assertRaises(comparison.StageFailure):
                    comparison.main(argv)

            progress = json.loads(
                (args.output_dir / "pipeline_progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["error"]["stage"], "diagnostic")
            self.assertEqual(progress["error"]["exit_code"], 7)
            self.assertIn("specific model-server failure", progress["error"]["log_tail"])
            self.assertIn("StageFailure", progress["error"]["traceback"])
            self.assertIn("run_all", progress["error"]["command_shell"])


if __name__ == "__main__":
    unittest.main()
