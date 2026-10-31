"""Read-only support inspection exposes actions without rewriting results."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from causal_memory_control import inspect_interactive_support as tool


def fixture(directory):
    root = Path(directory)
    for index, task_id in enumerate(("pddl:gripper:prob03", "pddl:barman:p08")):
        attempt = root / "support/attempts" / f"{index:04d}-fixture"
        attempt.mkdir(parents=True)
        row = {"task_id": task_id, "state": "Initial observation with goal",
            "outcome": {"success": 0.0, "steps": 3, "invalid_actions": 2},
            "trajectory": [
                {"step": 1, "action": "## Action Plan", "valid_action": False, "observation": "invalid"},
                {"step": 2, "action": "## Action Plan", "valid_action": False, "observation": "invalid"},
                {"step": 3, "action": "check valid actions", "valid_action": True,
                 "observation": "Valid actions are: move rooma roomb"}],
            "agent_transcript": ["prompt text", "Act 3: check valid actions"]}
        episode = attempt / "episode.json"
        episode.write_text(json.dumps(row))
        marker = root / "support/committed" / f"{index:04d}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"task_id": task_id,
            "directory": str((attempt / "committed").relative_to(root)),
            "episode_sha256": hashlib.sha256(episode.read_bytes()).hexdigest()}))
    return root


def contents(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class InspectSupportTests(unittest.TestCase):
    def test_exact_task_filter_prints_all_actions_and_changes_no_result_bytes(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as out:
            root = fixture(directory)
            before = contents(root)
            tool.main(["--root", str(root), "--task-id", "pddl:gripper:prob03",
                       "--steps", "0", "--transcript-tail", "1"])
            text = out.getvalue()
            self.assertIn("committed_support=2 successful_support=0", text)
            self.assertIn("invalid_actions=4 invalid_rate=66.7%", text)
            self.assertIn("=== pddl:gripper:prob03 ===", text)
            self.assertNotIn("=== pddl:barman:p08 ===", text)
            self.assertIn("Valid actions are: move rooma roomb", text)
            self.assertIn('["## Action Plan", 2]', text)
            self.assertIn("Act 3: check valid actions", text)
            self.assertEqual(contents(root), before)

    def test_steps_limit_summary_mode_missing_task_and_negative_limit(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as out:
            root = fixture(directory)
            tool.inspect(root, steps=1)
            self.assertIn("2 more steps", out.getvalue())
            out.seek(0)
            out.truncate(0)
            tool.inspect(root, summary_only=True)
            self.assertNotIn("recorded_environment_steps", out.getvalue())
            with self.assertRaisesRegex(ValueError, "no committed support episode"):
                tool.inspect(root, task_ids=["unknown"])
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                tool.inspect(root, steps=-1)

    def test_tampered_episode_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture(directory)
            episode = root / "support/attempts/0000-fixture/episode.json"
            episode.write_text(episode.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                tool.episodes(root)
            marker = root / "support/committed/0000.json"
            commit = json.loads(marker.read_text())
            commit["directory"] = "../outside/committed"
            marker.write_text(json.dumps(commit))
            with self.assertRaisesRegex(ValueError, "invalid committed directory"):
                tool.episodes(root)


if __name__ == "__main__":
    unittest.main()
