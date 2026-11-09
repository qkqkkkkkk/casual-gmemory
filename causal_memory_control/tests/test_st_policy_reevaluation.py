"""Runtime rule changes reuse fitted predictions and matched measurements."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from causal_memory_control import reevaluate_st_policy as cli
from causal_memory_control import st_policy_evaluate as wrapper
from causal_memory_control.interactive_data import digest, save, sha
from causal_memory_control.interactive_runtime import ActionStore
from causal_memory_control.runtime_gate import GMemoryExposureGate
from causal_memory_control.st_policy import DATASET_RULES, PositiveSTGate, positive_decisions, resolve_rule
from causal_memory_control.types import PotentialOutcomePrediction


def prediction(value):
    return PotentialOutcomePrediction(.5 + value / 2, .5 - value / 2, value,
        "neutral", 20., "bootstrap_semantic_st_direct_u", True, 10)


def decisions(candidates, utilities):
    return [{"memory_id": c["memory_id"], "action": "KEEP", "prediction": asdict(prediction(u))}
            for c, u in zip(candidates, utilities)]


class Estimator:
    fitted = True

    def predict(self, features):
        return prediction(features["value"])


class Features:
    version = 2

    def build(self, event):
        return {"value": {"negative-a": -.3, "negative-b": -.1, "zero": 0., "positive": .2}[event.memory.content]}


def episode(task_id, candidates, drops=()):
    success = float(bool(drops))
    return {"task_id": task_id, "seed": 2000, "initial_fingerprint": "initial-" + task_id,
        "retrieval_fingerprint": "retrieval-" + task_id, "candidates": candidates,
        "drop_ids": sorted(drops), "query": "goal", "state": "observation", "task_config": {},
        "exposure_count": 4, "trajectory": [{"action": str(sorted(drops))}],
        "outcome": {"success": success, "progress": success, "steps": 1, "invalid_actions": 0}}


def pddl_fixture(directory, *, missing_multi=False):
    root = Path(directory) / "source"
    config = {"output_dir": str(root), "repo": str(Path(__file__).resolve().parents[2]),
        "evaluation_seed": 2000, "exclude_ce": True, "code_sha256": {},
        "experiments": {"pddl": {"tasks": [{"id": t, "group": t, "files": {}} for t in ("a", "b", "c")],
            "split": {"support": [], "fit": [], "dev": [], "test": ["a", "b", "c"]}}}}
    save(root / "benchmark_manifest.json", config)
    source = root / "pddl"
    (source / "support/g-memory").mkdir(parents=True)
    save(source / "snapshot.json", {"memory_dir": "support/g-memory", "files_sha256": {},
                                    "config_digest": digest(config)})
    for name in ("semantic_linear", "semantic_mlp", "semantic_st"):
        save(source / name / "checkpoint.json", {"schema": "gmemory-team-exposure-gate-v1",
            "estimator": {"schema": "st-direct-utility-estimator-v1"}})
    checkpoints = {name: sha(source / name / "checkpoint.json") for name in
                   ("semantic_linear", "semantic_mlp", "semantic_st")}
    store = ActionStore(source / "evaluation/actions", {"config_digest": digest(config),
        "snapshot_sha256": sha(source / "snapshot.json"), "checkpoints_sha256": checkpoints, "phase": "test"})
    for task_id, utilities in (("a", [-.3, -.1, .2]), ("b", []), ("c", [0.])):
        candidates = [{"memory_id": f"m{i}", "memory_type": "insight", "content": f"memory{i}",
                       "metadata": {"retrieval_index": i}} for i in range(len(utilities))]
        baseline = store.put(episode(task_id, candidates))
        save(store.root / "baselines" / (digest([task_id, 2000]) + ".json"), baseline)
        drops = [f"m{i}" for i, u in enumerate(utilities) if u <= 0]
        if drops and not (task_id == "a" and missing_multi):
            store.put(episode(task_id, candidates, drops))
        old = {"policy": "semantic_st", "task_id": task_id, "seed": 2000,
            "initial_fingerprint": baseline["initial_fingerprint"],
            "retrieval_fingerprint": baseline["retrieval_fingerprint"],
            "config_digest": digest(config), "retrieved_count": len(candidates), "dropped_count": 0,
            "drop_ids": [], "decisions": decisions(candidates, utilities),
            "outcome": baseline["outcome"], "action_keys": [baseline["action_key"]]}
        save(source / "evaluation/policy_parts" / (digest(task_id) + ".json"), [old])
    return root


def source_bytes(root):
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


class STPolicyTests(unittest.TestCase):
    def test_dataset_mapping_and_manual_overrides(self):
        for name, expected in (("fever", "calibrated"), ("hotpotqa", "positive_unlimited"),
                               ("pddl", "positive_unlimited")):
            with self.subTest(dataset=name):
                self.assertEqual(resolve_rule("dataset_default", name), expected)
                self.assertEqual(resolve_rule("calibrated", name), "calibrated")
                self.assertEqual(resolve_rule("positive_unlimited", name), "positive_unlimited")
        with self.assertRaisesRegex(ValueError, "no default"):
            resolve_rule("dataset_default", "alfworld")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            resolve_rule("unknown", "fever")

    def test_opt_in_changes_multiple_drops_and_zero_without_changing_legacy(self):
        kwargs = {"mode": "learned", "estimator": Estimator(), "feature_builder": Features()}
        legacy, optional = GMemoryExposureGate(**kwargs), PositiveSTGate(**kwargs)
        result = ([], [], ["negative-a", "negative-b", "zero", "positive"])
        for gate in (legacy, optional):
            gate.begin_task("a", {"task_main": "goal", "task_description": "state"})
        self.assertEqual(legacy.filter_retrieval(result, query_task="goal"), result)
        self.assertEqual(optional.filter_retrieval(result, query_task="goal"), ([], [], ["positive"]))
        self.assertEqual([d.action.name for d in optional._task_decisions], ["DROP", "DROP", "DROP", "KEEP"])
        self.assertEqual((legacy.kappa, legacy.max_drops), (1.96, 1))
        self.assertEqual((optional.kappa, optional.delta, optional.max_drops), (0., 0., None))

    def test_bad_or_untrained_saved_predictions_fail_instead_of_fabricating_scores(self):
        row = decisions([{"memory_id": "a"}], [-.1])[0]
        for field, value in (("utility", float("nan")), ("utility", True), ("calibrated", False),
                             ("source", "another_estimator")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                positive_decisions([{**row, "prediction": {**row["prediction"], field: value}}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            positive_decisions([row, row])
        self.assertEqual(positive_decisions([]), [])

    def test_wrapper_defaults_to_original_and_restores_binding_on_interrupt(self):
        runner = SimpleNamespace(GMemoryExposureGate=GMemoryExposureGate)
        used = []

        def execute(argv):
            used.append((runner.GMemoryExposureGate, argv))
            if "interrupt" in argv:
                raise KeyboardInterrupt()

        runner.main = execute
        with mock.patch.object(wrapper.importlib, "import_module", return_value=runner):
            wrapper.main(["--dataset", "fever"])
            with self.assertRaises(KeyboardInterrupt):
                wrapper.main(["--dataset", "fever", "--st-deletion-rule", "positive_unlimited", "interrupt"])
        self.assertIs(used[0][0], GMemoryExposureGate)
        self.assertIs(used[1][0], PositiveSTGate)
        self.assertEqual(used[1][1][-6:], ["--kappa", "0", "--delta", "0", "--max-drops", "-1"])
        self.assertIs(runner.GMemoryExposureGate, GMemoryExposureGate)

    def test_wrapper_dataset_defaults_and_explicit_overrides_do_not_leak_between_runs(self):
        runner = SimpleNamespace(GMemoryExposureGate=GMemoryExposureGate)
        observed = []
        runner.main = lambda argv: observed.append((runner.GMemoryExposureGate, argv))
        with mock.patch.object(wrapper.importlib, "import_module", return_value=runner):
            wrapper.main(["--dataset", "hotpotqa"])
            wrapper.main(["--dataset", "fever"])
            wrapper.main(["--dataset", "hotpotqa", "--st-deletion-rule", "calibrated"])
            wrapper.main(["--dataset", "fever", "--st-deletion-rule", "positive_unlimited"])
        self.assertEqual([cls for cls, _ in observed],
                         [PositiveSTGate, GMemoryExposureGate, GMemoryExposureGate, PositiveSTGate])
        self.assertEqual(observed[1][1], [])
        self.assertEqual(observed[2][1], [])
        self.assertIs(runner.GMemoryExposureGate, GMemoryExposureGate)

    def test_reference_audit_stops_changed_retrieval_before_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "old.jsonl"
            reference.write_text(json.dumps({"task_id": "a", "retrieval_fingerprint": "different", "decisions": []}) + "\n")
            runner = SimpleNamespace(GMemoryExposureGate=GMemoryExposureGate)

            def execute(argv):
                gate = runner.GMemoryExposureGate(mode="learned", estimator=Estimator(), feature_builder=Features())
                gate.begin_task("a", {})
                gate.filter_retrieval(([], [], []), query_task="goal")

            runner.main = execute
            with mock.patch.object(wrapper.importlib, "import_module", return_value=runner):
                with self.assertRaisesRegex(ValueError, "saved-score plan"):
                    wrapper.main(["--dataset", "hotpotqa", "--st-deletion-rule", "positive_unlimited",
                                  "--st-reference-log", str(reference)])
            self.assertIs(runner.GMemoryExposureGate, GMemoryExposureGate)


class STReevaluationTests(unittest.TestCase):
    def argv(self, root, output, rule="positive_unlimited"):
        return ["--datasets", "pddl", "--interactive-root", str(root), "--output-dir", str(output),
                "--st-deletion-rule", rule]

    def test_plan_then_recompute_from_old_actions_without_models_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = pddl_fixture(directory)
            output = Path(directory) / "new"
            before = source_bytes(root)
            argv = self.argv(root, output)
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", side_effect=AssertionError("model load")):
                plan = cli.main([*argv, "--plan-only"])
                self.assertEqual(plan["datasets"]["pddl"]["maximum_new_episodes"], 0)
                self.assertFalse(output.exists())
                result = cli.read(cli.main(argv))["datasets"]["pddl"]
                self.assertEqual(result["n"], 3)
                self.assertEqual(result["empty_candidate_tasks"], 1)
                self.assertAlmostEqual(result["semantic_st"], 2 / 3)
                self.assertEqual(result["old_st"], 0.)
                self.assertEqual(result["mean_drop_count"], 1.)
                cli.main(argv)
                with self.assertRaisesRegex(ValueError, "changed"):
                    cli.main(self.argv(root, output, "calibrated"))
            self.assertEqual(before, source_bytes(root))
            self.assertTrue((output / "results.md").is_file())
            self.assertEqual(cli.read(output / "progress.json")["status"], "completed")

    def test_dataset_default_uses_positive_pddl_and_explicit_old_rule_keeps_original_results(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = pddl_fixture(directory)
            output = Path(directory) / "old-rule-replay"
            argv = ["--datasets", "pddl", "--interactive-root", str(root), "--output-dir", str(output)]
            self.assertEqual(cli.parse_args(argv).st_deletion_rule, "dataset_default")
            result = cli.read(cli.main(argv))["datasets"]["pddl"]
            self.assertEqual(result["runtime_deletion_rule"], "positive_unlimited")
            self.assertAlmostEqual(result["semantic_st"], 2 / 3)
            plan = cli.read(output / "manifest.json")
            self.assertEqual(plan["policy"]["datasets"]["pddl"]["rule"], "positive_unlimited")
            with self.assertRaisesRegex(ValueError, "changed"):
                cli.main([*argv, "--st-deletion-rule", "calibrated"])
            old_output = Path(directory) / "explicit-old-rule"
            result = cli.read(cli.main(self.argv(root, old_output, "calibrated")))["datasets"]["pddl"]
            self.assertEqual((result["semantic_st"], result["old_st"], result["mean_drop_count"]), (0., 0., 0.))
            self.assertEqual(result["runtime_deletion_rule"], "calibrated")

    def test_mixed_plan_resolves_each_dataset_without_mutating_global_selection(self):
        suite = cli.parse_args(["--output-dir", "unused"])
        def qa_plan(suite, name, identity):
            return {"plan": {"deletion_rule": resolve_rule(suite.st_deletion_rule, name)}}
        def pddl_plan(suite, identity):
            return {"plan": {"deletion_rule": resolve_rule(suite.st_deletion_rule, "pddl")}}
        with mock.patch.object(cli, "prepare_qa", side_effect=qa_plan), \
             mock.patch.object(cli, "prepare_pddl", side_effect=pddl_plan):
            _, manifest = cli.prepare(suite)
            effective = {name: plan["deletion_rule"] for name, plan in manifest["datasets"].items()}
            self.assertEqual(effective, DATASET_RULES)
            self.assertEqual({name: spec["rule"] for name, spec in manifest["policy"]["datasets"].items()}, DATASET_RULES)
            self.assertEqual(suite.st_deletion_rule, "dataset_default")
            self.assertEqual(manifest["new_training_runs"], 0)

    def test_new_combination_runs_once_and_interruption_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = pddl_fixture(directory, missing_multi=True)
            output = Path(directory) / "new"
            calls, interrupted = [], [True]

            class Runner:
                def __init__(self, *args, **kwargs):
                    pass

                def run(self, task, seed, drops, expected):
                    calls.append((task["id"], tuple(drops)))
                    if interrupted[0]:
                        interrupted[0] = False
                        raise KeyboardInterrupt()
                    return episode(task["id"], expected["candidates"], drops)

                def close(self):
                    pass

            before = source_bytes(root)
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", Runner), \
                 mock.patch("causal_memory_control.interactive_worker.verify_inputs") as verify:
                argv = self.argv(root, output)
                with self.assertRaises(KeyboardInterrupt):
                    cli.main(argv)
                self.assertEqual(cli.read(output / "progress.json")["status"], "interrupted")
                cli.main(argv)
                cli.main(argv)
                self.assertEqual(calls, [("a", ("m0", "m1")), ("a", ("m0", "m1"))])
                self.assertEqual(verify.call_count, 2)
            self.assertEqual(before, source_bytes(root))

    def test_incomplete_or_tampered_inputs_and_nested_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = pddl_fixture(directory)
            with self.assertRaisesRegex(ValueError, "non-nested"):
                cli.main(self.argv(root, root / "new") + ["--plan-only"])
            output = Path(directory) / "new"
            argv = self.argv(root, output)
            cli.main(argv)
            path = root / "pddl/evaluation/policy_parts" / (digest("a") + ".json")
            records = cli.read(path)
            records[0]["decisions"][0]["prediction"]["utility"] = -.2
            save(path, records)
            with self.assertRaisesRegex(ValueError, "changed"):
                cli.main(argv)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                cli.main(argv)

    def test_sqlite_cache_copy_preserves_source_and_committed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "old.sqlite", Path(directory) / "new.sqlite"
            with sqlite3.connect(source) as connection:
                connection.execute("CREATE TABLE answers (key TEXT)")
                connection.execute("INSERT INTO answers VALUES ('old-answer')")
            before = source.read_bytes()
            cli.copy_cache(source, target)
            with sqlite3.connect(target) as connection:
                self.assertEqual(connection.execute("SELECT key FROM answers").fetchone()[0], "old-answer")
                connection.execute("INSERT INTO answers VALUES ('new-answer')")
            cli.copy_cache(source, target)
            self.assertEqual(source.read_bytes(), before)
            with sqlite3.connect(target) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0], 2)

    def test_qa_reuses_original_checkpoint_cache_seed_and_partition_for_each_dataset(self):
        from causal_memory_control.run_hotpotqa_gate_comparison import _evaluation_command

        for dataset, requested_rule in (("fever", "positive_unlimited"), ("hotpotqa", "positive_unlimited"),
                                        ("fever", "dataset_default"), ("hotpotqa", "dataset_default")):
            with self.subTest(dataset=dataset, rule=requested_rule), tempfile.TemporaryDirectory() as directory, \
                 redirect_stdout(io.StringIO()):
                rule = resolve_rule(requested_rule, dataset)
                root = Path(directory).resolve()
                qa, source, trained, output = (root / n for n in ("qa", "full", "trained", "new"))
                save(qa / "reevaluation_manifest.json", {"trained_root": str(trained)})
                save(trained / "csm_st_manifest.json", {"source_root": str(source)})
                save(source / "benchmark_manifest.json", {})
                memory = source / "artifacts" / dataset / "g-memory"
                save(memory / "causal_snapshot_manifest.json", {"evaluation_ids": [0, 1, 2]})
                checkpoint = trained / "st" / "checkpoint.json"
                save(checkpoint, {"schema": "gmemory-team-exposure-gate-v1",
                    "estimator": {"schema": "st-direct-utility-estimator-v1"}})
                args = SimpleNamespace(memory_dir=memory, training_claims=1, evaluation_count=3, final_claims=2,
                                       data=root / "data.json", api_key=None)
                old = qa / dataset / "semantic_st"
                base = qa / dataset / "native_gmemory"
                settings = {"endpoint": "http://127.0.0.1:11442/v1", "model": "saved-model", "seed": 2111,
                    "temperature": .7, "node_num": 3, "graph_type": "Chain", "embedding_model": "saved-encoder",
                    "successful_topk": 3, "failed_topk": 0, "insights_topk": 3, "threshold": 0.}
                run = {**settings, "evaluation_ids": [1, 2], "evaluation_offset": 1,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
                    "kappa": 1.96, "delta": .1, "max_drops": 0}
                manifest = {"design": {"task_state_policy": "isolated-task-state-v1"}, "run": run}
                save(old / "run_manifest.json", manifest)
                save(base / "run_manifest.json", manifest)
                values = []
                for task_id in (1, 2):
                    candidates = [{"memory_id": f"m{i}"} for i in range(3)] if task_id == 1 else []
                    row = {"task_id": task_id, "run_metadata": {"benchmark": "FEVER_binary_offline"
                        if dataset == "fever" else "HotpotQA_distractor_offline"},
                        "retrieval_fingerprint": f"retrieval-{task_id}", "decisions": decisions(candidates, [-.3, 0, .2]),
                        "reward": float(task_id == 2), "done": task_id == 2, "dropped_count": 0, "keep_rate": 1.,
                        "outcome": {"prediction": str(task_id == 2)}}
                    for d in row["decisions"]:
                        d["action"] = "keep"  # QA logs use lowercase, unlike interactive policy records.
                    values.append(row)
                for path in (old / "memory_gate.jsonl", base / "memory_gate.jsonl"):
                    path.write_text("".join(json.dumps(r) + "\n" for r in values))
                with sqlite3.connect(old / "llm_cache.sqlite") as connection:
                    connection.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, response TEXT)")

                class Adapter:
                    def validate_checkpoint(self, *call_args):
                        pass

                    def evaluation_command(self, child, **kwargs):
                        return _evaluation_command(child, runner_module=f"causal_memory_control.{dataset}_experiment", **kwargs)

                    def evaluation_complete(self, child, target, **kwargs):
                        return (target / "summary.json").exists()

                commands = []

                def execute(command, log, **kwargs):
                    commands.append(command)
                    self.assertEqual(command[3], "causal_memory_control.st_policy_evaluate")
                    for flag, expected in (("--checkpoint", str(checkpoint)),
                            ("--cache-seed-from", str(old / "llm_cache.sqlite")), ("--seed", "2111"),
                            ("--max-drops", "-1" if rule == "positive_unlimited" else "0"),
                            ("--kappa", "0.0" if rule == "positive_unlimited" else "1.96"),
                            ("--delta", "0.0" if rule == "positive_unlimited" else "0.1"),
                            ("--st-deletion-rule", rule), ("--evaluation-offset", "1"),
                            ("--endpoint", settings["endpoint"])):
                        self.assertEqual(command[command.index(flag) + 1], expected)
                    destination = Path(command[command.index("--output-dir") + 1])
                    destination.mkdir(parents=True)
                    generated = deepcopy(values)
                    for row in generated:
                        if rule == "positive_unlimited":
                            row["decisions"] = positive_decisions(row["decisions"])
                            row["dropped_count"] = sum(d["action"] == "DROP" for d in row["decisions"])
                            row.update(reward=1., done=True)
                            row["outcome"]["prediction"] = "True"
                    (destination / "memory_gate.jsonl").write_text("".join(json.dumps(r) + "\n" for r in generated))
                    save(destination / "summary.json", {})
                    save(destination / "run_manifest.json", {})

                argv = ["--datasets", dataset, "--qa-root", str(qa), "--output-dir", str(output),
                        "--st-deletion-rule", requested_rule]
                before = {**source_bytes(qa), **source_bytes(source), **source_bytes(trained)}
                with mock.patch("causal_memory_control.source_run.import_source_dataset",
                                side_effect=lambda *a: (Adapter(), deepcopy(args))), \
                     mock.patch.object(cli, "run_logged", side_effect=execute):
                    plan = cli.main([*argv, "--plan-only"])
                    self.assertFalse(output.exists())
                    self.assertEqual(plan["datasets"][dataset]["changed_drop_tasks"], int(rule == "positive_unlimited"))
                    self.assertEqual(plan["datasets"][dataset]["deletion_rule"], rule)
                    self.assertEqual(commands, [])
                    summary = cli.read(cli.main(argv))["datasets"][dataset]
                    self.assertEqual(summary["semantic_st"], 1. if rule == "positive_unlimited" else .5)
                    self.assertEqual(summary["runtime_deletion_rule"], rule)
                    self.assertEqual(summary["old_st"], .5)
                    self.assertTrue(summary["comparison"]["paired_control_valid"])
                    cli.main(argv)
                    self.assertEqual(len(commands), 1)
                self.assertEqual(before, {**source_bytes(qa), **source_bytes(source), **source_bytes(trained)})


if __name__ == "__main__":
    unittest.main()
