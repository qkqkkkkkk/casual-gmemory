from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from causal_memory_control.interactive_data import (digest, inventory, locked, make_split,
    pddl_cross_validation, pddl_inventory, save, validate_split, workload)
from causal_memory_control.interactive_envs import Episode, make_expert, outcome, planner_inputs
from causal_memory_control.interactive_runtime import ActionStore, audit, event_from_row
from causal_memory_control.interactive_training import label_parts, label_plan, plan_messages
from causal_memory_control import interactive_worker as worker
from causal_memory_control.interactive_report import report
from causal_memory_control import run_interactive_gate_benchmark as cli

REPO = Path(__file__).resolve().parents[2]


def task(i, dataset="pddl"):
    return {"id": str(i), "group": f"problem:{i}", "dataset": dataset,
            "domain": "blockworld", "difficulty": "easy", "config": {}}


def measured(task_id="3", seed=2000, drops=()):
    candidates = [{"memory_id": s, "memory_type": "insight", "content": s,
                   "metadata": {"retrieval_index": i}} for i, s in enumerate(("bad", "good", "other"))]
    success = float("bad" in drops)
    return {"schema": "interactive-episode-v1", "task_id": task_id, "seed": seed,
        "query": "public goal", "state": "public observation", "task_config": {"game_name": "blockworld"},
        "initial_fingerprint": "initial-" + task_id, "retrieval_fingerprint": "retrieval-" + task_id,
        "candidates": candidates, "exposure_count": 4, "drop_ids": list(drops),
        "outcome": outcome(dataset="pddl", raw_reward=success, done=bool(success), won=bool(success), steps=2, max_steps=2),
        "trajectory": [{"action": "a", "observation": "b", "won": bool(success)}],
        "llm_calls": 1, "cache_hits": 0}


class DataTests(unittest.TestCase):
    def test_real_inventory_and_selected_scene_groups(self):
        data = inventory(REPO, ["pddl", "alfworld"])
        self.assertEqual({k: len(v) for k, v in data["pddl"]["split"].items()},
                         {"support": 8, "fit": 28, "dev": 12, "test": 12})
        self.assertEqual({k: len(v) for k, v in data["alfworld"]["split"].items()},
                         {"support": 50, "fit": 80, "dev": 20, "test": 134})
        for name, values in data.items():
            validate_split(values["tasks"], values["split"])
        jobs = workload(data)
        self.assertEqual(sum(v["diagnostic_schedules_upper_bound_shared_keep"] for v in jobs.values()), 1960)
        self.assertEqual(sum(v["test_schedules_upper_bound_shared_actions"] for v in jobs.values()), 1314)

    def test_annotation_domain_identity_is_not_legacy_row_position(self):
        rows, _ = pddl_inventory(REPO)
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["pddl:barman:p01"]["annotation_id"], 30)
        self.assertEqual(by_id["pddl:gripper:prob01"]["annotation_id"], 0)
        self.assertEqual(by_id["pddl:blockworld:problem10"]["config"]["problem_index"], 1)
        self.assertEqual(by_id["pddl:blockworld:problem10"]["annotation_id"], 21)

    def test_group_leakage_and_duplicate_ids_fail(self):
        rows = [task(i) for i in range(4)]
        split = dict(zip(("support", "fit", "dev", "test"), (["0"], ["1"], ["2"], ["3"])))
        validate_split(rows, split)
        rows[3]["group"] = rows[1]["group"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split(rows, split)

    def test_plan_only_does_not_initialize_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "never-created"
            with mock.patch("causal_memory_control.interactive_runtime.model_identity", side_effect=AssertionError("service")), \
                 mock.patch.object(worker, "snapshot", side_effect=AssertionError("runtime")), redirect_stdout(io.StringIO()) as stream:
                cli.main(["--datasets", "pddl", "--output-dir", str(output), "--plan-only"])
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(stream.getvalue())["writes_or_model_loads_or_service_calls"], 0)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            cli.parse_args(["--datasets", "fever"])

    def test_configuration_change_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            locked(path, {"seed": 1})
            locked(path, {"seed": 1})
            with self.assertRaisesRegex(ValueError, "changed"):
                locked(path, {"seed": 2})

    def test_large_alf_split_reserves_extra_test_scenes(self):
        data = inventory(REPO, ["alfworld"], alf_counts=(100, 800, 200), alf_heldout=500)["alfworld"]
        self.assertEqual({k: len(v) for k, v in data["split"].items()},
                         {"support": 100, "fit": 800, "dev": 200, "test": 634})
        self.assertEqual(len(data["primary_test_ids"]), 134)
        self.assertEqual(len(data["test_cohorts"]["train_scene_holdout"]), 500)
        validate_split(data["tasks"], data["split"])
        ids = {r["id"]: r for r in data["tasks"]}
        self.assertTrue(all(ids[t]["official_split"] == "train" for t in data["test_cohorts"]["train_scene_holdout"]))
        self.assertTrue(all(ids[t]["official_split"] != "train" for t in data["primary_test_ids"]))

    def test_pddl_five_folds_have_disjoint_outer_tests_and_inner_partitions(self):
        rows, _ = pddl_inventory(REPO)
        folds = pddl_cross_validation(rows)
        self.assertEqual(len(folds), 5)
        tests = []
        for split in folds:
            validate_split(rows, split)
            self.assertEqual({k: len(v) for k, v in split.items()},
                             {"support": 8, "fit": 32, "dev": 8, "test": 12})
            tests += split["test"]
        self.assertEqual(len(set(tests)), 60)
        self.assertEqual(set(tests), {r["id"] for r in rows})
        self.assertEqual(folds, pddl_cross_validation(rows))
        self.assertNotEqual(folds, pddl_cross_validation(rows, seed=72))

    def test_cv_job_manifests_and_costs_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            data = inventory(REPO, ["pddl"], pddl_folds=5)
            config = {"output_dir": directory, "experiments": data}
            jobs = cli.dataset_jobs(config, "pddl")
            self.assertEqual(len({j["output_dir"] for _, j in jobs}), 5)
            for index, (_, job) in enumerate(jobs):
                self.assertNotIn("cv_splits", job["experiments"]["pddl"])
                self.assertEqual(job["cv_context"]["fold"], index)
                self.assertEqual(job["cv_context"]["parent_config_digest"], digest(config))
            costs = workload(data)["pddl"]
            self.assertEqual(costs["diagnostic_schedules_upper_bound_shared_keep"], 2800)
            self.assertEqual(costs["test_schedules_upper_bound_shared_actions"], 540)
            self.assertEqual(costs["per_fold_tasks"]["fit"], 32)
            self.assertEqual(costs["unique_registered_tasks"], 60)



class MetricAndCEtests(unittest.TestCase):
    def test_planner_constant_dedup_preserves_initial_facts_and_rejects_type_change(self):
        domain = "(define (domain x) (:constants wrench jack - tool))"
        problem = "(define (problem p) (:objects wrench jack - tool boot - container) (:init (in jack boot)))"
        d, p, meta = planner_inputs(domain, problem)
        self.assertEqual(d, domain)
        self.assertEqual(meta["removed_duplicate_objects"], ["jack", "wrench"])
        self.assertIn("(in jack boot)", p)
        self.assertIn("boot - container", p)
        self.assertNotIn("wrench jack - tool", p)
        with self.assertRaisesRegex(ValueError, "types disagree"):
            planner_inputs(domain, problem.replace("wrench jack - tool", "wrench jack - wheel"))

    def test_terminal_failure_timeout_and_success_are_distinct(self):
        def result(done, won, steps):
            return outcome(dataset="alfworld", raw_reward=0, done=done, won=won, steps=steps, max_steps=3)
        self.assertEqual(result(True, False, 1)["success"], 0)
        self.assertEqual(result(True, False, 1)["termination_reason"], "failed_terminal")
        self.assertEqual(result(False, False, 3)["termination_reason"], "step_limit")
        self.assertEqual(result(True, True, 2)["success"], 1)

    def test_ce_sign_one_all_score_per_list_and_no_fake_gold(self):
        class Scorer:
            calls = 0
            def score(self, messages, answers):
                self.calls += 1
                self.assert_answers = answers
                text = messages[-1]["content"]
                return {"ce": 3 + 2 * ("bad-memory" in text) - ("good-memory" in text)}
        row = {"id": "1", "query": "goal", "state": "observation", "expert_validated": True,
            "answers": ["pickup b1\nstack b1 b2"], "candidates": [
                {"text": "bad-memory"}, {"text": "good-memory"}]}
        scorer = Scorer()
        labelled = label_plan(row, scorer)
        self.assertEqual(labelled["ci"], [-2, 1])
        self.assertEqual(scorer.calls, 3)
        self.assertNotIn(row["answers"][0], json.dumps(plan_messages(row, row["candidates"])))
        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory) / "parts"
            label_parts([row], parts, scorer)
            count = scorer.calls
            label_parts([row], parts, scorer)
            self.assertEqual(scorer.calls, count)
            changed = {**row, "answers": ["different"]}
            with self.assertRaisesRegex(ValueError, "changed"):
                label_parts([changed], parts, scorer)
        for changed in ({**row, "answers": [True]}, {**row, "expert_validated": False}, {**row, "answers": [""]}):
            with self.assertRaisesRegex(ValueError, "validated"):
                label_plan(changed, scorer)

    def test_runtime_event_never_contains_expert_or_outcome(self):
        row = measured()
        row["answers"] = ["hidden expert"]
        event = event_from_row(row, 0)
        text = str(event)
        self.assertNotIn("hidden expert", text)
        self.assertNotIn("termination_reason", text)
        self.assertEqual(event.query, row["query"])

    def test_action_store_resume_retrieval_parity_and_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ActionStore(directory, {"snapshot": "frozen", "seed": 2000})
            baseline = store.put(measured())
            branch = store.put(measured(drops=["bad"]))
            self.assertEqual(store.get("3", 2000, baseline, ["bad"]), branch)
            broken = measured()
            broken["outcome"]["won"] = True
            with self.assertRaisesRegex(ValueError, "parity"):
                audit(broken, baseline, [])
            broken = deepcopy(branch)
            broken["candidates"][0]["content"] = "changed"
            with self.assertRaisesRegex(ValueError, "changed"):
                audit(broken, baseline, ["bad"])
            conflict = measured(drops=["bad"])
            conflict["outcome"]["success"] = 0
            with self.assertRaisesRegex(ValueError, "conflicting"):
                store.put(conflict)
            with self.assertRaisesRegex(ValueError, "changed"):
                ActionStore(directory, {"snapshot": "different"})


class PipelineTests(unittest.TestCase):
    def test_cv_report_aggregates_exactly_once_and_audits_child_provenance(self):
        from causal_memory_control.interactive_report import POLICIES
        from causal_memory_control.csm_st_data import write_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"output_dir": str(root), "experiments": inventory(REPO, ["pddl"], pddl_folds=5),
                      "evaluation_seed": 2000, "random_seeds": [11, 23, 47]}
            methods = [m for m, _ in POLICIES] + ["random_drop_seed11", "random_drop_seed23", "random_drop_seed47"]
            for _, child in cli.dataset_jobs(config, "pddl"):
                child_root = Path(child["output_dir"])
                save(child_root / "benchmark_manifest.json", child)
                save(child_root / "pddl/snapshot.json", {"fold": child["cv_context"]["fold"]})
                for method in ("semantic_linear", "semantic_mlp", "semantic_st", "csm_st"):
                    save(child_root / "pddl" / method / "checkpoint.json", {"mock": method})
                records = []
                for task_id in child["experiments"]["pddl"]["split"]["test"]:
                    baseline = measured(task_id)
                    success = child["cv_context"]["fold"] % 2
                    baseline["outcome"] = outcome(dataset="pddl", raw_reward=success,
                        done=bool(success), won=bool(success), steps=1, max_steps=2)
                    baseline["action_key"] = "action-" + task_id
                    records += [worker.policy_record(child, m, baseline, baseline, []) for m in methods]
                write_rows(child_root / "pddl/evaluation/policies.jsonl", records)
            summary = report(config)["pddl"]
            self.assertEqual(summary["evaluation_design"], "five_fold_out_of_fold")
            self.assertEqual(summary["policies"]["always_use"]["n"], 60)
            self.assertAlmostEqual(summary["policies"]["always_use"]["success"], .4)
            self.assertEqual(len(summary["policies"]["always_use"]["outer_fold_success"]), 5)
            self.assertTrue((root / "pddl/cv_sources.json").exists())
            child_path = root / "pddl_folds/fold_00/benchmark_manifest.json"
            child = json.loads(child_path.read_text())
            child["cv_context"]["test_outcomes_used_for_tuning"] = True
            save(child_path, child)
            with self.assertRaisesRegex(ValueError, "manifest"):
                report(config)

    def test_report_keeps_extra_test_out_of_primary_unseen_score(self):
        from causal_memory_control.interactive_report import POLICIES
        from causal_memory_control.csm_st_data import write_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{**task(i, "alfworld"), "task_type": "put"} for i in range(5)]
            data = {"tasks": rows, "protocol": "extended-synthetic-test",
                "split": {"support": ["0"], "fit": ["1"], "dev": ["2"], "test": ["3", "4"]},
                "primary_test_ids": ["3"], "test_cohorts": {"valid_unseen": ["3"], "train_scene_holdout": ["4"]}}
            config = {"output_dir": str(root), "experiments": {"alfworld": data},
                      "evaluation_seed": 2000, "random_seeds": [11, 23, 47]}
            save(root / "alfworld/snapshot.json", {"mock": True})
            for method in ("semantic_linear", "semantic_mlp", "semantic_st", "csm_st"):
                save(root / "alfworld" / method / "checkpoint.json", {"mock": method})
            records = []
            for task_id in ("3", "4"):
                baseline = measured(task_id)
                baseline["outcome"] = outcome(dataset="alfworld", raw_reward=int(task_id == "4"),
                    done=task_id == "4", won=task_id == "4", steps=1, max_steps=2)
                baseline["action_key"] = "action-" + task_id
                for method in [m for m, _ in POLICIES] + ["random_drop_seed11", "random_drop_seed23", "random_drop_seed47"]:
                    records.append(worker.policy_record(config, method, baseline, baseline, []))
            write_rows(root / "alfworld/evaluation/policies.jsonl", records)
            stats = report(config)["alfworld"]["policies"]["always_use"]
            self.assertEqual(stats["n"], 1)
            self.assertEqual(stats["success"], 0)
            self.assertEqual(stats["cohort_results"]["train_scene_holdout"]["success"], 1)
            self.assertIn("train_scene_holdout", (root / "matrix_results.csv").read_text())

    def test_read_only_qa_import_rejects_old_state_and_keeps_missing_methods(self):
        from causal_memory_control.interactive_prior_results import import_qa
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "reevaluation_manifest.json", {"schema": "trained-gates-evaluation-v1",
                "datasets": {"fever": {"test_tasks": 1}}})
            run_path = root / "fever/native_gmemory/run_manifest.json"
            save(run_path, {"design": {"task_state_policy": "isolated-task-state-v1", "evaluation_ids": [3]}})
            row = {"task_id": 3, "reward": 1, "done": True, "retrieval_fingerprint": "fp",
                   "retrieved_count": 1, "dropped_count": 0, "keep_rate": 1, "outcome": {},
                   "run_metadata": {"seed": 2000}, "decisions": []}
            log = root / "fever/native_gmemory/memory_gate.jsonl"
            log.write_text(json.dumps(row) + "\n")
            before = log.read_bytes()
            imported = import_qa(root)
            self.assertEqual(imported["fever"]["policies"]["always_use"]["primary"], 1)
            self.assertNotIn("random_drop", imported["fever"]["policies"])
            self.assertEqual(log.read_bytes(), before)
            save(run_path, {"design": {"evaluation_ids": [3]}})
            with self.assertRaisesRegex(ValueError, "isolation"):
                import_qa(root)

    def test_support_transaction_recovery_preserves_committed_snapshot(self):
        calls = []
        class Runner:
            def __init__(self, config, root, memory_dir, **kwargs):
                self.memory = Path(memory_dir)
                self.memory.mkdir(parents=True, exist_ok=True)
            def run(self, t, seed):
                calls.append(t["id"])
                (self.memory / (t["id"] + ".txt")).write_text("success")
                return measured(t["id"], seed)
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            data = {"tasks": [task(i) for i in range(4)],
                "split": {"support": ["0", "1"], "fit": ["2"], "dev": ["3"], "test": []}}
            config = {"output_dir": directory, "experiments": {"pddl": data}}
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", Runner):
                worker.support(config, "pddl")
                frozen = worker.snapshot(config, "pddl")
                self.assertTrue((frozen / "0.txt").exists())
                self.assertTrue((frozen / "1.txt").exists())
                worker.support(config, "pddl")
                self.assertEqual(calls, ["0", "1"])
                (frozen / "0.txt").write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    worker.snapshot(config, "pddl")

    def test_native_macnet_adapter_with_fake_llm_and_memory(self):
        from causal_memory_control.interactive_envs import quiet_native_imports
        quiet_native_imports()
        from mas.memory.mas_memory.memory_base import MASMemoryBase
        from causal_memory_control.interactive_runtime import NativeRunner

        class Memory(MASMemoryBase):
            def retrieve_memory(self, **kwargs):
                return self.apply_retrieval_gate(([], [], ["sample memory"]),
                    query_task=kwargs["query_task"], task_state=self.current_task_context.task_description)
        class Chat:
            def __init__(self, *args, **kwargs):
                self.calls = self.cache_hits = 0
            def __call__(self, *args, **kwargs):
                self.calls += 1
                return "check valid actions"
            def close(self):
                pass
        rows, _ = pddl_inventory(REPO)
        selected = next(r for r in rows if r["id"] == "pddl:blockworld:problem1")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             mock.patch("causal_diagnostic.oracle_recipient.seeded_client.SeededCachedChat", Chat), \
             mock.patch("mas.memory.mas_memory.GMemory.GMemory", Memory), \
             mock.patch("mas.utils.EmbeddingFunc", return_value=None):
            config = {"repo": str(REPO), "model": "mock", "endpoint": "http://invalid.test/v1", "max_steps": 2,
                      "few_shots": 0, "successful_topk": 0, "insights_topk": 1, "temperature": .7,
                      "embedding_model": "never-load"}
            runner = NativeRunner(config, directory, Path(directory) / "unused")
            try:
                baseline = runner.run(selected, 2000)
                self.assertEqual(baseline["outcome"]["steps"], 2)
                self.assertEqual(baseline["exposure_count"], 4)
                self.assertIn("state_chain", baseline["message_graph"])
                replay = runner.run(selected, 2000, expected=baseline)
                audit(replay, baseline, [])
                self.assertEqual(len(replay["reset_fingerprints"]), 2)
            finally:
                runner.close()

    def test_collection_eight_policies_oracle_and_resume(self):
        calls = []
        class Runner:
            def __init__(self, *args, **kwargs):
                pass
            def run(self, task, seed, drops=(), expected=None):
                calls.append((task["id"], seed, tuple(drops)))
                row = measured(task["id"], seed, drops)
                if expected:
                    audit(row, expected, drops)
                return row
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {"tasks": [task(i) for i in range(4)], "protocol": "synthetic-test",
                "split": dict(zip(("support", "fit", "dev", "test"), (["0"], ["1"], ["2"], ["3"])))}
            config = {"output_dir": str(root), "experiments": {"pddl": data},
                "evaluation_seed": 2000, "diagnostic_seed": 0, "repeats": 2, "random_seeds": [11, 23, 47]}
            save(root / "pddl/snapshot.json", {"mock": True})
            for method in ("semantic_linear", "semantic_mlp", "semantic_st", "csm_st"):
                save(root / "pddl" / method / "checkpoint.json", {"mock": method})
            gates = {m: SimpleNamespace(method=m, close=lambda: None) for m in
                     ("semantic_linear", "semantic_mlp", "semantic_st", "csm_st")}
            def selection(gate, baseline):
                return (([] if gate.method == "semantic_linear" else ["bad", "other"] if gate.method == "csm_st"
                         else ["bad"]), [{"mock": gate.method}])
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", Runner), \
                 mock.patch.object(worker, "snapshot", return_value=root / "memory"), \
                 mock.patch("causal_memory_control.interactive_training.learned_gates", return_value=gates), \
                 mock.patch("causal_memory_control.interactive_training.select", side_effect=selection):
                worker.collect(config, "pddl")
                before = len(calls)
                self.assertEqual(before, 2 * 2 * 4)  # two tasks, two repeats, KEEP+3 SINGLE DROP
                worker.collect(config, "pddl")
                self.assertEqual(len(calls), before)
                worker.evaluate(config, "pddl")
                self.assertEqual(len(calls) - before, 6)  # KEEP+3 SINGLE+ALL+new CSM subset
                scores = report(config)
                self.assertEqual(len(scores["pddl"]["policies"]), 11)  # 8 main rows + 3 Random seeds
                self.assertEqual(scores["pddl"]["policies"]["oracle"]["success"], 1)
                self.assertEqual(scores["pddl"]["policies"]["global_drop"]["success"], 1)
                before = len(calls)
                worker.evaluate(config, "pddl")
                self.assertEqual(len(calls), before)
                self.assertTrue((root / "matrix_results.md").exists())
                path = root / "pddl/evaluation/policies.jsonl"
                rows = path.read_text().splitlines()
                path.write_text("\n".join(rows[:-1]) + "\n")
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    report(config)

    def test_real_pddl_reset_and_verified_symbolic_plan(self):
        rows, _ = pddl_inventory(REPO)
        row = next(r for r in rows if r["id"] == "pddl:blockworld:problem1")
        with redirect_stdout(io.StringIO()):
            episode = Episode(row, REPO, 30)
            try:
                episode.step("pickup b1")
                episode.reset()
                self.assertEqual(len(set(episode.resets)), 1)
                self.assertEqual(episode.steps, 0)
            finally:
                episode.close()
            plan = make_expert(row, REPO)
        self.assertEqual(plan["validation"]["success"], 1)
        self.assertEqual(plan["validation"]["invalid_actions"], 0)
        self.assertGreater(len(plan["actions"]), 0)

    def test_barman_expert_hyphen_actions_replay_without_nltk(self):
        rows, _ = pddl_inventory(REPO)
        row = next(r for r in rows if r["id"] == "pddl:barman:p01")
        with redirect_stdout(io.StringIO()), mock.patch("nltk.word_tokenize", side_effect=AssertionError("NLTK is not required")):
            plan = make_expert(row, REPO)
        self.assertEqual(plan["validation"]["success"], 1)
        self.assertEqual(plan["validation"]["invalid_actions"], 0)
        self.assertTrue(any("fill-shot" in action for action in plan["actions"]))

    def test_tyreworld_expert_replays_after_duplicate_constant_normalization(self):
        rows, _ = pddl_inventory(REPO)
        row = next(r for r in rows if r["id"] == "pddl:tyreworld:p01")
        with redirect_stdout(io.StringIO()):
            plan = make_expert(row, REPO)
        self.assertEqual(plan["validation"]["success"], 1)
        self.assertEqual(plan["validation"]["invalid_actions"], 0)
        self.assertEqual(plan["planner_input_normalization"]["removed_duplicate_objects"], ["jack", "pump", "wrench"])


if __name__ == "__main__":
    unittest.main()
