from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from causal_memory_control import run_full_gate_benchmark as full
from causal_memory_control import fever_experiment, hotpotqa_experiment
from causal_memory_control.planned_drop import PlannedDropGate, build_plan, indexed_rows, load_plan
from causal_memory_control.runtime_gate import GMemoryExposureGate
from causal_memory_control.dataset_adapters import HotpotQADatasetAdapter
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_tests
from causal_memory_control.tests import test_fever_experiment as fever_tests


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def base_rows(benchmark="FEVER_binary_offline"):
    result = []
    for i, reward in enumerate((0.0, 1.0)):
        result.append({"schema": "test", "task_id": i, "task": {"claim": f"claim {i}"},
            "mode": "always_keep", "run_metadata": {"benchmark": benchmark},
            "retrieval_fingerprint": f"fingerprint-{i}", "retrieved_count": 2,
            "dropped_count": 0, "kept_count": 2, "keep_rate": 1.0,
            "reward": reward, "done": bool(reward),
            "outcome": {"prediction": str(reward), "answer_f1": reward, "answer_em": reward},
            "decisions": [
                {"memory_id": "trajectory-1", "memory_type": "trajectory", "retrieval_rank": 1,
                 "memory_content": "example", "action": "keep"},
                {"memory_id": "insight-1", "memory_type": "insight", "retrieval_rank": 1,
                 "memory_content": "rule", "action": "keep"}]})
    return result


def apply_plan(rows, plan):
    result = copy.deepcopy(rows)
    for row in result:
        drops = plan["tasks"][str(row["task_id"])]["drop_memory_ids"]
        row["mode"] = "planned_drop"
        for decision in row["decisions"]:
            decision["action"] = "drop" if decision["memory_id"] in drops else "keep"
        row.update(dropped_count=len(drops), kept_count=2-len(drops), keep_rate=1-len(drops)/2)
        if drops:
            # Removing an insight helps task 0; removing a trajectory harms task 1.
            reward = float("insight-1" in drops) if row["task_id"] == 0 else float("trajectory-1" not in drops)
            row.update(reward=reward, done=bool(reward))
            row["outcome"].update(prediction=str(reward), answer_f1=reward, answer_em=reward)
    return result


def cache(path, values=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        db.executemany("INSERT INTO responses VALUES (?,?)", (values or {"shared": "answer"}).items())


class InterventionPlanTests(unittest.TestCase):
    def test_all_top1_and_random_are_different_and_outcome_free(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"baseline.jsonl"
            rows = base_rows()
            rows[0]["task"]["gold_answer"] = "SECRET_GOLD"
            save_rows(path, rows)
            all_plan = build_plan(path, policy="global_all")
            self.assertEqual(all_plan["tasks"]["0"]["drop_memory_ids"], ["trajectory-1", "insight-1"])
            top = build_plan(path, policy="global_top1")
            self.assertEqual(top["tasks"]["0"]["drop_memory_ids"], ["trajectory-1"])
            random_plan = build_plan(path, policy="random_one", seed=9)
            self.assertNotIn("SECRET_GOLD", json.dumps(random_plan))
            self.assertTrue(all(len(task["drop_memory_ids"]) == 1 for task in random_plan["tasks"].values()))
            for row in rows:
                row["reward"] = 1-row["reward"]
            save_rows(path, list(reversed(rows)))
            again = build_plan(path, policy="random_one", seed=9)
            self.assertEqual(random_plan["tasks"], again["tasks"])

    def test_random_matches_zero_one_reference_without_reading_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_rows(root/"baseline", base_rows())
            reference = base_rows()
            reference[1]["dropped_count"] = 1
            save_rows(root/"reference", reference)
            plan = build_plan(root/"baseline", policy="random_one", reference_path=root/"reference")
            self.assertEqual(len(plan["tasks"]["0"]["drop_memory_ids"]), 0)
            self.assertEqual(len(plan["tasks"]["1"]["drop_memory_ids"]), 1)
            reference[0]["retrieval_fingerprint"] = "changed"
            save_rows(root/"reference", reference)
            with self.assertRaisesRegex(ValueError, "retrieval mismatch"):
                build_plan(root/"baseline", policy="random_one", reference_path=root/"reference")

    def test_runtime_drops_all_and_checks_original_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = GMemoryExposureGate(mode="always_keep")
            control.begin_task(1, {})
            retrieval = ([], [], ["one", "two"])
            control.filter_retrieval(retrieval, query_task="q", task_state="state")
            row = control.end_task(0, False)
            save_rows(root/"baseline", [row])
            full._save(root/"plan", build_plan(root/"baseline", policy="global_all"))
            gate = PlannedDropGate(root/"plan")
            gate.begin_task(1, {})
            self.assertEqual(gate.filter_retrieval(retrieval, query_task="q", task_state="state"), ([], [], []))
            result = gate.end_task(0, False)
            self.assertEqual(result["dropped_count"], 2)
            self.assertEqual(result["mode"], "planned_drop")
            gate.begin_task(1, {})
            with self.assertRaisesRegex(ValueError, "retrieval changed"):
                gate.filter_retrieval(([], [], ["changed"]), query_task="q", task_state="state")

    def test_plan_validation_rejects_unknown_ids_and_outcome_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_rows(root/"baseline", base_rows())
            plan = build_plan(root/"baseline", policy="candidate_index", index=0)
            plan["tasks"]["0"]["drop_memory_ids"] = ["unknown"]
            full._save(root/"plan", plan)
            with self.assertRaisesRegex(ValueError, "invalid intervention candidates"):
                load_plan(root/"plan")
            plan["selection_uses_outcomes"] = True
            full._save(root/"plan", plan)
            with self.assertRaisesRegex(ValueError, "outcome-selected"):
                load_plan(root/"plan")


class EvaluatorIntegrationTests(unittest.TestCase):
    def test_hotpot_planned_drop_logs_and_resumes_and_rejects_plan_change(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data, memory = hotpot_tests.HotpotRunnerTests.fixture(root)
            hotpot_tests.HotpotRunnerTests.patches(stack)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            hotpot_tests._HotpotMacNet.fail_id = None
            baseline, output = root/"base", root/"drop"
            hotpotqa_experiment.main(hotpot_tests.HotpotRunnerTests.argv(data, memory, baseline))
            plan = root/"plan.json"
            full._save(plan, build_plan(baseline/"memory_gate.jsonl", policy="global_all"))
            argv = [*hotpot_tests.HotpotRunnerTests.argv(data, memory, output), "--mode", "planned_drop", "--intervention-plan", str(plan)]
            hotpotqa_experiment.main(argv)
            rows = indexed_rows(output/"memory_gate.jsonl")
            self.assertTrue(all(row["dropped_count"] == 2 for row in rows.values()))
            shared = full.base.parse_args([])
            shared.data, shared.memory_dir, shared.output_dir = data, memory, root
            shared.training_claims, shared.evaluation_count, shared.final_claims = 1, 3, 2
            shared.model, shared.embedding_model, shared.node_num = "fake-model", "fake-embedding", 2
            shared.endpoint = "http://unused.invalid/v1"
            command = [sys.executable, "-u", "-m", "causal_memory_control.hotpotqa_experiment", *argv]
            self.assertTrue(full.planned_ready(HotpotQADatasetAdapter(), shared, output, command, plan))
            before = hotpot_tests._HotpotMacNet.schedule_count
            hotpotqa_experiment.main([*argv, "--resume"])
            self.assertEqual(hotpot_tests._HotpotMacNet.schedule_count, before)
            payload = json.loads(plan.read_text())
            payload["tasks"]["2"]["drop_memory_ids"] = []
            full._save(plan, payload)
            with self.assertRaisesRegex(SystemExit, "resume arguments"):
                hotpotqa_experiment.main([*argv, "--resume"])

    def test_fever_planned_drop_uses_same_evaluator_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            fixture = fever_tests.FeverRunnerTests()
            data, memory = fixture._fixture(root)
            stack.enter_context(mock.patch.object(fever_experiment, "_load_runtime_components", lambda: None))
            for patch in fixture._patches():
                stack.enter_context(patch)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            common = ["--data",str(data),"--memory-dir",str(memory),"--claims","2",
                      "--model","fake-model","--node-num","2","--embedding-model","fake-embedding",
                      "--endpoint","http://unused.invalid/v1","--candidate-kinds","trajectory,insight"]
            fever_experiment.main([*common,"--output-dir",str(root/"base")])
            plan = root/"plan.json"
            full._save(plan, build_plan(root/"base/memory_gate.jsonl", policy="global_all"))
            fever_experiment.main([*common,"--output-dir",str(root/"drop"),"--mode","planned_drop",
                                   "--intervention-plan",str(plan)])
            self.assertTrue(all(row["dropped_count"] == 1 for row in indexed_rows(root/"drop/memory_gate.jsonl").values()))


class CacheAndOracleTests(unittest.TestCase):
    def test_interrupted_cache_publish_recovers_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root/"source.sqlite", root/"union.sqlite"
            cache(source)
            original = Path.replace
            def fail_publish(path, target):
                if path.name == "union.building.sqlite":
                    raise KeyboardInterrupt("interrupted before cache publish")
                return original(path, target)
            with mock.patch.object(Path, "replace", new=fail_publish):
                with self.assertRaises(KeyboardInterrupt):
                    full.merge_caches([source], destination)
            self.assertFalse(destination.exists())
            self.assertTrue(destination.with_suffix(".sources.json").exists())
            full.merge_caches([source], destination)
            self.assertTrue(destination.exists())

    def test_cache_union_is_immutable_restartable_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache(root/"a.sqlite", {"same":"yes", "a":"one"})
            cache(root/"b.sqlite", {"same":"yes", "b":"two"})
            full.merge_caches([root/"a.sqlite",root/"b.sqlite"],root/"union.sqlite")
            digest = full.file_sha256(root/"union.sqlite")
            full.merge_caches([root/"a.sqlite",root/"b.sqlite"],root/"union.sqlite")
            self.assertEqual(full.file_sha256(root/"union.sqlite"),digest)
            with sqlite3.connect(root/"union.sqlite") as db:
                self.assertEqual(db.execute("SELECT count(*) FROM responses").fetchone()[0],3)
            cache(root/"bad.sqlite", {"same":"different"})
            with self.assertRaisesRegex(ValueError,"conflicting cached responses"):
                full.merge_caches([root/"a.sqlite",root/"bad.sqlite"],root/"bad_union.sqlite")

    def test_oracle_is_posthoc_and_at_least_every_measured_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root/"native_gmemory/memory_gate.jsonl"
            save_rows(baseline, base_rows())
            sources = {"native_gmemory":baseline}
            for name,policy,index in [("all","global_all",0),("one","candidate_index",0),("two","candidate_index",1)]:
                path = root/name/"memory_gate.jsonl"
                save_rows(path, apply_plan(base_rows(),build_plan(baseline,policy=policy,index=index)))
                sources[name] = path
            args = mock.Mock(output_dir=root,bootstrap_samples=50,gate_evaluation_seed=2000)
            result = full.build_oracle(args,sources)
            self.assertEqual(result["gate_mean_reward"],1.0)
            rows = indexed_rows(root/"oracle_selective_drop/memory_gate.jsonl")
            self.assertTrue(all(row["oracle"]["uses_test_outcomes"] for row in rows.values()))
            self.assertEqual(rows["1"]["oracle"]["selected_source"],"native_gmemory")
            self.assertTrue(all(rows[key]["reward"] >= row["reward"] for path in sources.values()
                                for key,row in indexed_rows(path).items()))
            full.build_oracle(args,sources)

    def test_oracle_rejects_same_action_with_different_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root/"native_gmemory/memory_gate.jsonl"
            save_rows(baseline,base_rows())
            plan = build_plan(baseline,policy="candidate_index",index=1)
            rows = apply_plan(base_rows(),plan)
            save_rows(root/"a",rows)
            rows[0]["reward"] = .5
            save_rows(root/"b",rows)
            args = mock.Mock(output_dir=root,bootstrap_samples=5,gate_evaluation_seed=2000)
            with self.assertRaisesRegex(ValueError,"same intervention has different outcomes"):
                full.build_oracle(args,{"native_gmemory":baseline,"a":root/"a","b":root/"b"})

    def test_oracle_requires_every_single_candidate_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root/"native_gmemory/memory_gate.jsonl"
            save_rows(baseline, base_rows())
            save_rows(root/"one", apply_plan(base_rows(), build_plan(baseline, policy="candidate_index", index=0)))
            args = mock.Mock(output_dir=root, bootstrap_samples=5, gate_evaluation_seed=2000)
            with self.assertRaisesRegex(ValueError, "lacks exhaustive"):
                full.build_oracle(args, {"native_gmemory": baseline, "one": root/"one"})


class FullBenchmarkTests(unittest.TestCase):
    @staticmethod
    def data(root):
        fever,hotpot = root/"fever.jsonl",root/"hotpot.json"
        save_rows(fever,[{"id":i,"claim":f"claim {i}","label":"SUPPORTS" if i%2 else "REFUTES"} for i in range(10)])
        hotpot.write_text(json.dumps([{"_id":str(i),"question":f"question {i}","answer":"answer",
             "context":[["page",["sentence"]]],"supporting_facts":[["page",0]]} for i in range(10)]))
        return fever,hotpot

    def test_plan_only_uses_all_data_without_writes(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root = Path(directory)
            fever,hotpot = self.data(root)
            plans = full.main(["--fever-data",str(fever),"--hotpotqa-data",str(hotpot),
                               "--output-dir",str(root/"output"),"--support-count","2","--plan-only"])
            for plan in plans.values():
                self.assertEqual(plan["support_questions"]+plan["gate_training_questions"]+plan["final_test_questions"],10)
                self.assertEqual(plan["final_test_questions"],2)
            self.assertFalse((root/"output").exists())

    def test_seven_row_report_formats_real_scores_and_pending_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full._save(root/"fever/native_gmemory/summary.json",{"mean_reward":.71})
            full._save(root/"hotpotqa/native_gmemory/summary.json",{"mean_answer_f1":.4123})
            payload = full.write_reports(root,["fever","hotpotqa"])
            self.assertEqual(len(payload["rows"]),7)
            self.assertEqual(payload["rows"][0]["FEVER"],71)
            self.assertEqual(payload["rows"][0]["HotpotQA"],41.23)
            self.assertIsNone(payload["rows"][1]["FEVER"])
            self.assertIn("pending",(root/"policy_results.md").read_text())
            self.assertIn("71.00%",(root/"policy_results.html").read_text())
            self.assertIn("Oracle Selective Drop",(root/"policy_results.svg").read_text())
            self.assertEqual(ET.fromstring((root/"policy_results.svg").read_text()).tag,
                             "{http://www.w3.org/2000/svg}svg")

    def test_full_orchestration_failure_resume_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root = Path(directory)
            fever,hotpot = self.data(root)
            out = root/"output"
            argv = ["--fever-data",str(fever),"--hotpotqa-data",str(hotpot),"--output-dir",str(out),
                    "--support-count","2","--bootstrap-samples","100","--api-key","PRIVATE_TOKEN"]
            calls,fail = [],[True]

            def ready_matrix(adapter,args):
                return (args.output_dir/"matrix_progress.json").is_file()

            def ready_plan(adapter,args,output,command,plan):
                return (output/"summary.json").is_file()

            def execute(stage,command,log_path,api_key):
                calls.append(stage)
                self.assertEqual(api_key,"PRIVATE_TOKEN")
                self.assertNotIn("PRIVATE_TOKEN",json.dumps(command))
                name = stage.split(":")[0]
                target = out/name
                if stage.endswith(":matrix"):
                    benchmark = "FEVER_binary_offline" if name=="fever" else "HotpotQA_distractor_offline"
                    for arm in ("native_gmemory",*full.matrix.DEFAULT_ARMS):
                        save_rows(target/arm/"memory_gate.jsonl",base_rows(benchmark))
                        cache(target/arm/"llm_cache.sqlite")
                        if arm=="native_gmemory":
                            full._save(target/arm/"summary.json",{"mean_reward":.5,"mean_answer_f1":.5})
                        else:
                            full._save(target/arm/"comparison.json",full.compare(
                                target/"native_gmemory/memory_gate.jsonl",target/arm/"memory_gate.jsonl",bootstrap_samples=1))
                    full._save(target/"matrix_progress.json",{"status":"completed"})
                    return
                output = Path(command[command.index("--output-dir")+1])
                if stage=="fever:random_drop" and fail[0]:
                    fail[0] = False
                    output.mkdir(parents=True,exist_ok=True)
                    full._save(output/"run_manifest.json",{})
                    raise RuntimeError("simulated API timeout at random_drop")
                plan = load_plan(command[command.index("--intervention-plan")+1])
                save_rows(output/"memory_gate.jsonl",apply_plan(list(indexed_rows(target/"native_gmemory/memory_gate.jsonl").values()),plan))
                shutil.copy2(command[command.index("--cache-seed-from")+1],output/"llm_cache.sqlite")
                full._save(output/"summary.json",{"completed":True})

            stack.enter_context(mock.patch.object(full,"matrix_ready",side_effect=ready_matrix))
            stack.enter_context(mock.patch.object(full,"planned_ready",side_effect=ready_plan))
            stack.enter_context(mock.patch.object(full.base,"_run_command",side_effect=execute))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError,"simulated API timeout"):
                full.main(argv)
            state = json.loads((out/"benchmark_progress.json").read_text())
            self.assertEqual(state["status"],"failed")
            self.assertEqual(state["error"]["stage"],"fever:random_drop")
            self.assertIn("simulated API timeout",state["error"]["traceback"])
            full.main(argv)
            payload = json.loads((out/"policy_results.json").read_text())
            self.assertEqual(len(payload["rows"]),7)
            self.assertTrue(all(row["FEVER"] is not None and row["HotpotQA"] is not None for row in payload["rows"]))
            self.assertEqual(payload["rows"][-1]["FEVER"],100)
            self.assertEqual(calls.count("fever:matrix"),1)
            self.assertEqual(calls.count("fever:always_global_drop"),1)
            before = len(calls)
            full.main(argv)
            self.assertEqual(len(calls),before)
            self.assertNotIn("PRIVATE_TOKEN",(out/"benchmark_manifest.json").read_text())
            self.assertNotIn("PRIVATE_TOKEN",(out/"benchmark_progress.json").read_text())
            with self.assertRaisesRegex(SystemExit,"configuration changed"):
                full.main([*argv,"--random-seed","99"])


if __name__ == "__main__":
    unittest.main()
