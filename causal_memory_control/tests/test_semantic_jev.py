from __future__ import annotations

from contextlib import ExitStack, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError

import numpy as np

from causal_memory_control import jev_api, train_gate
from causal_memory_control.estimator import PotentialOutcomeExample
from causal_memory_control.features import HashEmbedder, TeamExposureFeatureBuilder
from causal_memory_control.jev_gate import (
    CachedJevJudge, JevExposureGate, calibrate_threshold, judgment_request,
    main as jev_main,
)
from causal_memory_control.neural_estimator import MLPUtilityEstimator
from causal_memory_control.runtime_gate import GMemoryExposureGate, save_gate_checkpoint
from causal_memory_control.semantic_features import SentenceTransformerEmbedder, SemanticTeamFeatureBuilder
from causal_memory_control import run_hotpotqa_gate_matrix as matrix
from causal_memory_control.tests import test_hotpotqa_gate as hotpot_test
from causal_memory_control.compare_gate_runs import main as compare_main
from causal_memory_control.jsonl import read_jsonl


class TinyEncoder:
    def embed(self,text):
        return [0,0,1] if "harmful" in text else [0,1,0] if "helpful" in text else [1,0,0]

    def config(self):
        return {"type":"sentence-transformer-chunk-mean-v1","model":"test-encoder",
                "revision":"fixed-test","chunk_tokens":192,"dimensions":3}

    def close(self):
        pass


def event_fixture():
    gate = GMemoryExposureGate(mode="always_keep",feature_builder=SemanticTeamFeatureBuilder(TinyEncoder()))
    gate.begin_task(10,{"task_main":"Question?","task_description":"Supplied context",
                        "answer":"hidden-gold","label":"hidden-gold"})
    gate.set_exposure_agent_ids(["a","b","decision"])
    retrieval = gate.adapter.adapt(([],[],["harmful memory","helpful memory"]))
    return gate,gate._build_event(retrieval,retrieval.candidates[0],index=0,
                                 query_task="Question?",task_state="Supplied context")


def branch_fixture(root):
    memory = root/"memory"
    memory.mkdir()
    (memory/"causal_snapshot_manifest.json").write_text(json.dumps({"design_hash":"snapshot"}))
    source = root/"diagnostic"/"branches.jsonl"
    source.parent.mkdir()
    (source.parent/"run_manifest.json").write_text(json.dumps({"design":{
        "snapshot_design_hash":"snapshot","benchmark":"HotpotQA_distractor_offline","source_md5":"source",
    }}))
    rows = []
    for task_id in range(10):
        candidates = [{"candidate_id":name,"kind":"insight","text":f"{name} memory","index":i}
                      for i,name in enumerate(("harmful","helpful"))]
        for candidate in candidates:
            for condition in ("use_all","global_drop"):
                outcome = 0.5 if condition=="use_all" else (0.9 if candidate["index"]==0 else 0.1)
                rows.append({"task_id":task_id,"candidate":candidate,"candidate_set":candidates,
                    "task":{"task_main":f"Question {task_id}?","task_description":"Supplied context",
                            "gold_answer":"hidden-gold"},
                    "run_hash":"run","sample_seed":0,"repeat_index":0,"exposure_count":3,
                    "retrieval_sizes":{"insights":2},"condition":condition,
                    "outcome":{"team_score":outcome}})
    source.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return source,memory


class SemanticFeatureTests(unittest.TestCase):
    def test_semantic_coordinates_are_full_and_gold_is_excluded(self):
        gate,event = event_fixture()
        features = gate.feature_builder.build(event)
        self.assertEqual(features["semantic_memory::2"],1)
        self.assertEqual(features["semantic_query_memory_difference::2"],1)
        self.assertEqual(features["semantic_peer_mean::1"],1)
        self.assertNotIn("memory_text::0",features)
        self.assertNotIn("answer",gate._event_task_metadata())

    def test_train_and_runtime_semantic_features_match(self):
        gate,event = event_fixture()
        candidates = [{"candidate_id":memory.memory_id,"kind":"insight","text":memory.content,"index":index}
                      for index,memory in enumerate(event.candidate_set)]
        row = {"task_id":10,"candidate":candidates[0],"candidate_set":candidates,
               "task":{"task_main":event.query,"task_description":event.task_state,"gold_answer":"hidden-gold"},
               "exposure_count":3,"retrieval_sizes":{"insights":2}}
        examples = train_gate._examples_from_branches([
            {**row,"condition":"use_all","outcome":{"team_score":0.4}},
            {**row,"condition":"global_drop","outcome":{"team_score":0.9}},
        ],metric="team_score",task_ids=None,hash_dimensions=256,
        feature_builder=SemanticTeamFeatureBuilder(TinyEncoder()))
        self.assertEqual(examples[0].features,gate.feature_builder.build(event))

    def test_long_states_are_chunked_and_encoder_cache_resumes(self):
        class Tokenizer:
            def encode(self,text,**_):
                return text.split()
            def decode(self,tokens,**_):
                return " ".join(tokens)
            def num_special_tokens_to_add(self,**_):
                return 2

        class Encoder:
            tokenizer,max_seq_length = Tokenizer(),10
            calls = []

            def encode(self,chunks,**_):
                self.calls.extend(chunks)
                return np.array([[1.0,0.0] if "tail" not in chunk else [0.0,1.0] for chunk in chunks])

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/"semantic.sqlite"
            embedder = SentenceTransformerEmbedder("fake",revision="r1",cache_path=cache,chunk_tokens=8)
            embedder._model = Encoder()
            text = "one "*8+"tail"
            vector = embedder.embed(text)
            self.assertEqual(len(embedder._model.calls),2)
            self.assertGreater(vector[1],0.5)
            embedder.close()
            resumed = SentenceTransformerEmbedder("fake",revision="r1",cache_path=cache,chunk_tokens=8,expected_dimensions=2)
            self.assertEqual(resumed.embed(text),vector)
            self.assertIsNone(resumed._model)
            resumed.close()
            mismatch = SentenceTransformerEmbedder("fake",revision="r1",cache_path=cache,chunk_tokens=8,expected_dimensions=3)
            with self.assertRaisesRegex(ValueError,"dimension"):
                mismatch.embed(text)
            mismatch.close()


class NeuralEstimatorTests(unittest.TestCase):
    def test_nonlinear_fit_and_json_checkpoint_roundtrip(self):
        import torch

        examples = [PotentialOutcomeExample({"x":x},0.9 if x>0 else 0.1,0.1 if x>0 else 0.9,
                                           group_id=str(group),event_id=f"{group}-{x}")
                    for group in range(6) for x in (-1.0,1.0)]
        estimator = MLPUtilityEstimator(ensemble_size=2,min_samples=4,hidden_sizes=(8,4),
                                        epochs=120,learning_rate=0.02,batch_tasks=6)
        rng = torch.random.get_rng_state().clone()
        self.assertTrue(estimator.fit(examples))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertGreater(estimator.predict({"x":1}).utility,0.5)
        self.assertLess(estimator.predict({"x":-1}).utility,-0.5)
        restored = MLPUtilityEstimator.from_dict(json.loads(json.dumps(estimator.to_dict())))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertAlmostEqual(restored.predict({"x":0.7}).utility,estimator.predict({"x":0.7}).utility,places=6)
        self.assertGreater(estimator.contrastive_pairs,0)

    def test_training_cli_saves_semantic_mlp_and_restores_runtime(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source,memory = branch_fixture(root)
            checkpoint = root/"mlp.json"
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder",return_value=TinyEncoder()))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            train_gate.main(["--input",str(source),"--memory-dir",str(memory),"--output",str(checkpoint),
                "--feature-version","3","--estimator","mlp","--ensemble-size","2",
                "--metric","team_score",
                "--mlp-hidden-sizes","8,4","--mlp-epochs","2","--calibration-fraction","0.2",
                "--semantic-model","test-encoder","--semantic-revision","fixed-test"])
            payload = json.loads(checkpoint.read_text())
            self.assertEqual(payload["estimator"]["schema"],"mlp-utility-estimator-v1")
            self.assertEqual(payload["feature_builder"]["encoder"]["dimensions"],3)
            metadata = payload["training_metadata"]
            self.assertFalse(set(metadata["fit_task_ids"]) & set(metadata["policy_calibration"]["calibration_task_ids"]))
            gate = GMemoryExposureGate.from_checkpoint(checkpoint,mode="learned",memory_dir=memory)
            gate.begin_task(30,{"task_main":"Question?","task_description":"Supplied context"})
            gate.filter_retrieval(([],[],["harmful memory","helpful memory"]),query_task="Question?",task_state="Supplied context")
            self.assertEqual(len(gate._task_decisions),2)
            self.assertTrue(all(d.prediction.source=="bootstrap_semantic_mlp" for d in gate._task_decisions))
            gate.close()


class JevProtocolTests(unittest.TestCase):
    def request_fixture(self):
        _,event = event_fixture()
        return {**judgment_request(event,"jev_direct"),"provider":{"model_id":"jev-1.13.0"}}

    def test_http_protocol_and_probability_path(self):
        reply = {"model":"jev-1.13.0","answers":{"retain_helpful":{"type":"noul","noul":0.76}},
                 "usage":{"input_tokens":321,"output_tokens":28}}
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(reply).encode()
        with mock.patch.dict("os.environ",{"TYPESAFE_API_KEY":"test-secret"}),mock.patch.object(jev_api,"urlopen",return_value=response) as call:
            result = jev_api.judge(self.request_fixture())
            request = call.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(request.full_url,"https://api.typesafe.ai/v1/systemone")
            self.assertEqual(payload["model"],"jev-1.13.0")
            self.assertEqual(payload["questions"]["retain_helpful"]["type"],"noul")
            self.assertEqual(result["probabilities"]["retain_helpful"],0.76)
            self.assertNotIn("test-secret",json.dumps(payload))
            self.assertNotIn("hidden-gold",json.dumps(payload))

    def test_invalid_probability_and_http_error_fail_without_leaking_key(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"answers":{"retain_helpful":{"noul":1.2}}}'
        with mock.patch.dict("os.environ",{"TYPESAFE_API_KEY":"test-secret"}),mock.patch.object(jev_api,"urlopen",return_value=response):
            with self.assertRaisesRegex(jev_api.JevAPIError,"probability"):
                jev_api.judge(self.request_fixture())
        error = HTTPError("https://example.invalid",401,"test-secret",{},None)
        with mock.patch.dict("os.environ",{"TYPESAFE_API_KEY":"test-secret"}),mock.patch.object(jev_api,"urlopen",side_effect=error):
            with self.assertRaises(jev_api.JevAPIError) as caught:
                jev_api.judge(self.request_fixture())
            self.assertIn("401",str(caught.exception))
            self.assertNotIn("test-secret",str(caught.exception))


class JevPolicyTests(unittest.TestCase):
    def config(self):
        return {"adapter":"causal_memory_control.jev_api:judge","model_id":"jev-1.13.0","adapter_version":"v1"}

    def test_cache_prevents_duplicate_calls_and_preserves_question_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            _,event = event_fixture()
            cache = Path(directory)/"jev.sqlite"
            judge = CachedJevJudge(self.config(),cache)
            function = mock.Mock(return_value={"probabilities":{"retain_helpful":0.1},"usage":{"input_tokens":10,"output_tokens":2}})
            with mock.patch.object(judge,"_load_adapter",return_value=function):
                self.assertEqual(judge.score(event,"jev_direct")[0],0.1)
                judge.score(event,"jev_direct")
            self.assertEqual(function.call_count,1)
            self.assertEqual((judge.calls,judge.cache_hits,judge.input_tokens),(1,1,10))
            judge.close()
            resumed = CachedJevJudge(self.config(),cache)
            self.assertEqual(resumed.score(event,"jev_direct")[0],0.1)
            self.assertEqual(resumed.calls,0)
            resumed.close()

    def test_failed_judgment_keeps_memory_and_records_real_error(self):
        with tempfile.TemporaryDirectory() as directory:
            judge = CachedJevJudge(self.config(),Path(directory)/"jev.sqlite")
            gate = JevExposureGate(judge=judge,threshold=0.5,mode="jev_direct",max_drops=1)
            gate.begin_task(1,{"task_main":"Question?"})
            with mock.patch.object(judge,"_load_adapter",return_value=mock.Mock(side_effect=jev_api.JevAPIError("Jev HTTP 401"))):
                result = gate.filter_retrieval(([],[],["insight"]),query_task="Question?",task_state="State")
            row = gate.end_task(1,True)
            self.assertEqual(result[2],["insight"])
            self.assertEqual(row["outcome"]["jev_errors"],1)
            decision = row["decisions"][0]
            self.assertEqual(decision["reason"],"jev_error_default_keep")
            self.assertFalse(decision["prediction"]["calibrated"])
            self.assertNotIn("retention_score",decision["policy_scores"])
            gate.close()

    def test_jev_drop_budget_and_scores_are_not_counterfactual_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            judge = CachedJevJudge(self.config(),Path(directory)/"jev.sqlite")
            gate = JevExposureGate(judge=judge,threshold=0.5,mode="jev_direct",max_drops=1)
            gate.begin_task(1,{"task_main":"Question?"})
            with mock.patch.object(judge,"_load_adapter",return_value=lambda _: {"probabilities":{"retain_helpful":0.1}}):
                result = gate.filter_retrieval(([],[],["one","two"]),query_task="Question?",task_state="State")
            row = gate.end_task(1,True)
            self.assertEqual(len(result[2]),1)
            self.assertEqual(row["dropped_count"],1)
            self.assertEqual(row["decisions"][0]["prediction"]["q_use"],0)
            self.assertIn("not_causal_utility",row["decisions"][0]["prediction"]["source"])
            gate.close()

    def test_calibration_selects_keep_all_if_drops_harm(self):
        _,event = event_fixture()
        harmful = [(event,PotentialOutcomeExample({},1,0,group_id=str(i)),0.1) for i in range(5)]
        selected = calibrate_threshold(harmful,[0,0.5,1])
        self.assertEqual(selected["recommended_max_drops"],0)
        beneficial = [(event,PotentialOutcomeExample({},0,1,group_id=str(i)),0.1) for i in range(5)]
        selected = calibrate_threshold(beneficial,[0,0.25,0.5])
        self.assertEqual(selected["recommended_threshold"],0.25)

    def test_empirical_harm_constraint_can_reject_positive_average_gain(self):
        _,event = event_fixture()
        scored = [(event,PotentialOutcomeExample({},0,1,group_id="a"),0.1),
                  (event,PotentialOutcomeExample({},0.1,0,group_id="b"),0.1)]
        self.assertEqual(calibrate_threshold(scored,[0.5])["recommended_max_drops"],1)
        self.assertEqual(calibrate_threshold(scored,[0.5],max_changed_harm_rate=0.25)["recommended_max_drops"],0)

    def test_decomposed_general_guidance_can_have_value_without_direct_relevance(self):
        from causal_memory_control.jev_gate import retention_score

        score = retention_score({"claim_specific_relevance":0.1,"general_guidance":0.9,"conflict":0.0,"redundancy":0.1},"jev_decomposed")
        self.assertGreater(score,0.8)

    def test_calibration_checkpoint_can_be_loaded_and_uses_only_held_out_calls(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            source,memory = branch_fixture(root)
            checkpoint = root/"jev.json"
            calls = []

            def service(request):
                calls.append(request)
                state = json.loads(request["state"])
                score = 0.1 if "harmful" in state["target_memory"]["text"] else 0.9
                return {"probabilities":{"retain_helpful":score}}

            stack.enter_context(mock.patch("causal_memory_control.jev_api.judge",side_effect=service))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            jev_main(["--input",str(source),"--output",str(checkpoint),"--memory-dir",str(memory),
                      "--mode","jev_direct","--cache",str(root/"jev.sqlite")])
            payload = json.loads(checkpoint.read_text())
            self.assertEqual(len(calls),4)
            self.assertNotIn("hidden-gold",json.dumps(calls))
            self.assertEqual(payload["training_metadata"]["policy_calibration"]["recommended_max_drops"],1)
            self.assertEqual(payload["training_metadata"]["fit_task_ids"],[])
            gate = GMemoryExposureGate.from_checkpoint(checkpoint,mode="jev_direct",memory_dir=memory,
                                                       jev_cache_path=root/"jev.sqlite")
            gate.begin_task(30,{"task_main":"Question?"})
            result = gate.filter_retrieval(([],[],["harmful memory","helpful memory"]),query_task="Question?",task_state="Supplied context")
            self.assertEqual(result[2],["helpful memory"])
            gate.close()


class JevRunnerTests(unittest.TestCase):
    def test_failed_judgments_invalidate_report_and_retry_only_affected_task(self):
        runner = hotpot_test.runner
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data,memory = hotpot_test.HotpotRunnerTests.fixture(root)
            baseline,output = root/"native_gmemory",root/"jev_direct"
            hotpot_test.HotpotRunnerTests.patches(stack)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            hotpot_test._HotpotMacNet.fail_id = None
            hotpot_test._HotpotMacNet.schedule_count = 0
            runner.main(hotpot_test.HotpotRunnerTests.argv(data,memory,baseline))
            # The fake chat client has no real SQLite file; only file identity is
            # needed here, while the actual runner executes the memory hook.
            (baseline/"llm_cache.sqlite").write_bytes(b"fake-cache")
            snapshot = json.loads((memory/"causal_snapshot_manifest.json").read_text())
            context = {key:snapshot[key] for key in ("benchmark","source_md5","model","graph_type","node_num","embedding_model")}
            context.update(snapshot_design_hash=snapshot["design_hash"],successful_topk=3,failed_topk=0,insights_topk=3,threshold=0.0)
            checkpoint = root/"jev.json"
            checkpoint.write_text(json.dumps({"schema":matrix.evaluation.CHECKPOINT_SCHEMA,
                "jev_policy":{"mode":"jev_direct","prompt_version":"memory-retention-judgments-v1",
                    "threshold":0.5,"provider":{"adapter":"causal_memory_control.jev_api:judge","model_id":"jev-1.13.0","adapter_version":"v1"}},
                "training_metadata":{"training_task_ids":[1],"candidate_kinds":["trajectory","insight"],
                    "candidate_kind_ranks":{"trajectory":[1],"insight":[1]},"training_contexts":[context],
                    "policy_calibration":{"recommended_max_drops":1}}}))
            argv = [*hotpot_test.HotpotRunnerTests.argv(data,memory,output),"--mode","jev_direct",
                    "--checkpoint",str(checkpoint),"--cache-seed-from",str(baseline/"llm_cache.sqlite")]
            calls = []
            fail = True

            def service(request):
                state = json.loads(request["state"])
                calls.append(state["question"])
                if fail and "site 2" in state["question"] and state["target_memory"]["kind"]=="insight":
                    raise jev_api.JevAPIError("Jev HTTP 503")
                return {"probabilities":{"retain_helpful":0.9}}

            stack.enter_context(mock.patch("causal_memory_control.jev_api.judge",side_effect=service))
            runner.main(argv)
            rows = read_jsonl(output/"memory_gate.jsonl")
            self.assertEqual([row["outcome"]["jev_errors"] for row in rows],[1,0])
            report = output/"comparison.json"
            compare_argv = ["--baseline",str(baseline/"memory_gate.jsonl"),"--gate",str(output/"memory_gate.jsonl"),
                            "--output",str(report),"--bootstrap-samples","10","--require-policy-success"]
            with self.assertRaisesRegex(SystemExit,"Jev"):
                compare_main(compare_argv)
            self.assertFalse(json.loads(report.read_text())["policy_valid"])
            schedules,calls_before = hotpot_test._HotpotMacNet.schedule_count,len(calls)
            fail = False
            runner.main([*argv,"--resume","--retry-jev-errors"])
            self.assertEqual(hotpot_test._HotpotMacNet.schedule_count,schedules+1)
            self.assertEqual(len(calls),calls_before+1)
            self.assertTrue(list(output.glob("memory_gate.jsonl.before-jev-retry-*")))
            self.assertTrue(all(row["outcome"]["jev_errors"]==0 for row in read_jsonl(output/"memory_gate.jsonl")))
            self.assertTrue(compare_main(compare_argv)["policy_valid"])
            options,args = matrix.parse_args([])
            args.data,args.memory_dir,args.output_dir,args.checkpoint = data,memory,root,checkpoint
            args.training_claims,args.evaluation_count,args.final_claims = 1,3,2
            args.model,args.embedding_model,args.node_num = "fake-model","fake-embedding",2
            args.endpoint = "http://unused.invalid/v1"
            self.assertTrue(matrix._evaluation_complete(options,args,"jev_direct",output))
            with (output/"memory_gate.jsonl").open("a") as handle:
                handle.write(json.dumps(rows[0])+"\n")
            with self.assertRaisesRegex(RuntimeError,"duplicate"):
                matrix._evaluation_complete(options,args,"jev_direct",output)


class MatrixTests(unittest.TestCase):
    def test_default_local_pipeline_trains_evaluates_and_resumes_without_jev(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            data,memory = hotpot_test.HotpotRunnerTests.fixture(root)
            prototype = json.loads(data.read_text())[2]
            data.write_text(json.dumps([{**prototype,"_id":f"h{i}","question":f"Where is site {i}?"} for i in range(13)]))
            snapshot_path = memory/"causal_snapshot_manifest.json"
            snapshot = json.loads(snapshot_path.read_text())
            snapshot["design"].update(source_md5=hotpot_test.file_md5(data),evaluation_ids=list(range(1,13)))
            snapshot.update(snapshot["design"])
            snapshot["design_hash"] = hotpot_test.stable_hash(snapshot["design"])
            snapshot_path.write_text(json.dumps(snapshot))
            diagnostic = root/"diagnostic"
            argv = ["--data",str(data),"--memory-dir",str(memory),"--diagnostic-results",str(diagnostic),
                    "--matrix-dir",str(root/"matrix"),"--model","fake-model","--embedding-model","fake-embedding",
                    "--endpoint","http://unused.invalid/v1","--node-num","2","--support-count","1",
                    "--evaluation-count","12","--smoke-claims","1","--mlp-epochs","2",
                    "--mlp-hidden-sizes","8,4","--semantic-model","test-encoder","--semantic-revision","fixed-test",
                    "--bootstrap-samples","100"]
            options,args = matrix.parse_args(argv)
            matrix.base._resolve_plan(args)
            context = {key:snapshot[key] for key in ("benchmark","source_md5","model","graph_type","node_num","embedding_model")}
            context.update(snapshot_design_hash=snapshot["design_hash"],successful_topk=3,failed_topk=0,insights_topk=3,threshold=0.0)

            def collect():
                from causal_diagnostic.hotpotqa_oracle_recipient.data import load_hotpotqa

                examples = load_hotpotqa(data)
                message = hotpot_test._FakeMessage()
                candidates = [{"candidate_id":"trajectory-rank1","kind":"trajectory","index":0,
                               "task_main":message.task_main,"task_description":message.task_description,
                               "trajectory":message.task_trajectory,"label":True},
                              {"candidate_id":"insight-rank1","kind":"insight","index":0,
                               "text":"Read both supporting paragraphs."}]
                for seed,suffix in ((0,""),(1000,"_retest")):
                    path = diagnostic/f"seed{seed}{suffix}"
                    path.mkdir(parents=True)
                    (path/"run_manifest.json").write_text(json.dumps({"design":context}))
                    rows = []
                    for raw in examples[1:1+args.training_claims]:
                        task = hotpot_test.runner.prepare_example(raw)
                        for candidate in candidates:
                            for repeat in range(2):
                                for condition in ("use_all","global_drop"):
                                    reward = 0.5 if condition=="use_all" else 0.9 if candidate["kind"]=="insight" else 0.1
                                    rows.append({"task_id":raw["id"],"task":task,"candidate":candidate,
                                        "candidate_set":candidates,"condition":condition,"run_hash":str(seed),
                                        "sample_seed":seed+repeat,"repeat_index":repeat,"exposure_count":3,
                                        "retrieval_sizes":{"successful_trajectories":1,"insights":1},
                                        "outcome":{"team_score":reward}})
                    (path/"branches.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))

            commands = []

            def execute(stage,command,log_path,**_):
                commands.append(stage)
                module = command[3]
                if stage=="diagnostic":
                    collect()
                elif module=="causal_memory_control.train_gate":
                    train_gate.main(command[4:])
                elif module=="causal_memory_control.jev_gate":
                    jev_main(command[4:])
                elif module=="causal_memory_control.hotpotqa_experiment":
                    output = hotpot_test.runner.main(command[4:])
                    cache = output.parent/"llm_cache.sqlite"
                    if not cache.exists():
                        cache.write_bytes(b"fake-cache")
                else:
                    compare_main(command[4:])

            hotpot_test.HotpotRunnerTests.patches(stack)
            hotpot_test._HotpotMacNet.fail_id = None
            stack.enter_context(mock.patch.object(matrix.base,"_run_command",side_effect=execute))
            stack.enter_context(mock.patch.object(matrix.base,"_diagnostic_complete",side_effect=lambda _:(diagnostic/"seed1000_retest"/"branches.jsonl").exists()))
            stack.enter_context(mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder",return_value=TinyEncoder()))
            jev_service = stack.enter_context(mock.patch("causal_memory_control.jev_api.judge",
                side_effect=AssertionError("default matrix must not call Jev")))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            result = matrix.main(argv)
            payload = json.loads(result.read_text())
            self.assertEqual([row["method"] for row in payload["results"]],["native_gmemory",*matrix.DEFAULT_ARMS])
            self.assertEqual(len(commands),11)
            for arm in matrix.DEFAULT_ARMS:
                comparison = json.loads((result.parent/arm/"comparison.json").read_text())
                self.assertEqual(comparison["paired_tasks"],args.final_claims)
                self.assertTrue(comparison["paired_control_valid"])
                self.assertTrue(comparison["policy_valid"])
            self.assertIn("semantic_mlp",(result.parent/"matrix_results.md").read_text())
            self.assertNotIn("Jev errors",(result.parent/"matrix_results.md").read_text())
            self.assertFalse((result.parent/"jev_cache.sqlite").exists())
            matrix.main(argv)
            self.assertEqual(len(commands),11)
            jev_service.assert_not_called()

            # Updating an interrupted v1 experiment must rerun its evaluations,
            # retain fitted checkpoints, and resume normally thereafter.
            saved_checkpoints = {arm:(result.parent/"checkpoints"/f"{arm}.json").read_bytes()
                                 for arm in matrix.DEFAULT_ARMS}
            old_log = (result.parent/"semantic_linear/memory_gate.jsonl").read_bytes()
            for method in ("native_gmemory", "semantic_linear"):
                path = result.parent/method/"run_manifest.json"
                manifest = json.loads(path.read_text())
                manifest["runner_schema"] = "gmemory-team-exposure-hotpotqa-evaluation-v1"
                manifest["design"]["runner_schema"] = manifest["runner_schema"]
                manifest["design"].pop("task_state_policy")
                path.write_text(json.dumps(manifest))
            before = len(commands)
            matrix.main(argv)
            repair_stages = commands[before:]
            self.assertIn("native_gmemory", repair_stages)
            self.assertIn("semantic_linear:evaluate", repair_stages)
            self.assertNotIn("diagnostic", repair_stages)
            self.assertFalse(any(stage.endswith(":train") for stage in repair_stages))
            for arm, payload in saved_checkpoints.items():
                self.assertEqual((result.parent/"checkpoints"/f"{arm}.json").read_bytes(), payload)
                self.assertTrue(json.loads((result.parent/arm/"comparison.json").read_text())["paired_control_valid"])
            backups = list((result.parent/"evaluation_backups").iterdir())
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0]/"semantic_linear/memory_gate.jsonl").read_bytes(), old_log)
            before = len(commands)
            matrix.main(argv)
            self.assertEqual(len(commands), before)

    def test_matrix_builds_true_mlp_command_and_common_baseline_cache(self):
        options,args = matrix.parse_args(["--matrix-dir","/tmp/test-gate-matrix"])
        args.training_claims,args.final_claims = 400,100
        child = matrix._arm_args(args,"semantic_mlp")
        train_command = matrix.base._train_command(child)
        self.assertEqual(train_command[train_command.index("--estimator")+1],"mlp")
        self.assertEqual(train_command[train_command.index("--feature-version")+1],"3")
        with mock.patch.object(matrix.base,"_effective_kappa",return_value=0.1),mock.patch.object(matrix.base,"_load_json",return_value={"training_metadata":{"policy_calibration":{"recommended_max_drops":1}}}):
            evaluation = matrix._evaluation_command(options,child,"semantic_mlp",args.output_dir/"semantic_mlp")
        self.assertEqual(evaluation[evaluation.index("--cache-seed-from")+1],str(args.output_dir/"native_gmemory"/"llm_cache.sqlite"))

    def test_all_five_arms_resume_after_late_failure_without_rerunning_completed_stages(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data,_ = hotpot_test.HotpotRunnerTests.fixture(root)
            sample = json.loads(data.read_text())[0]
            data.write_text(json.dumps([{**sample,"_id":f"h{i}"} for i in range(13)]))
            argv = ["--data",str(data),"--matrix-dir",str(root/"matrix"),"--support-count","1",
                    "--evaluation-count","10","--smoke-claims","1","--memory-dir",str(root/"snapshot"),
                    "--diagnostic-results",str(root/"diagnostic"),"--arms",",".join(matrix.ARMS)]
            complete,executed = set(),[]
            fail = True

            def command(stage,command,log_path,**_):
                executed.append(stage)
                if fail and stage=="jev_decomposed:evaluate":
                    log_path.parent.mkdir(parents=True,exist_ok=True)
                    log_path.write_text("simulated child failure")
                    raise matrix.base.StageFailure(stage,7,command,log_path,"simulated child failure")
                if "--output" in command and stage.endswith(":train"):
                    path = Path(command[command.index("--output")+1])
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps({"training_metadata":{"policy_calibration":{"recommended_max_drops":1}}}))
                complete.add(stage)

            stack.enter_context(mock.patch.object(matrix.base,"_run_command",side_effect=command))
            stack.enter_context(mock.patch.object(matrix.base,"_diagnostic_complete",side_effect=lambda _:"diagnostic" in complete))
            stack.enter_context(mock.patch.object(matrix.base,"_validated_evaluation_complete",side_effect=lambda *a,**k:"native_gmemory" in complete))
            stack.enter_context(mock.patch.object(matrix.base,"_checkpoint_complete",side_effect=lambda a:f"{a.checkpoint.stem}:train" in complete))
            stack.enter_context(mock.patch.object(matrix,"_jev_complete",side_effect=lambda m,a,arm,**k:f"{arm}:train" in complete))
            stack.enter_context(mock.patch.object(matrix,"_evaluation_complete",side_effect=lambda m,a,arm,p,**k:f"{arm}:evaluate" in complete))
            stack.enter_context(mock.patch.object(matrix,"_comparison_complete",side_effect=lambda a,p:f"{p.name}:compare" in complete))
            stack.enter_context(mock.patch.object(matrix.base,"_effective_kappa",return_value=0.1))
            stack.enter_context(mock.patch.object(matrix,"_report",return_value=[]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            with self.assertRaises(matrix.base.StageFailure):
                matrix.main(argv)
            state = json.loads((root/"matrix"/"matrix_progress.json").read_text())
            self.assertEqual(state["error"]["stage"],"jev_decomposed:evaluate")
            self.assertIn("simulated child failure",state["error"]["log_tail"])
            self.assertEqual(len(complete),15)
            fail = False
            previous = len(executed)
            matrix.main(argv)
            self.assertEqual(executed[previous:],["jev_decomposed:evaluate","jev_decomposed:compare"])
            previous = len(executed)
            matrix.main(argv)
            self.assertEqual(len(executed),previous)


if __name__ == "__main__":
    unittest.main()
