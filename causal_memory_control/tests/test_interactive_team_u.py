from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from causal_memory_control import interactive_team_u as team
from causal_memory_control import interactive_worker as worker
from causal_memory_control import interactive_training as training
from causal_memory_control import run_interactive_gate_benchmark as cli
from causal_memory_control.csm_st_data import Deduplicator, read_rows
from causal_memory_control.interactive_data import digest, save
from causal_memory_control.interactive_protocol import learned_methods, policies, training_options
from causal_memory_control.interactive_report import report
from causal_memory_control.local_global_team_u import gate_class, checkpoint_spec, load_predictor
from causal_memory_control.tests.test_interactive_benchmark import REPO, measured, task


class Encoder:
    def embed(self, text):
        return [0, 1] if text == "other" else [1, 0]

    def close(self):
        pass


def fixture(directory, dataset="pddl"):
    root = Path(directory)
    config = {"output_dir": str(root), "repo": str(REPO), "include_local_global_team_u": True,
        "diagnostic_seed": 0, "evaluation_seed": 2000, "repeats": 2, "random_seeds": [11, 23, 47],
        "train_device": "cpu", "evaluation_device": "cpu", "encoder_model": "tiny-offline",
        "embedding_model": "tiny-dedup", "experiments": {dataset: {
            "tasks": [{**task(i, dataset), "task_type": "put"} for i in range(6)], "protocol": "synthetic_test",
            "split": {"support": ["0"], "fit": ["1", "2"], "dev": ["3"], "test": ["4", "5"]}}}}
    save(root / dataset / "snapshot.json", {"mock": True})
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, selected, seed, drops=(), expected=None):
            from causal_memory_control.interactive_runtime import audit
            calls.append((selected["id"], seed, tuple(drops)))
            row = measured(selected["id"], seed, drops)
            success = int("bad" in drops) if "bad" in drops or "good" in drops else seed % 2
            row["outcome"].update(success=float(success), raw_reward=float(success), progress=float(success),
                                  won=bool(success), done=bool(success))
            if expected:
                audit(row, expected, drops)
            return row

        def close(self):
            pass

    return root, config, Runner, calls


def collect(config, dataset, runner):
    with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", runner), \
         mock.patch.object(worker, "snapshot", return_value=Path(config["output_dir"]) / "unused-memory"):
        worker.collect(config, dataset)


class InteractiveTeamUTests(unittest.TestCase):
    def test_original_measured_labels_pairing_no_expert_ce_or_source_writes(self):
        for dataset in ("pddl", "alfworld"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root, config, runner, calls = fixture(directory, dataset)
                collect(config, dataset, runner)
                source = root / dataset / "diagnostic"
                before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
                with mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=Encoder()), \
                     mock.patch("causal_memory_control.csm_st_scoring.HuggingFaceCEScorer", side_effect=AssertionError("no CE")), \
                     mock.patch("causal_memory_control.csm_st_data.intervention_plan", side_effect=AssertionError("no relabeling")):
                    team.labels(config, dataset)
                    team.labels(config, dataset)
                rows = read_rows(root / dataset / "local_global_team_u/team_u_labels.jsonl")
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(r["team_u"] == [-.5, .5, 0] for r in rows))
                self.assertTrue(all(r["paired_samples_per_candidate"] == [2, 2, 2] for r in rows))
                self.assertTrue(all(len(r["candidates"]) == 3 for r in rows))
                self.assertNotIn("answers", rows[0])
                self.assertNotIn("ci", rows[0])
                self.assertEqual(before, {p: p.read_bytes() for p in source.rglob("*") if p.is_file()})
                self.assertEqual(len(calls), 24)
                raw_path = next((source / "tasks").glob("*.json"))
                raw_path.write_text(raw_path.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    team.validate_labels(config, dataset)

    def test_malformed_measured_coverage_and_aggregates_are_rejected(self):
        for failure in ("duplicate", "missing", "seed", "aggregate", "candidate", "drop_context"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root, config, runner, _ = fixture(directory)
                collect(config, "pddl", runner)
                path = root / "pddl/diagnostic/tasks" / (digest("1") + ".json")
                row = json.loads(path.read_text())
                if failure == "duplicate":
                    row["branches"].append(row["branches"][0])
                elif failure == "missing":
                    row["branches"].pop()
                elif failure == "seed":
                    row["branches"][0]["sample_seed"] = 100
                elif failure == "aggregate":
                    row["q_use"] = .75
                elif failure == "drop_context":
                    branch_path = root / "pddl/diagnostic/actions/parts" / (row["branches"][0]["drop_key"] + ".json")
                    branch = json.loads(branch_path.read_text())
                    branch["state"] = "changed public context"
                    branch["record_sha256"] = digest({k: v for k, v in branch.items() if k != "record_sha256"})
                    save(branch_path, branch)
                else:
                    row["candidates"][0]["content"] = "changed"
                save(path, row)
                with self.assertRaises(ValueError):
                    team.measured_lists(config, "pddl")

    def test_full_list_scoring_semantic_dedup_and_unlimited_nonpositive_deletion(self):
        class Predictor:
            def predict(self, query, state, candidates):
                self.inputs = (query, state, candidates)
                return [.2, .9, 0]

        predictor = Predictor()
        gate = gate_class()(predictor=predictor, deduplicator=Deduplicator(Encoder(), .98))
        gate.interactive_target = "team_u"
        row = measured()
        row["answers"] = ["HIDDEN GOLD PLAN"]
        drops, decisions = training.select(gate, row)
        self.assertEqual(drops, ["good", "other"])
        self.assertEqual(len(predictor.inputs[2]), 3)
        self.assertNotIn("HIDDEN GOLD PLAN", str(predictor.inputs))
        self.assertEqual(decisions[1]["reason"], "team_u_duplicate")
        self.assertEqual(decisions[1]["policy_scores"]["predicted_team_u"], .9)
        self.assertEqual(decisions[2]["reason"], "team_u_nonpositive")
        row["candidates"] = []
        self.assertEqual(training.select(gate, row), ([], []))

    def test_tiny_actual_training_exports_team_u_checkpoint_and_resumes_without_ce(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from causal_memory_control import csm_st_model

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root, config, runner, _ = fixture(directory)
            collect(config, "pddl", runner)
            with mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=Encoder()):
                team.labels(config, "pddl")
            vocabulary = root / "vocab.txt"
            vocabulary.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "public", "goal", "bad", "good", "other"]))
            tokenizer = BertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=True)
            original = csm_st_model.LocalGlobalCSM

            def network(model_config, **kwargs):
                encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=8,
                    num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, max_position_embeddings=512))
                return original(model_config, encoder)

            with mock.patch.object(csm_st_model, "LocalGlobalCSM", side_effect=network), \
                 mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 mock.patch.object(team, "training_options", return_value={**training_options(config, team_u=True), "epochs": 1}), \
                 mock.patch("causal_memory_control.csm_st_scoring.HuggingFaceCEScorer", side_effect=AssertionError("no CE")):
                team.train(config, "pddl")
            target = root / "pddl/local_global_team_u"
            payload, spec, _ = checkpoint_spec(target / "checkpoint.json")
            self.assertNotIn("csm_st", payload)
            self.assertEqual((spec["config"]["global_layers"], spec["config"]["heads"]), (3, 8))
            self.assertEqual(spec["dedup"]["placement"], "post_score")
            self.assertIn("dev_team_u_sign_accuracy", payload["training_diagnostics"]["loss_history"][0])
            self.assertIn("predicted_team_u", read_rows(target / "dev_predictions.jsonl")[0])
            predictor = load_predictor(target / "checkpoint.json", "cpu")
            self.assertTrue(all(torch.isfinite(torch.tensor(predictor.predict("q", "s", [
                {"id": "a", "kind": "insight", "text": "bad"}])))))
            before = (target / "checkpoint.json").read_bytes()
            team.train(config, "pddl")
            self.assertEqual(before, (target / "checkpoint.json").read_bytes())

    def test_nine_policy_evaluation_action_reuse_reporting_and_incomplete_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config, runner, calls = fixture(directory)
            collect(config, "pddl", runner)
            for method in learned_methods(config):
                save(root / "pddl" / method / "checkpoint.json", {"mock": method})
            gates = {m: SimpleNamespace(method=m, close=lambda: None) for m in learned_methods(config)}
            def selection(gate, baseline):
                drops = {"semantic_linear": [], "semantic_mlp": ["bad"], "semantic_st": ["bad"],
                         "csm_st": ["bad", "other"], "local_global_team_u": ["good", "other"]}[gate.method]
                return drops, [{"predicted_team_u": -.1}] if gate.method == "local_global_team_u" else []
            with mock.patch("causal_memory_control.interactive_runtime.NativeRunner", runner), \
                 mock.patch.object(worker, "snapshot", return_value=root / "unused"), \
                 mock.patch.object(training, "learned_gates", return_value=gates), \
                 mock.patch.object(training, "select", side_effect=selection):
                before = len(calls)
                worker.evaluate(config, "pddl")
                self.assertEqual(len(calls) - before, 14)
                summary = report(config)["pddl"]
                self.assertEqual(len(summary["policies"]), 12)
                self.assertEqual(summary["policies"]["local_global_team_u"]["n"], 2)
                self.assertIn("Local–Global + Team U + semantic dedup", (root / "matrix_results.md").read_text())
                before = len(calls)
                worker.evaluate(config, "pddl")
                self.assertEqual(len(calls), before)
                part = next((root / "pddl/evaluation/policy_parts").glob("*.json"))
                save(part, [r for r in json.loads(part.read_text()) if r["policy"] != "local_global_team_u"])
                with self.assertRaisesRegex(ValueError, "strategy coverage"):
                    worker.evaluate(config, "pddl")

    def test_opt_in_plan_and_old_eight_policy_default(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as stream:
            target = Path(directory) / "untouched"
            cli.main(["--datasets", "pddl", "--pddl-folds", "5", "--output-dir", str(target),
                      "--include-local-global-team-u", "--plan-only"])
            plan = json.loads(stream.getvalue())
            self.assertEqual(len(plan["policies"]), 9)
            self.assertEqual(plan["workload"]["pddl"]["test_schedules_upper_bound_shared_actions"], 600)
            self.assertEqual(plan["workload"]["pddl"]["team_u_new_label_schedules"], 0)
            self.assertIn("labels_team_u", plan["stages"])
            self.assertFalse(target.exists())
        self.assertEqual(len(policies({})), 8)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(["--stage", "train_team_u"])

    def test_simulation_oof_two_cohorts_and_all_artifacts_are_marked(self):
        from causal_memory_control.interactive_demo import generate

        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("causal_memory_control.interactive_report.interval", side_effect=lambda v, **kw: [min(v), max(v)]):
            root = Path(directory)
            result = generate(root, alf_unseen=3, alf_extra=4)
            self.assertEqual(result["pddl"]["policies"]["local_global_team_u"]["n"], 60)
            self.assertEqual(len(result["pddl"]["checkpoints_sha256"]), 5)
            self.assertEqual(result["alfworld"]["policies"]["local_global_team_u"]["n"], 3)
            self.assertTrue(json.loads((root / "matrix_results.json").read_text())["simulation"])
            self.assertTrue((root / "matrix_results.md").read_text().startswith("**模拟输出"))
            with (root / "matrix_results.csv").open() as stream:
                self.assertTrue(all(r["simulation"] == "True" for r in csv.DictReader(stream)))
            self.assertIn("人工构造", (root / "example_results.md").read_text())
            source = root / "alfworld/evaluation/policies.jsonl"
            rows = read_rows(source)
            rows[0].pop("simulation")
            from causal_memory_control.csm_st_data import write_rows
            write_rows(source, rows)
            with self.assertRaisesRegex(ValueError, "mix simulated"):
                report(json.loads((root / "benchmark_manifest.json").read_text()))

    def test_ce_and_team_u_must_use_the_same_frozen_semantic_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root, config, runner, _ = fixture(directory)
            collect(config, "pddl", runner)
            save(root / "pddl/csm_st/dedup_identity.json", {"model": "different"})
            with mock.patch("causal_memory_control.semantic_features.SentenceTransformerEmbedder", return_value=Encoder()):
                with self.assertRaisesRegex(ValueError, "differs from the CE"):
                    team.labels(config, "pddl")


if __name__ == "__main__":
    unittest.main()
