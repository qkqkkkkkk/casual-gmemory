from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from causal_memory_control import csm_st_evaluate
from causal_memory_control.csm_st_data import (Deduplicator, candidate_text, digest, extract_lists, frequency_weights,
    intervention_plan, read_rows, split_ids)
from causal_memory_control.csm_st_model import (CSMConfig, CSMPredictor, LocalGlobalCSM,
    contrastive_loss, tokenize_lists)
from causal_memory_control.csm_st_runtime import CSMExposureGate
from causal_memory_control.csm_st_scoring import (CachedCEScorer, HuggingFaceCEScorer,
    label_list, label_lists, messages)
from causal_memory_control.csm_st_training import train, validate_label_rows
from causal_memory_control.types import ExposureAction
from causal_memory_control.tests.test_semantic_jev import branch_fixture


class FakeCE:
    identity = {"model": "fake", "reduction": "mean", "generator_role": "test_proxy"}

    def __init__(self):
        self.calls = 0

    def score(self, prompt, answers):
        self.calls += 1
        text = prompt[-1]["content"]
        ce = 3.0 + 2 * ("harmful" in text) - 2 * ("helpful" in text)
        return {"ce": ce, "prompt_tokens": 10, "answer_tokens": [1]}

    def close(self):
        pass


def list_row(index=0, ci=None):
    row = {"id": str(index), "task_id": str(index), "query": f"question {index}",
        "state": f"task context {index}", "answers": ["gold-hidden"],
        "origin_task_ids": [str(index)], "candidates": [
            {"id": "bad", "kind": "insight", "rank": 1, "text": "harmful memory"},
            {"id": "good", "kind": "insight", "rank": 2, "text": "helpful memory"}]}
    if ci is not None:
        row["ci"] = ci
    return row


def tiny_components(root):
    vocabulary = root / "vocab.txt"
    vocabulary.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "task", "context",
        "question", "harmful", "helpful", "memory", "source", "0", "1", "2", "3", "4", "5", "6"]))
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=True)
    config = CSMConfig(encoder="tiny-offline", heads=2, global_layers=2, mlp_hidden=8,
        dropout=0, max_pair_tokens=32, pair_microbatch=2, gradient_checkpointing=False)
    encoder = BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, max_position_embeddings=64,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0))
    return config, tokenizer, LocalGlobalCSM(config, encoder)


class CSMDataScoringTests(unittest.TestCase):
    def test_empty_source_description_matches_runtime_not_old_feature_fallback(self):
        from causal_memory_control.gmemory_adapter import GMemoryRetrievalAdapter

        payload = {"kind": "trajectory", "task_main": "source-only-main", "task_description": "",
                   "trajectory": "reasoning", "key_steps": "steps"}
        message = SimpleNamespace(task_main=payload["task_main"], task_description="",
            task_trajectory="reasoning", extra_fields={"key_steps": "steps"}, label=True)
        actual = GMemoryRetrievalAdapter().adapt(([message], [], [])).candidates[0].content
        self.assertEqual(candidate_text(payload), actual)
        self.assertNotIn("source-only-main", actual)

    def test_rarity_weights_and_split_stay_fit_only(self):
        values = [list_row(i, [0, 0]) for i in range(9)] + [list_row(9, [-2, 2])]
        weights = frequency_weights(values)
        self.assertEqual(weights[-1], 10)
        self.assertAlmostEqual(weights[0], 10 / 9)
        fit, dev = split_ids(range(10), .2, 71)
        self.assertFalse(fit & dev)
        self.assertEqual((len(fit), len(dev)), (8, 2))

    def test_old_branch_text_reused_but_rewards_never_become_new_ci(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = branch_fixture(Path(directory))
            with redirect_stderr(io.StringIO()):
                rows = extract_lists([source], range(10), forbidden_ids=[99])
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[0]["answers"], ["hidden-gold"])
            self.assertNotIn("ci", rows[0])
            self.assertNotIn("outcome", rows[0])
            with self.assertRaisesRegex(ValueError, "overlap"):
                extract_lists([source], range(10), forbidden_ids=[0])

    def test_ci_sign_all_reuse_cache_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            root = Path(directory)
            fake = FakeCE()
            scorer = CachedCEScorer(fake, root / "ce.sqlite")
            rows = label_lists([list_row()], root / "labels.jsonl", "hotpotqa", scorer, root / "progress.json")
            self.assertEqual(rows[0]["ci"], [-2, 2])
            self.assertEqual(scorer.calls, 3)  # one USE ALL, two leave-one-out
            label_lists([list_row()], root / "labels.jsonl", "hotpotqa", scorer, root / "progress.json")
            self.assertEqual(scorer.calls, 3)
            label_list(list_row(), "hotpotqa", scorer)
            self.assertEqual(scorer.hits, 3)
            broken = list_row()
            broken["candidates"][0]["text"] = "different"
            with self.assertRaisesRegex(ValueError, "changed"):
                label_lists([broken], root / "labels.jsonl", "hotpotqa", scorer, root / "progress.json")
            scorer.close()

    def test_teacher_forcing_only_gold_tokens_no_sampling_or_truncation(self):
        class Tokenizer:
            chat_template, eos_token_id = "actual-test-template", 5

            def apply_chat_template(self, prompt, **kwargs):
                self.prompt = prompt
                return [1, 2]

            def encode(self, answer, **kwargs):
                return [3, 4]

        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(max_position_embeddings=16, to_dict=lambda: {"test": True})

            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
                self.asserted = not use_cache
                logits = torch.zeros((*input_ids.shape, 8))
                for i in range(input_ids.shape[1] - 1):
                    logits[0, i, input_ids[0, i + 1]] = 2
                return SimpleNamespace(logits=logits[:, -logits_to_keep:])

        tokenizer, network = Tokenizer(), Network()
        scorer = HuggingFaceCEScorer("fake", tokenizer=tokenizer, network=network,
                                    device="cpu", dtype="float32", max_length=16)
        prompt = messages("hotpotqa", "query", "supplied context", list_row()["candidates"])
        value = scorer.score(prompt, ["gold-hidden"])
        self.assertAlmostEqual(value["ce"], math.log(math.exp(2) + 7) - 2, places=6)
        self.assertTrue(network.asserted)
        self.assertFalse(network.marker.requires_grad)
        self.assertNotIn("gold-hidden", json.dumps(tokenizer.prompt))
        scorer.max_length = 4
        with self.assertRaisesRegex(ValueError, "no truncation"):
            scorer.score(prompt, ["gold-hidden"])

    def test_interrupted_list_reuses_completed_forwards_not_partial_labels(self):
        class InterruptedCE(FakeCE):
            def score(self, prompt, answers):
                if self.calls == 2:
                    raise RuntimeError("interrupted prefill")
                return super().score(prompt, answers)

        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            root = Path(directory)
            first = CachedCEScorer(InterruptedCE(), root / "cache.sqlite")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                label_lists([list_row()], root / "labels.jsonl", "hotpotqa", first, root / "progress.json")
            first.close()
            self.assertFalse((root / "labels.jsonl").exists())
            second = CachedCEScorer(FakeCE(), root / "cache.sqlite")
            result = label_lists([list_row()], root / "labels.jsonl", "hotpotqa", second, root / "progress.json")
            self.assertEqual(result[0]["ci"], [-2, 2])
            self.assertEqual((second.calls, second.hits), (1, 2))
            second.close()

    def test_dedup_and_fit_only_recombination(self):
        class Encoder:
            def embed(self, text):
                return [1, 0] if text.endswith("0") else [0, 1]

        dedup = Deduplicator(Encoder(), threshold=0.99)
        kept, duplicates = dedup.indices(["text0", "similar0", "text1"])
        self.assertEqual((kept, duplicates), ([0, 2], {1: 0}))
        fit_rows = [list_row(0, [-2, 2]), list_row(1, [-2, 2])]
        mixed = intervention_plan(fit_rows, Encoder(), hard_threshold=0.1, gamma=0.1)
        self.assertTrue(mixed)
        self.assertEqual({r["intervention"] for r in mixed}, {"positive", "negative"})
        for row in mixed:
            self.assertNotIn("ci", row)
            self.assertLessEqual(set(row["origin_task_ids"]), {"0", "1"})
        # The new labels are re-scored on the mixed lists, not copied.
        scored = [label_list(r, "hotpotqa", FakeCE()) for r in mixed]
        validate_label_rows([*fit_rows, *scored, list_row(2, [0, 0])], fit_ids={"0", "1"}, dev_ids={"2"})
        scored[0]["origin_task_ids"].append("2")
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_label_rows(scored, fit_ids={"0", "1"}, dev_ids={"2"})


class CSMModelTests(unittest.TestCase):
    def test_list_awareness_permutation_padding_gradients_and_unbounded_head(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(1)
            config, tokenizer, network = tiny_components(Path(directory))
            network.eval()
            row = list_row()
            tokens, counts = tokenize_lists(tokenizer, [row], config, "cpu")
            scores, _, _ = network(tokens, counts)
            reversed_row = {**row, "candidates": row["candidates"][::-1]}
            tokens2, counts2 = tokenize_lists(tokenizer, [reversed_row], config, "cpu")
            reversed_scores, _, _ = network(tokens2, counts2)
            torch.testing.assert_close(scores, reversed_scores.flip(1), atol=1e-5, rtol=1e-5)
            single = {**row, "candidates": row["candidates"][:1]}
            tokens3, counts3 = tokenize_lists(tokenizer, [single, row], config, "cpu")
            batched, _, valid = network(tokens3, counts3)
            self.assertEqual(valid.tolist(), [[True, False], [True, True]])
            alone = CSMPredictor(network, tokenizer).predict(row["query"], row["state"], single["candidates"])
            self.assertGreater(abs(float(scores[0, 0].detach()) - alone[0]), 1e-7)
            torch.testing.assert_close(scores[0], batched[1], atol=1e-5, rtol=1e-5)
            network.train()
            scores.square().mean().backward()
            self.assertIsNotNone(network.encoder.embeddings.word_embeddings.weight.grad)
            with torch.no_grad():
                network.output[-1].weight.zero_()
                network.output[-1].bias.fill_(3)
            result = CSMPredictor(network, tokenizer).predict(row["query"], row["state"], row["candidates"])
            self.assertEqual(result, [3, 3])

    def test_contrastive_pairs_and_no_pairs_stay_finite(self):
        representations = torch.randn(3, 8, requires_grad=True)
        loss, pairs = contrastive_loss(representations, [0, 0.01, 2])
        self.assertEqual(pairs, 2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(representations.grad)
        empty, pairs = contrastive_loss(representations, [0, 0, 0])
        self.assertEqual((float(empty.detach()), pairs), (0, 0))

    def test_train_export_offline_load_and_artifact_tamper(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            torch.manual_seed(2)
            config, tokenizer, network = tiny_components(root)
            options = {"seed": 2024, "device": "cpu", "epochs": 2, "batch_tasks": 2,
                "encoder_learning_rate": 0.001, "head_learning_rate": 0.001, "weight_decay": 0,
                "beta": 0.1, "temperature": 1, "positive_epsilon": 0.1, "negative_epsilon": 0.3,
                "rarity_alpha": 10, "rarity_bin_width": 1, "trivial_threshold": 5,
                "hard_threshold": 5, "trivial_keep_fraction": 1, "local_files_only": True}
            rows = [list_row(i, [-2, 2]) for i in range(5)]
            payload = train(rows, root / "new", model_config=config, options=options,
                fit_ids={"0", "1", "2", "3"}, dev_ids={"4"}, metadata={}, dedup_config={},
                label_provenance={"proxy": True}, network=network, tokenizer=tokenizer)
            self.assertEqual(payload["training_metadata"]["policy_calibration"]["recommended_max_drops"], -1)
            rng = torch.random.get_rng_state().clone()
            predictor = CSMPredictor.load(root / "new" / "checkpoint.json")
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(len(predictor.predict("question", "context", rows[0]["candidates"])), 2)
            self.assertEqual(len(payload["training_diagnostics"]["loss_history"]), 2)
            with (root / "new" / "model" / "weights.pt").open("ab") as handle:
                handle.write(b"damaged")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                CSMPredictor.load(root / "new" / "checkpoint.json")


class CSMRuntimeTests(unittest.TestCase):
    def test_multiple_drops_zero_drop_duplicates_and_candidate_order(self):
        class Predictor:
            def predict(self, query, state, candidates):
                self.candidates = candidates
                return [-2, 0, 3]

        predictor = Predictor()
        gate = CSMExposureGate(predictor=predictor, candidate_kinds=("insight",), mode="learned")
        gate.begin_task(100, {"task_main": "query", "gold_label": "NEVER INPUT"})
        result = gate.filter_retrieval(([], [], ["bad", "zero", "good", "good"]), query_task="query", task_state="state")
        self.assertEqual(result, ([], [], ["good"]))
        row = gate.end_task(1, True)
        self.assertEqual(row["dropped_count"], 3)
        self.assertIsNone(row["max_drops"])
        self.assertEqual(len(predictor.candidates), 3)
        self.assertEqual(row["decisions"][-1]["reason"], "csm_duplicate")
        gate.close()
        with self.assertRaisesRegex(ValueError, "unlimited"):
            CSMExposureGate(predictor=predictor, max_drops=1)

    def test_new_evaluation_binding_restores_old_runner_even_after_error(self):
        from causal_memory_control import fever_experiment

        original = fever_experiment.GMemoryExposureGate
        def fail(_):
            self.assertIsNot(fever_experiment.GMemoryExposureGate, original)
            raise RuntimeError("simulated")
        with mock.patch.object(fever_experiment, "main", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                csm_st_evaluate.main(["--dataset", "fever"])
        self.assertIs(fever_experiment.GMemoryExposureGate, original)


if __name__ == "__main__":
    unittest.main()
