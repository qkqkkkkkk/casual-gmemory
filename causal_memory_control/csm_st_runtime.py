"""Independent list-aware CSM deployment; original runtime/ST remain untouched."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from .csm_st_data import Deduplicator
from .runtime_gate import CHECKPOINT_SCHEMA, ExposureDecision, GMemoryExposureGate
from .types import ExposureAction, PotentialOutcomePrediction


class CSMExposureGate(GMemoryExposureGate):
    def __init__(self, *, predictor, deduplicator=None, **kwargs):
        kwargs.setdefault("max_drops", None)
        kwargs.setdefault("kappa", 0.0)
        kwargs.setdefault("delta", 0.0)
        if kwargs.get("mode", "learned") != "learned":
            raise ValueError("CSM entry only runs its new learned policy")
        if kwargs.get("kappa", 0) != 0 or kwargs.get("delta", 0) != 0 or kwargs.get("max_drops") is not None:
            raise ValueError("CSM uses CI>0 selection, kappa=delta=0 and unlimited drops")
        super().__init__(estimator=SimpleNamespace(fitted=True), **kwargs)
        self.predictor = predictor
        self.deduplicator = deduplicator or Deduplicator()
        self._scores, self._duplicates = {}, {}

    @classmethod
    def from_checkpoint(cls, checkpoint_path, **kwargs):
        from .csm_st_model import CSMPredictor
        from .semantic_features import SentenceTransformerEmbedder

        payload = json.loads(Path(checkpoint_path).read_text())
        if payload.get("schema") != CHECKPOINT_SCHEMA or "csm_st" not in payload:
            raise ValueError("new CSM entry requires a CSM-ST checkpoint, not an old ST checkpoint")
        kwargs.pop("semantic_cache_path", None)
        kwargs.pop("jev_cache_path", None)
        spec = payload["csm_st"]
        predictor = CSMPredictor.load(checkpoint_path, device=getattr(cls, "device", "cpu"))
        dedup = spec["dedup"]
        from .compare_gate_runs import file_sha256

        for name, expected in dedup.get("local_model_sha256", {}).items():
            if file_sha256(Path(dedup["model"]) / name) != expected:
                raise ValueError("local dedup model changed since CE collection")
        encoder = SentenceTransformerEmbedder(dedup["model"], revision=dedup.get("revision"),
            chunk_tokens=dedup["chunk_tokens"], cache_path=Path(checkpoint_path).parent / "dedup_runtime.sqlite")
        return cls(predictor=predictor, deduplicator=Deduplicator(encoder, dedup["threshold"]),
                   checkpoint_metadata=payload["training_metadata"], **kwargs)

    def filter_retrieval(self, result, *, query_task, task_state="", **kwargs):
        retrieval = self.adapter.adapt(result)
        if retrieval.failed_count:
            raise ValueError("CSM labels exclude failed trajectories; require failed_topk=0")
        candidates = retrieval.candidates
        if any(c.memory_type not in self.candidate_kinds for c in candidates):
            raise ValueError("CSM must score the full registered candidate list")
        kept, duplicates = self.deduplicator.indices([c.content for c in candidates])
        texts = [{"id": candidates[i].memory_id, "kind": candidates[i].memory_type,
                  "text": candidates[i].content} for i in kept]
        scores = self.predictor.predict(query_task, task_state or query_task, texts)
        if len(scores) != len(kept):
            raise ValueError("CSM predictor returned wrong list length")
        self._scores = {candidates[i].memory_id: v for i, v in zip(kept, scores)}
        self._duplicates = {candidates[i].memory_id: candidates[j].memory_id for i, j in duplicates.items()}
        # The parent preserves the original retrieval fingerprint, exposure logs,
        # memory formatting, failed-list handling and task-state reset semantics.
        return super().filter_retrieval(result, query_task=query_task, task_state=task_state, **kwargs)

    def _decide(self, event):
        duplicate = self._duplicates.get(event.memory.memory_id)
        value = self._scores[duplicate or event.memory.memory_id]
        keep = duplicate is None and value > 0
        prediction = PotentialOutcomePrediction(
            q_use=0.0, q_drop=0.0, utility=value,
            utility_class="positive" if value > 0 else "nonpositive", uncertainty=0.0,
            source="csm_st_ce_direct_ci_no_potential_outcome_heads", calibrated=False,
            training_samples=self.checkpoint_metadata.get("training_samples", 0))
        return ExposureDecision(event.event_id, event.memory.memory_id, event.memory.memory_type,
            ExposureAction.KEEP if keep else ExposureAction.DROP, value,
            "csm_duplicate" if duplicate else "csm_positive_ci" if keep else "csm_nonpositive_ci", prediction,
            policy_scores={"predicted_ci": value, "duplicate_of": duplicate,
                "selection_rule": "keep_iff_ci_gt_zero", "uncertainty_calibration": False,
                "q_use_q_drop_are_placeholders": True})

    def close(self):
        if self.deduplicator.encoder is not None:
            self.deduplicator.encoder.close()
        super().close()

