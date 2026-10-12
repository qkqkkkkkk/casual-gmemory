"""Task-local helpers for the isolated HotpotQA causal runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from mas.memory.common import MASMessage
from causal_memory_control.insight_metadata import (
    enrich_insight_payload,
    load_insight_catalog,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def memory_config(memory_dir: Path) -> dict[str, Any]:
    memory_dir = memory_dir.resolve()
    if not memory_dir.is_dir():
        raise FileNotFoundError(f"GMemory directory does not exist: {memory_dir}")
    return {
        "working_dir": str(memory_dir.parent),
        "hop": 1,
        "read_only": True,
    }


def _message_id(message: MASMessage) -> str:
    payload = json.dumps(
        MASMessage.to_dict(message), ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def candidate_metadata(
    frozen: Any,
    kind: str,
    index: int,
    *,
    memory_dir: Path | None = None,
) -> dict[str, Any]:
    if kind == "trajectory":
        candidate = frozen.successful[index]
        extra = candidate.extra_fields if isinstance(candidate.extra_fields, dict) else {}
        return {
            "kind": kind,
            "index": index,
            "candidate_id": f"trajectory-{_message_id(candidate)}",
            "task_main": candidate.task_main,
            "task_description": candidate.task_description,
            "label": candidate.label,
            "trajectory": candidate.task_trajectory,
            "key_steps": extra.get("key_steps"),
        }
    if kind == "insight":
        candidate = frozen.insights[index]
        payload = {
            "kind": kind,
            "index": index,
            "candidate_id": "insight-"
            + hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16],
            "text": candidate,
        }
        return enrich_insight_payload(
            payload,
            load_insight_catalog(memory_dir) if memory_dir is not None else {},
        )
    raise ValueError(f"unsupported candidate kind: {kind}")
