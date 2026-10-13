"""Isolate benchmark tasks while preserving collaboration within each task."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


TASK_STATE_POLICY = "isolated-task-state-v1"
LEGACY_EVALUATION_SCHEMAS = frozenset({
    "gmemory-team-exposure-hotpotqa-evaluation-v1",
    "gmemory-team-exposure-fever-evaluation-v1",
})


def reset_task_execution_state(mas: Any) -> None:
    """Reset node buffers once per question, leaving graph edges intact."""
    for node in (*mas._agent_nodes.values(), mas._decision_node):
        node.clear_state()


def validate_task_state_policy(manifest: Mapping[str, Any], output: Path) -> None:
    if manifest.get("design", {}).get("task_state_policy") != TASK_STATE_POLICY:
        raise SystemExit(
            f"{output} was evaluated without isolated task state; its results "
            "cannot be resumed with this evaluator. Rerun the comparison pipeline "
            "to archive legacy evaluations, or use a new --output-dir. "
            "Frozen memory, counterfactual data and checkpoints can be reused."
        )


def archive_legacy_evaluations(root: Path, methods: Sequence[str]) -> Path | None:
    """Recover old pipelines without accepting results with cross-task history.

    A legacy baseline invalidates all dependent evaluations; otherwise only
    legacy arms are moved. Training artifacts and the pipeline identity remain
    in place. A second invocation resumes the new isolated evaluations.
    """
    legacy = []
    evaluations = []
    for method in methods:
        output = root / method
        manifest_path = output / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = manifest.get("design", {}).get("runner_schema", manifest.get("runner_schema"))
        if schema in LEGACY_EVALUATION_SCHEMAS:
            legacy.append(output)
            evaluations.append(output)
        elif schema in {value.removesuffix("v1") + "v2" for value in LEGACY_EVALUATION_SCHEMAS}:
            evaluations.append(output)
    if not legacy:
        return None

    # All dependent caches/manifests bind the baseline. Rebuild those arms
    # together if their reference evaluation must be regenerated.
    outputs = evaluations if any(path.name == "native_gmemory" for path in legacy) else legacy

    backup_root = root / "evaluation_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix="before-task-isolation-", dir=backup_root))
    reports = [root / name for name in (
        "comparison.json", "matrix_results.json", "matrix_results.csv", "matrix_results.md",
        "matrix_progress.json", "pipeline_progress.json",
    ) if (root / name).is_file()]
    moved = []
    for path in (*outputs, *reports):
        path.rename(backup / path.name)
        moved.append(path.name)
    (backup / "migration.json").write_text(json.dumps({
        "reason": "legacy evaluations retained agent history across independent tasks",
        "task_state_policy": TASK_STATE_POLICY,
        "archived": moved,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Archived legacy evaluations in {backup}; rerunning final evaluations "
        "with isolated task state. Frozen memory, diagnostics and checkpoints are retained."
    )
    return backup
