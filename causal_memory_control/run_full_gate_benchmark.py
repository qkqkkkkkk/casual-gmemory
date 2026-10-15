"""Full-data FEVER/HotpotQA seven-policy benchmark with resumable audit logs.

Reuses the existing split, training code, estimators, MAS and evaluation metric.
The oracle is a post-hoc, same-sample envelope, never a deployable gate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import sys
import traceback

from tqdm import tqdm

from . import run_hotpotqa_gate_comparison as base
from . import run_hotpotqa_gate_matrix as matrix
from .compare_gate_runs import compare, file_sha256
from .dataset_adapters import load_dataset_adapter
from .planned_drop import build_plan, indexed_rows, load_plan


SCHEMA = "full-data-seven-policy-benchmark-v1"
POLICIES = (
    ("native_gmemory", "Always Use (baseline)"),
    ("always_global_drop", "Always Global Drop"),
    ("random_drop", "Random Drop"),
    ("linear", "Linear Gate"),
    ("semantic_linear", "Semantic Linear Gate"),
    ("semantic_mlp", "Semantic MLP"),
    ("oracle_selective_drop", "Oracle Selective Drop (post-hoc upper bound)"),
)
NOTES = (
    "FEVER = binary label Accuracy (NEI excluded), not official evidence-aware FEVER Score. "
    "HotpotQA = Answer F1, not supporting-fact/joint F1. Main table values are percentages; detailed JSON scores/deltas use 0..1. "
    "All valid input-file samples are assigned to disjoint support, diagnostic train/dev and test sets. "
    "Oracle is selected AFTER test outcomes: KEEP, each single-memory DROP, and the Global-Drop action. "
    "With Global DROP ALL, oracle may drop all memories and is NOT budget-matched to learned gates. "
    "Oracle is an optimistic envelope of observed sampled outcomes, not an upper bound on expected performance."
)


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


def _save(path, payload):
    base._write_json(Path(path), _jsonable(payload))


def _strip_key(argv):
    """Keep credentials out of command strings, manifests, logs and exceptions."""
    result, skip = [], False
    for value in argv:
        if skip:
            skip = False
        elif value == "--api-key":
            skip = True
        elif not value.startswith("--api-key="):
            result.append(value)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--datasets", default="fever,hotpotqa")
    parser.add_argument("--fever-data", type=Path, default=Path("data/fever/fever_dev.jsonl"))
    parser.add_argument("--hotpotqa-data", type=Path,
                        default=Path("data/hotpotqa/raw/hotpot_dev_distractor_v1.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("causal_memory_control/results/full_gate_benchmark_v1"))
    parser.add_argument("--global-drop", choices=("all", "top1"), default="all",
                        help="all = no-memory control; top1 = hide trajectory rank 1 from all agents")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-drop-reference", choices=("none", *matrix.DEFAULT_ARMS), default="none",
                        help="none: one random DROP per test task; otherwise match that gate's per-task 0/1 budget")
    parser.add_argument("--plan-only", action="store_true")
    args, shared = parser.parse_known_args(argv)
    args.datasets = list(dict.fromkeys(value.strip() for value in args.datasets.split(",") if value.strip()))
    if not args.datasets or set(args.datasets) - {"fever", "hotpotqa"}:
        parser.error("--datasets must select fever,hotpotqa")
    forbidden = {"--arms", "--matrix-dir", "--memory-dir", "--diagnostic-results", "--data", "--test", "--checkpoint"}
    if any(value.split("=", 1)[0] in forbidden for value in shared):
        parser.error("dataset paths and the three gate arms are managed by this runner; use a fresh --output-dir")
    args.output_dir = args.output_dir.resolve()
    args.shared_argv = shared
    return args


def prepare_dataset(suite, name):
    adapter = load_dataset_adapter(name)
    root = suite.output_dir / name
    shared = list(suite.shared_argv)
    if not any(value.split("=", 1)[0] == "--endpoint" for value in shared):
        shared += ["--endpoint", "http://127.0.0.1:11434/v1"]
    argv = [*shared, "--data", str(getattr(suite, name + "_data").resolve()),
            "--matrix-dir", str(root), "--arms", ",".join(matrix.DEFAULT_ARMS), "--use-all-data",
            "--memory-dir", str(suite.output_dir / "artifacts" / name / "g-memory"),
            "--diagnostic-results", str(suite.output_dir / "artifacts" / name / "diagnostic")]
    _, args = matrix.parse_args(argv, adapter=adapter)
    adapter.resolve_plan(args)
    adapter.validate_args(args)
    return adapter, args, _strip_key(argv)


def matrix_ready(adapter, args):
    state = base._load_json(args.output_dir / "matrix_progress.json")
    if not state or state.get("status") != "completed":
        return False
    native = args.output_dir / "native_gmemory"
    if not adapter.evaluation_complete(args, native, mode="always_keep"):
        return False
    for arm in matrix.DEFAULT_ARMS:
        child = matrix._arm_args(args, arm)
        if not (adapter.checkpoint_complete(child)
                and adapter.evaluation_complete(child, args.output_dir / arm, mode="learned")
                and matrix._comparison_complete(args, args.output_dir / arm)):
            return False
    return True


def merge_caches(sources, output):
    """Freeze one immutable seed cache, rejecting conflicting seeded responses."""
    output = Path(output)
    provenance = {str(path.resolve()): file_sha256(path) for path in sources}
    manifest = output.with_suffix(".sources.json")
    previous = base._load_json(manifest)
    if previous is not None and previous.get("sources") != provenance:
        raise ValueError("oracle seed cache sources changed; use original artifacts or a fresh output directory")
    if output.exists():
        if previous != {"sources": provenance, "cache_sha256": file_sha256(output)}:
            raise ValueError("oracle seed cache or its sources changed; use original artifacts or a fresh output directory")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.sqlite")
    # Restartable, disposable construction file, never an existing experiment DB.
    with sqlite3.connect(temporary) as destination:
        destination.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        destination.execute("CREATE TABLE IF NOT EXISTS binary_scores (key TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
        for path in sources:
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
                available = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for table, column in (("responses", "response"), ("binary_scores", "payload_json")):
                    if table not in available:
                        continue
                    for key, value in source.execute(f"SELECT key, {column} FROM {table}"):
                        existing = destination.execute(f"SELECT {column} FROM {table} WHERE key=?", (key,)).fetchone()
                        if existing is not None and existing[0] != value:
                            raise ValueError(f"conflicting cached responses for seeded prompt {key}; backend reproducibility must be resolved")
                        destination.execute(f"INSERT OR IGNORE INTO {table} VALUES (?,?)", (key, value))
        destination.commit()
    # Write the descriptor first: interruption before rename can rebuild the
    # disposable file without stranding a completed DB with no provenance.
    _save(manifest, {"sources": provenance, "cache_sha256": file_sha256(temporary)})
    temporary.replace(output)


def save_plan(path, expected):
    if path.exists():
        if base._load_json(path) != expected:
            raise ValueError(f"intervention plan changed: {path}; use a fresh output directory")
    else:
        _save(path, expected)


def planned_command(adapter, args, output, plan, cache):
    command = adapter.evaluation_command(args, mode="always_keep", output=output,
                                         resume=(output / "run_manifest.json").is_file())
    command[command.index("--mode") + 1] = "planned_drop"
    command[command.index("--max-drops") + 1] = "-1"  # exact plan, including DROP ALL
    if "--cache-seed-from" in command:
        i = command.index("--cache-seed-from")
        del command[i:i + 2]
    return command + ["--intervention-plan", str(plan), "--cache-seed-from", str(cache)]


def planned_ready(adapter, args, output, command, plan):
    if not adapter.evaluation_complete(args, output, mode="planned_drop", command=command):
        return False
    run = base._load_json(output / "run_manifest.json")["run"]
    if (run.get("intervention_plan") != str(plan.resolve())
            or run.get("intervention_plan_sha256") != file_sha256(plan)):
        raise ValueError(f"completed evaluation uses a different intervention plan: {output}")
    expected = load_plan(plan)["tasks"]
    for task_id, row in indexed_rows(output / "memory_gate.jsonl").items():
        drops = [d["memory_id"] for d in row["decisions"] if d["action"] == "drop"]
        if (set(drops) != set(expected[task_id]["drop_memory_ids"])
                or row["retrieval_fingerprint"] != expected[task_id]["retrieval_fingerprint"]):
            raise ValueError(f"completed intervention disagrees with its plan: {output}, task {task_id}")
    return True


def audited_comparison(args, output):
    result = compare(args.output_dir / "native_gmemory" / "memory_gate.jsonl",
                     output / "memory_gate.jsonl", bootstrap_samples=args.bootstrap_samples,
                     seed=args.gate_evaluation_seed)
    _save(output / "comparison.json", result)
    if not result["paired_control_valid"] or not result["policy_valid"]:
        raise ValueError(f"invalid paired control: inspect {output / 'comparison.json'}")
    return result


def build_oracle(args, sources):
    """Post-hoc maximum over PREDECLARED actions; never calls the LLM."""
    root = args.output_dir
    baseline_path = root / "native_gmemory" / "memory_gate.jsonl"
    baseline = indexed_rows(baseline_path)
    inputs = {name: file_sha256(path) for name, path in sources.items()}
    output = root / "oracle_selective_drop"
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "oracle_manifest.json"
    identity = {"schema": "same-sample-oracle-envelope-v1", "sources": inputs,
                "selection": "max_reward_then_done_then_prefer_baseline",
                "uses_test_outcomes": True, "deployable": False}
    previous = base._load_json(manifest)
    if previous is not None and previous != identity:
        raise ValueError("oracle inputs changed; use original logs or a fresh output directory")
    indexed = {name: indexed_rows(path) for name, path in sources.items()}
    for name, rows in indexed.items():
        if set(rows) != set(baseline):
            raise ValueError(f"oracle branch {name} has different task IDs")
        result = compare(baseline_path, sources[name], bootstrap_samples=1, seed=0)
        if not result["paired_control_valid"] or not result["policy_valid"]:
            raise ValueError(f"oracle branch {name} has invalid paired controls")
    temporary = output / "memory_gate.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for task_id, base_row in baseline.items():
            choices, actions = [], {}
            for name, rows in indexed.items():
                row = rows[task_id]
                dropped = tuple(sorted(d["memory_id"] for d in row["decisions"] if d["action"] == "drop"))
                outcome = (row["reward"], row["done"], row.get("outcome", {}).get("prediction"))
                if dropped in actions and actions[dropped] != outcome:
                    raise ValueError(f"same intervention has different outcomes on oracle task {task_id}: {dropped}")
                actions[dropped] = outcome
                choices.append((name, row))
            required = {()} | {(d["memory_id"],) for d in base_row["decisions"]}
            if not required <= set(actions):
                raise ValueError(f"oracle task {task_id} lacks exhaustive single-candidate coverage")
            winner, row = max(choices, key=lambda item: (
                float(item[1]["reward"]), bool(item[1]["done"]), item[0] == "native_gmemory"))
            selected = copy.deepcopy(row)
            selected["mode"] = "oracle_selective_drop"
            selected["run_metadata"] = base_row["run_metadata"]
            selected["oracle"] = {"selected_source": winner, "uses_test_outcomes": True,
                "deployable": False, "choices": [
                    {"source": name, "reward": other["reward"], "done": other["done"],
                     "prediction": other.get("outcome", {}).get("prediction"),
                     "drop_memory_ids": [d["memory_id"] for d in other["decisions"] if d["action"] == "drop"]}
                    for name, other in choices]}
            handle.write(json.dumps(selected, ensure_ascii=False) + "\n")
    temporary.replace(output / "memory_gate.jsonl")
    _save(manifest, identity)
    result = audited_comparison(args, output)
    _save(output / "summary.json", {**result, "uses_test_outcomes": True,
                                    "deployable": False, "notes": NOTES})
    return result


def write_reports(root, datasets):
    rows, details = [], {}
    for policy, label in POLICIES:
        row = {"Policy": label}
        for name in datasets:
            path = root / name / policy
            if policy == "native_gmemory":
                summary = base._load_json(path / "summary.json")
                score = None if not summary else summary["mean_reward" if name == "fever" else "mean_answer_f1"]
            else:
                summary = base._load_json(path / "comparison.json")
                score = None if not summary else summary["gate_mean_reward"]
            row["FEVER" if name == "fever" else "HotpotQA"] = None if score is None else score * 100
            if summary:
                details.setdefault(name, {})[policy] = summary
        rows.append(row)
    payload = {"schema": SCHEMA, "metrics": {"FEVER": "binary Accuracy (%)", "HotpotQA": "Answer F1 (%)"},
               "notes": NOTES, "rows": rows, "details": details}
    _save(root / "policy_results.json", payload)
    fields = ["Policy", *("FEVER" if name == "fever" else "HotpotQA" for name in datasets)]
    with (root / "policy_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    rendered = [[str(row["Policy"]), *("pending" if row[field] is None else f"{row[field]:.2f}%" for field in fields[1:])]
                for row in rows]
    markdown = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |",
                *("| " + " | ".join(row) + " |" for row in rendered), "", NOTES]
    (root / "policy_results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    cells = "".join("<tr>" + "".join(f"<{tag}>{html.escape(value)}</{tag}>" for tag, value in
                                    [("th", row[0]), *(("td", value) for value in row[1:])]) + "</tr>" for row in rendered)
    document = "<!doctype html><meta charset='utf-8'><title>GMemory policy comparison</title>" \
        "<style>body{font-family:Arial,sans-serif;padding:32px;color:#111}table{border-collapse:collapse;min-width:760px}" \
        "th,td{padding:18px 24px;border:1px solid #ddd;text-align:center}tbody th{background:#ddd;max-width:340px}" \
        "thead{border-bottom:2px solid #111}p{max-width:1000px;line-height:1.6;color:#555}</style>" \
        "<table><thead><tr>" + "".join(f"<th>{field}</th>" for field in fields) + \
        "</tr></thead><tbody>" + cells + "</tbody></table><p>" + html.escape(NOTES) + "</p>"
    (root / "policy_results.html").write_text(document, encoding="utf-8")
    width, row_height = 1120, 78
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="755" viewBox="0 0 {width} 755">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#111" font-size="22">']
    labels = [fields, *rendered]
    for i, row in enumerate(labels):
        y = 12 + i * row_height
        svg += [f'<rect x="10" y="{y}" width="500" height="{row_height}" fill="#ddd"/>',
                f'<path d="M10 {y + row_height}H1110" stroke="#ccc"/>']
        for j, value in enumerate(row):
            x = 260 if j == 0 else 510 + (j - .5) * (600 / max(1, len(fields) - 1))
            if i == 7 and j == 0:
                svg += [f'<text x="{x}" y="{y+32}" text-anchor="middle" font-size="20">Oracle Selective Drop</text>',
                        f'<text x="{x}" y="{y+57}" text-anchor="middle" font-size="18">(same-sample post-hoc upper bound)</text>']
            else:
                svg.append(f'<text x="{x}" y="{y+47}" text-anchor="middle">{html.escape(value)}</text>')
    svg += ['</g>', '<text x="16" y="675" font-family="Arial" font-size="16">FEVER: binary Accuracy. HotpotQA: Answer F1. Values in %.</text>',
            '<text x="16" y="701" font-family="Arial" font-size="16">Oracle uses test outcomes and a larger action space; it is not deployable.</text>',
            '<text x="16" y="727" font-family="Arial" font-size="16">See policy_results.json and policy_results.md for confidence intervals, DROP details and definitions.</text>', '</svg>']
    (root / "policy_results.svg").write_text("\n".join(svg), encoding="utf-8")
    return payload


def main(argv=None):
    suite = parse_args(argv)
    prepared = {name: prepare_dataset(suite, name) for name in suite.datasets}
    configuration = {"schema": SCHEMA, "datasets": suite.datasets, "global_drop": suite.global_drop,
                     "random_seed": suite.random_seed, "random_drop_reference": suite.random_drop_reference,
                     "experiments": {name: adapter.configuration(args) for name, (adapter, args, _) in prepared.items()},
                     "semantic": {name: {k: getattr(args, k) for k in (
                         "semantic_model", "semantic_revision", "semantic_chunk_tokens", "mlp_hidden_sizes",
                         "mlp_epochs", "mlp_device", "mlp_learning_rate", "mlp_batch_tasks")}
                         for name, (_, args, _) in prepared.items()}}
    configuration = _jsonable(configuration)
    plans = {}
    for name, (adapter, args, _) in prepared.items():
        count = args.successful_topk + args.insights_topk
        if count < 1 or args.failed_topk != 0:
            raise SystemExit("this seven-policy protocol requires candidates and failed-topk=0")
        if suite.global_drop == "top1" and args.successful_topk < 1:
            raise SystemExit("global-drop top1 requires successful-topk >= 1")
        plans[name] = {**adapter.plan(args), "oracle_single_candidate_branches_per_test_task": count,
                       "maximum_final_team_inferences": args.final_claims * (6 + count),
                       "random_drop": "one per task" if suite.random_drop_reference == "none" else f"match {suite.random_drop_reference} task budgets"}
    print(json.dumps({"schema": SCHEMA, "plan_only": suite.plan_only, "datasets": plans,
                      "global_drop": suite.global_drop, "notes": NOTES}, ensure_ascii=False, indent=2))
    if suite.plan_only:
        return plans
    root = suite.output_dir
    manifest = root / "benchmark_manifest.json"
    previous = base._load_json(manifest)
    if previous is not None and previous != configuration:
        raise SystemExit("full benchmark configuration changed; use original arguments or a fresh --output-dir")
    if previous is None and root.exists() and any(root.iterdir()):
        raise SystemExit("non-empty benchmark directory has no benchmark_manifest.json")
    _save(manifest, configuration)
    _save(root / "experiment_plan.json", plans)
    state = {"status": "running", "current_stage": None, "stages": {}, "error": None}
    current_stage, current_command, current_log = "initialization", [], None

    def stage(name, action, ready=None, command=None, after=None):
        nonlocal current_stage, current_command, current_log
        current_stage, current_command = name, command or []
        current_log = root / "logs" / (name.replace(":", "_") + ".log")
        state.update(current_stage=name)
        state["stages"][name] = "running"
        _save(root / "benchmark_progress.json", state)
        if ready is None or not ready():
            if command:
                base._run_command(name, command, current_log, api_key=current_api_key)
            else:
                action()
            if ready is not None and not ready():
                raise RuntimeError(f"stage {name} exited but artifacts are incomplete")
        if after is not None:
            after()
        state["stages"][name] = "completed"
        _save(root / "benchmark_progress.json", state)
        progress.update(1)

    total = 1 + sum(5 + args.successful_topk + args.insights_topk for _, args, _ in prepared.values())
    current_api_key = None
    try:
        with tqdm(total=total, desc="Full-data policy benchmark", unit="stage") as progress:
            for name, (adapter, args, matrix_argv) in prepared.items():
                current_api_key = args.api_key
                command = [sys.executable, "-u", "-m", "causal_memory_control.run_gate_matrix", "--dataset", name, *matrix_argv]
                stage(name + ":matrix", None, lambda: matrix_ready(adapter, args), command)
                baseline = args.output_dir / "native_gmemory" / "memory_gate.jsonl"
                seed_cache = args.output_dir / "cache_seeds" / "matrix_union.sqlite"
                stage(name + ":freeze_cache", lambda: merge_caches(
                    [args.output_dir / arm / "llm_cache.sqlite" for arm in ("native_gmemory", *matrix.DEFAULT_ARMS)], seed_cache))
                sources = {arm: args.output_dir / arm / "memory_gate.jsonl" for arm in ("native_gmemory", *matrix.DEFAULT_ARMS)}
                jobs = [("always_global_drop", "global_all" if suite.global_drop == "all" else "global_top1", 0),
                        ("random_drop", "random_one", 0)]
                jobs += [(f"oracle_branches/drop_{i:03d}", "candidate_index", i)
                         for i in range(args.successful_topk + args.insights_topk)]
                cache = seed_cache
                for arm, policy, index in jobs:
                    plan = args.output_dir / "plans" / (arm.replace("/", "_") + ".json")
                    reference = (sources[suite.random_drop_reference]
                                 if policy == "random_one" and suite.random_drop_reference != "none" else None)
                    save_plan(plan, build_plan(baseline, policy=policy, seed=suite.random_seed,
                                              index=index, reference_path=reference))
                    output = args.output_dir / arm
                    command = planned_command(adapter, args, output, plan, cache)
                    stage(name + ":" + arm.replace("/", "_"), None,
                          lambda: planned_ready(adapter, args, output, command, plan), command,
                          after=lambda: audited_comparison(args, output))
                    sources[arm] = output / "memory_gate.jsonl"
                    cache = output / "llm_cache.sqlite"
                stage(name + ":oracle", lambda: build_oracle(args, sources))
                write_reports(root, suite.datasets)
            stage("reports", lambda: write_reports(root, suite.datasets))
        state.update(status="completed", current_stage=None)
        _save(root / "benchmark_progress.json", state)
        print((root / "policy_results.md").read_text(encoding="utf-8"))
        return root / "policy_results.json"
    except BaseException as exc:
        if current_log is not None:
            current_log.parent.mkdir(parents=True, exist_ok=True)
            with current_log.open("a", encoding="utf-8") as handle:
                handle.write("\n" + traceback.format_exc())
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error={
            "stage": current_stage, "type": type(exc).__name__, "message": str(exc),
            "command": current_command, "log_path": str(current_log) if current_log else None,
            "log_tail": base._tail(current_log) if current_log and current_log.is_file() else None,
            "traceback": traceback.format_exc()})
        _save(root / "benchmark_progress.json", state)
        raise


if __name__ == "__main__":
    main()
