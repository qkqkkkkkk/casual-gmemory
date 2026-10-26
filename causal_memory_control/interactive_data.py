"""Read-only inventory and grouped splits for the interactive benchmark.

No environment/model imports here: --plan-only must also work on a login node.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re

SCHEMA = "gmemory-interactive-success-v1"
DOMAINS = ("barman", "blockworld", "gripper", "tyreworld")
TYPES = {"pick_and_place_simple": "put", "look_at_obj_in_light": "examine",
         "pick_clean_then_place_in_recep": "clean", "pick_heat_then_place_in_recep": "heat",
         "pick_cool_then_place_in_recep": "cool", "pick_two_obj_and_place": "puttwo"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def locked(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"registered input/configuration changed: {path}; use a new output directory")
    else:
        save(path, value)


def files_identity(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob("*"))
            if p.is_file() and not p.name.endswith(("-shm", "-wal"))}


def pddl_inventory(repo):
    repo = Path(repo)
    root = repo / "tasks/envs/pddl_env/pddlgym/pddl"
    label = repo / "data/pddl/test.jsonl"
    annotations = [json.loads(s) for s in label.read_text().splitlines() if s.strip()]
    grouped = defaultdict(list)
    for row in annotations:
        grouped[row["additional_info"]["subtask"]].append(row)
    result = []
    for domain in DOMAINS:
        problems = sorted((root / domain).glob("*.pddl"))  # Native registration is lexicographic.
        if len(problems) != len(grouped[domain]):
            raise ValueError(f"PDDL annotation/problem count mismatch: {domain}")
        for index, (problem, annotation) in enumerate(zip(problems, grouped[domain])):
            text = re.sub(r";[^\n]*", "", problem.read_text()).lower()
            # Ignore the PDDL problem name when detecting duplicated initial states/goals.
            canonical = re.sub(r"\(problem\s+[^)]+\)", "(problem identity)", text)
            content = " ".join(canonical.split())
            result.append({"id": f"pddl:{domain}:{problem.stem}",
                "group": f"pddl:{domain}:{digest(content)}", "dataset": "pddl", "domain": domain,
                "difficulty": annotation["difficulty"], "annotation_id": annotation["id"],
                "annotation_subgoals": annotation["subgoals"],
                "files": {str(problem.relative_to(repo)): sha(problem),
                          str((root / (domain + '.pddl')).relative_to(repo)): sha(root / (domain + '.pddl'))},
                "config": {"game_name": domain, "problem_index": index,
                           "difficulty": annotation["difficulty"]}})
    return result, {str(label.relative_to(repo)): sha(label)}


def alf_inventory(repo):
    repo = Path(repo)
    source = repo / "data/alfworld/alfworld_tasks_suffix.json"
    tests = json.loads(source.read_text())
    test_map = {str(Path(r["gamefile"])): r for r in tests}
    if len(test_map) != len(tests):
        raise ValueError("duplicate ALFWorld test game")
    paths = sorted((repo / "data/alfworld/json_2.1.1/train").rglob("game.tw-pddl"))
    paths += [repo / p for p in test_map]
    result = []
    for path in paths:
        rel = str(path.relative_to(repo))
        traj = path.with_name("traj_data.json")
        data = json.loads(traj.read_text())
        task_type = data["task_type"]
        if task_type not in TYPES:
            continue
        if not json.loads(path.read_text()).get("solvable", False):
            if rel in test_map:
                raise ValueError(f"registered test game marked unsolvable: {path}")
            continue
        scene = data["scene"]["floor_plan"]
        result.append({"id": "alfworld:" + str(path.relative_to(repo / "data/alfworld/json_2.1.1").parent),
            "group": "alfworld:scene:" + scene, "scene": scene, "dataset": "alfworld",
            "task_type": task_type, "official_split": "test_134_valid_unseen" if rel in test_map else "train",
            "files": {rel: sha(path), str(traj.relative_to(repo)): sha(traj)},
            "config": {"env_kwargs": {"gamefile": rel}, "env_name": task_type,
                       "task_type": TYPES[task_type]}})
    if sum(r["official_split"] != "train" for r in result) != len(tests):
        raise ValueError("ALFWorld test inventory incomplete")
    return result, {str(source.relative_to(repo)): sha(source)}


def validate_split(rows, split):
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or set(split) != {"support", "fit", "dev", "test"}:
        raise ValueError("duplicate task identity or invalid partition names")
    ids, groups = set(), set()
    for name, selected in split.items():
        current = set(selected)
        if not selected or len(current) != len(selected) or not current <= by_id.keys():
            raise ValueError(f"empty/duplicate/unknown task IDs in {name}")
        grouped = {by_id[i]["group"] for i in selected}
        if ids & current or groups & grouped:
            raise ValueError(f"task/scene/problem leakage into {name}")
        ids |= current
        groups |= grouped


def make_split(rows, dataset, seed=71, alf_counts=(50, 80, 20), *, alf_heldout=0):
    rng = random.Random(seed)
    split = {k: [] for k in ("support", "fit", "dev", "test")}
    if dataset == "pddl":
        for domain in DOMAINS:
            pool = sorted((r for r in rows if r["domain"] == domain), key=lambda r: r["id"])
            rng.shuffle(pool)
            n = len(pool)
            counts = (2, n - 2 - 2 * (n // 5), n // 5, n // 5)
            start = 0
            for part, count in zip(split, counts):
                split[part] += [r["id"] for r in pool[start:start + count]]
                start += count
    else:
        split["test"] = [r["id"] for r in rows if r["official_split"] != "train"]
        forbidden = {r["group"] for r in rows if r["official_split"] != "train"}
        train = [r for r in rows if r["official_split"] == "train" and r["group"] not in forbidden]
        groups = sorted({r["group"] for r in train})
        rng.shuffle(groups)
        if len(groups) < 5:
            raise ValueError("ALFWorld needs independent training scenes")
        if alf_heldout:
            # A separate train-derived test cohort uses entirely reserved scenes.
            # The existing unseen test never becomes a mixed-distribution score.
            n = len(groups)
            a, b, c = max(1, n // 10), max(2, 6*n // 10), max(3, 8*n // 10)
            partitions = (groups[:a], groups[a:b], groups[b:c], groups[c:])
            parts = ("support", "fit", "dev", "test")
            counts = (*alf_counts, alf_heldout)
        else:
            a, b = max(1, len(groups) // 5), max(2, 4 * len(groups) // 5)
            partitions = (groups[:a], groups[a:b], groups[b:])
            parts, counts = ("support", "fit", "dev"), alf_counts
        for part, allowed, count in zip(parts, partitions, counts):
            pools = defaultdict(list)
            for row in train:
                if row["group"] in allowed:
                    pools[row["task_type"]].append(row)
            for pool in pools.values():
                rng.shuffle(pool)
            # Round robin task types avoids concentrating the small pilot in one type.
            selected = []
            while any(pools.values()) and len(selected) < count:
                for kind in sorted(pools):
                    if pools[kind] and len(selected) < count:
                        selected.append(pools[kind].pop()["id"])
            if len(selected) != count:
                raise ValueError(f"insufficient grouped ALFWorld data for {part}: {len(selected)} < {count}")
            split[part] += selected
    validate_split(rows, split)
    return {k: sorted(v) for k, v in split.items()}


def pddl_cross_validation(rows, seed=71):
    """Five outer folds; independent support/fit/dev inside each fold only."""
    by_domain = {}
    rng = random.Random(seed)
    for domain in DOMAINS:
        pool = sorted((r for r in rows if r["domain"] == domain), key=lambda r: r["id"])
        if len(pool) % 5:
            raise ValueError("registered PDDL domain count must be divisible by five")
        rng.shuffle(pool)
        by_domain[domain] = pool
    folds = []
    for fold in range(5):
        split = {k: [] for k in ("support", "fit", "dev", "test")}
        for domain, pool in by_domain.items():
            width = len(pool)//5
            test = pool[fold*width:(fold+1)*width]
            remaining = [r for r in pool if r not in test]
            random.Random(int(digest([seed, domain, fold]), 16)).shuffle(remaining)
            split["test"] += [r["id"] for r in test]
            split["support"] += [r["id"] for r in remaining[:2]]
            split["dev"] += [r["id"] for r in remaining[2:4]]
            split["fit"] += [r["id"] for r in remaining[4:]]
        split = {k: sorted(v) for k, v in split.items()}
        validate_split(rows, split)
        folds.append(split)
    tests = sum((f["test"] for f in folds), [])
    if len(tests) != len(set(tests)) or set(tests) != {r["id"] for r in rows}:
        raise ValueError("PDDL outer test folds must cover each registered problem exactly once")
    return folds


def inventory(repo, datasets, seed=71, alf_counts=(50, 80, 20), *, alf_heldout=0, pddl_folds=1):
    experiments = {}
    for name in datasets:
        rows, sources = (pddl_inventory if name == "pddl" else alf_inventory)(repo)
        split = make_split(rows, name, seed, alf_counts, alf_heldout=alf_heldout)
        chosen = set(sum(split.values(), []))
        experiments[name] = {"tasks": [r for r in rows if r["id"] in chosen], "split": split,
            "source_sha256": sources, "available_tasks": len(rows),
            "protocol": "registered60_grouped_holdout_not_full_official_test" if name == "pddl"
                        else "train_scene_grouped_support_fit_dev_existing134_valid_unseen_test"}
        if name == "pddl" and pddl_folds == 5:
            folds = pddl_cross_validation(rows, seed)
            experiments[name].update({"tasks": rows, "split": folds[0], "cv_splits": folds,
                                     "protocol": "registered60_nested5fold_out_of_fold_success"})
        if name == "alfworld" and alf_heldout:
            unseen = sorted(r["id"] for r in rows if r["official_split"] != "train")
            experiments[name].update({
                "protocol": "train_scene_grouped_plus_reserved_scene_test_and_separate134_unseen",
                "primary_test_ids": unseen,
                "test_cohorts": {"valid_unseen": unseen,
                    "train_scene_holdout": sorted(set(split["test"]) - set(unseen))}})
    return experiments


def dependencies():
    return {m: importlib.util.find_spec(m) is not None for m in
            ("numpy", "torch", "transformers", "sentence_transformers", "alfworld",
             "textworld", "gym", "nltk", "chromadb", "openai", "fast_downward")}


def workload(experiments, k=6, repeats=2):
    result = {}
    for name, data in experiments.items():
        n = {part: len(ids) for part, ids in data["split"].items()}
        per_fold = dict(n)
        folds = len(data.get("cv_splits", [None]))
        n = {part: count*folds for part, count in n.items()}
        diagnostic = n["fit"] + n["dev"]
        result[name] = {"tasks": n, "candidate_cap": k, "diagnostic_repeats": repeats,
            "diagnostic_logical_use_drop_rows": diagnostic * k * repeats * 2,
            "diagnostic_schedules_upper_bound_shared_keep": diagnostic * (k + 1) * repeats,
            "test_schedules_upper_bound_shared_actions": n["test"] * (k + 3),
            "expert_episodes": diagnostic, "ce_original_forwards_upper_bound": diagnostic * (k + 1),
            "ce_augmented_forwards_upper_bound": 2 * n["fit"] * (k + 1),
            "note": "test actions KEEP + K SINGLE DROP + DROP ALL + at most one new CSM subset; LLM calls exceed schedules"}
        if "test_cohorts" in data:
            result[name]["test_cohorts"] = {k: len(v) for k, v in data["test_cohorts"].items()}
            result[name]["primary_test_tasks"] = len(data["primary_test_ids"])
        if folds > 1:
            result[name].update({"folds": folds, "per_fold_tasks": per_fold,
                "unique_registered_tasks": len(data["tasks"]),
                "task_count_semantics": "task usages summed over folds; each problem tested exactly once"})
    return result
