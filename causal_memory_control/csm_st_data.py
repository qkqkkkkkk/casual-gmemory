"""Text/list data for the independent CSM-ST-on-GMemory experiment.

Old branch rewards are never used as CI labels. Only pre-exposure text, the
registered candidate lists and gold answers (for teacher forcing only) are read.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import tempfile

from tqdm import tqdm


SCHEMA = "csm-st-gmemory-lists-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def task_text(query, state):
    """The same outcome-free representation is used during fit and deployment."""
    query, state = str(query), str(state or query)
    return query if state == query else f"Task:\n{query}\n\nTask context:\n{state}"


def candidate_text(payload):
    """Match GMemoryRetrievalAdapter, including empty source descriptions."""
    if payload["kind"] == "insight":
        return str(payload.get("text") or "")
    description, steps, trajectory = (str(payload.get(key) or "")
                                     for key in ("task_description", "key_steps", "trajectory"))
    return "\n".join(part for part in (
        f"Source task: {description}" if description else "",
        f"Key steps: {steps}" if steps else "",
        f"Trajectory: {trajectory}" if trajectory else "") if part)


def split_ids(ids, fraction=0.2, seed=71):
    ids = sorted({str(v) for v in ids})
    if len(ids) < 5 or not 0 < fraction < 1:
        raise ValueError("CSM-ST requires >=5 diagnostic tasks and 0<dev fraction<1")
    random.Random(seed).shuffle(ids)
    count = max(1, min(len(ids) - 1, int(round(len(ids) * fraction))))
    return set(ids[count:]), set(ids[:count])


def extract_lists(sources, expected_ids, *, forbidden_ids=()):
    """Spool repeated branch text to scratch SQLite, retaining one list per task."""
    from .train_gate import _query

    expected = {str(v) for v in expected_ids}
    if expected & {str(v) for v in forbidden_ids}:
        raise ValueError("diagnostic tasks overlap support/final test")
    with tempfile.TemporaryDirectory(prefix="gmemory-csm-text-") as scratch:
        with sqlite3.connect(str(Path(scratch) / "lists.sqlite")) as db:
            db.execute("CREATE TABLE lists(task TEXT PRIMARY KEY, digest TEXT, payload TEXT)")
            for source in sources:
                with Path(source).open(encoding="utf-8") as handle:
                    for line in tqdm(handle, desc="Read CSM candidate lists", unit="branch"):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        task_id = str(row["task_id"])
                        if task_id not in expected:
                            raise ValueError(f"unexpected diagnostic task {task_id}")
                        task = row["task"]
                        gold = task.get("gold_label", task.get("gold_answer", task.get("label", task.get("answer"))))
                        answers = gold if isinstance(gold, list) else [gold]
                        if not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
                            raise ValueError(f"missing gold answer for CE supervision: {task_id}")
                        candidates = []
                        for raw in row["candidate_set"]:
                            content = candidate_text(raw)
                            if not content.strip():
                                raise ValueError("empty CSM candidate text")
                            candidates.append({"id": str(raw["candidate_id"]), "kind": raw["kind"],
                                "rank": int(raw["index"]) + 1, "text": content})
                        if not candidates or len({c["id"] for c in candidates}) != len(candidates):
                            raise ValueError("empty/duplicate candidate list")
                        payload = {"schema": SCHEMA, "id": task_id, "task_id": task_id,
                            "query": _query(task), "state": task.get("task_description", _query(task)),
                            "answers": answers, "candidates": candidates, "origin_task_ids": [task_id]}
                        identity = digest(payload)
                        old = db.execute("SELECT digest FROM lists WHERE task=?", (task_id,)).fetchone()
                        if old is not None and old[0] != identity:
                            raise ValueError(f"candidate/task context changed between branches: {task_id}")
                        if old is None:
                            db.execute("INSERT INTO lists VALUES (?,?,?)", (task_id, identity,
                                json.dumps(payload, ensure_ascii=False)))
                db.commit()
            result = [json.loads(r[0]) for r in db.execute("SELECT payload FROM lists ORDER BY task")]
    if {r["task_id"] for r in result} != expected:
        raise ValueError("diagnostic lists do not exactly cover registered tasks")
    return result


class Deduplicator:
    """Greedy, order-preserving dedup; fixed frozen encoder, never gold-aware."""
    def __init__(self, encoder=None, threshold=0.98):
        if not 0 < threshold <= 1:
            raise ValueError("dedup cosine threshold must be in (0,1]")
        self.encoder, self.threshold = encoder, threshold

    def indices(self, texts):
        import numpy as np

        kept, duplicates, normalized, vectors = [], {}, {}, {}
        for index, text in enumerate(texts):
            exact = " ".join(text.split())
            match = normalized.get(exact)
            if match is None and self.encoder is not None:
                vector = np.asarray(self.encoder.embed(text), dtype=float)
                norm = np.linalg.norm(vector)
                if not np.isfinite(vector).all() or norm == 0:
                    raise ValueError("invalid semantic dedup vector")
                vector = vector / norm
                match = next((k for k in kept if float(vector @ vectors[k]) >= self.threshold), None)
                vectors[index] = vector
            if match is None:
                kept.append(index)
                normalized[exact] = index
            else:
                duplicates[index] = match
        return kept, duplicates

    def apply(self, row):
        kept, duplicates = self.indices([c["text"] for c in row["candidates"]])
        return {**row, "candidates": [row["candidates"][i] for i in kept],
            "dedup_removed": {row["candidates"][i]["id"]: row["candidates"][j]["id"]
                              for i, j in duplicates.items()}}


def rarity(values, alpha=10.0):
    mean = sum(values) / len(values)
    return mean + alpha * math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def frequency_weights(rows, *, alpha=10.0, width=1.0):
    """Explicit histogram estimate of p(r), fitted on training lists only."""
    if width <= 0:
        raise ValueError("rarity histogram width must be positive")
    bins = [math.floor(rarity(row["ci"], alpha) / width) for row in rows]
    counts = Counter(bins)
    return [len(rows) / counts[b] for b in bins]


def intervention_plan(fit_rows, encoder, *, seed=2024, alpha=10.0, hard_threshold=5.0,
                      gamma=0.1, query_max_cosine=0.3, per_hard=1):
    """Eqs.5/10: fit-only donors; labels MUST be recomputed after mixing lists.

    A foreign context is not assumed harmful: that intuition is verified by CE.
    The seed, donor search bound and thresholds are explicit adaptation settings.
    """
    import numpy as np

    rng, result = random.Random(seed), []
    vectors = {}
    for row in fit_rows:
        vector = np.asarray(encoder.embed(row["query"]), dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(vector).all():
            raise ValueError("invalid query similarity vector")
        vectors[row["id"]] = vector / norm
    for source in fit_rows:
        if rarity(source["ci"], alpha) <= hard_threshold:
            continue
        informative = [c for c, value in zip(source["candidates"], source["ci"]) if value > gamma]
        if not informative:
            continue
        donors = [r for r in fit_rows if r["task_id"] != source["task_id"]]
        rng.shuffle(donors)
        donors = [r for r in donors[:256]
                  if float(vectors[source["id"]] @ vectors[r["id"]]) < query_max_cosine][:per_hard]
        for donor in donors:
            # Keep list size bounded; never evict the informative source items.
            width = max(len(source["candidates"]), len(informative) + 1)
            foreign = rng.sample(donor["candidates"], min(len(donor["candidates"]), width - len(informative)))
            for kind, anchor in (("positive", source), ("negative", donor)):
                candidates = [{**c, "id": f"mix-{i}-{digest(c)[:12]}"}
                              for i, c in enumerate([*informative, *foreign])]
                identity = f"mix-{kind}-{source['task_id']}-{donor['task_id']}"
                result.append({**{k: v for k, v in anchor.items() if k not in (
                    "ci", "ce", "dedup_removed", "input_digest", "label_type")},
                    "id": identity, "candidates": candidates,
                    "origin_task_ids": sorted({source["task_id"], donor["task_id"]}),
                    "intervention": kind})
    return result
