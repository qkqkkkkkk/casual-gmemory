"""Construct tyreworld plans from symbolic facts, then require native replay.

Only goal-relevant hubs are serviced. The registered problems contain many
irrelevant wheels/hubs; general search needlessly explores their permutations.
This is a deterministic expert, not a shortest-plan or Fast Downward result.
"""
from __future__ import annotations


def symbolic_plan(native):
    facts = {(literal.predicate.name, tuple(v.name for v in literal.variables))
             for literal in native.last_obs.literals}
    goals = {(literal.predicate.name, tuple(v.name for v in literal.variables))
             for literal in native.goal_literals}
    if not goals or any(name not in ("on", "in", "inflated") for name, _ in goals):
        raise ValueError("tyreworld constructive expert supports positive on/in/inflated goals")
    plan = []

    def action(name, *args, add=(), remove=()):
        plan.append(" ".join((name, *args)))
        facts.difference_update(remove)
        facts.update(add)

    def ensure_open(container):
        if ("isopen", (container,)) in facts:
            return
        if any((p, (container,)) not in facts for p in ("closed", "unlocked")):
            raise ValueError(f"tyreworld container cannot be opened: {container}")
        action("open", container, add=[("isopen", (container,))], remove=[("closed", (container,))])

    def fetch(obj):
        if ("have", (obj,)) in facts:
            return
        containers = sorted(args[1] for name, args in facts if name == "in" and args[0] == obj)
        if len(containers) != 1:
            raise ValueError(f"tyreworld object is not uniquely accessible: {obj}")
        container = containers[0]
        ensure_open(container)
        action("fetch", obj, container, add=[("have", (obj,))], remove=[("in", (obj, container))])

    replacements = {}
    for name, (wheel, hub) in sorted((name, args) for name, args in goals if name == "on"):
        if hub in replacements and replacements[hub] != wheel:
            raise ValueError("tyreworld goals request two wheels on the same hub")
        replacements[hub] = wheel
    stored = {args[0]: args[1] for name, args in goals if name == "in"}
    mounted = {args[1]: args[0] for name, args in facts if name == "on"}
    hubs = sorted(hub for hub, wheel in mounted.items()
                  if (hub in replacements and replacements[hub] != wheel) or wheel in stored)
    for hub in hubs:
        old = mounted[hub]
        replacement = replacements.get(hub)
        if replacement == old:
            raise ValueError("tyreworld goal requires a wheel both mounted and stored")
        nuts = sorted({args[0] for name, args in facts if name in ("tight", "loose") and args[1] == hub})
        if len(nuts) != 1 or any((p, (hub,)) not in facts for p in ("on-ground", "fastened")):
            raise ValueError(f"tyreworld hub is not initially grounded and fastened: {hub}")
        nut = nuts[0]
        fetch("wrench")
        fetch("jack")
        if ("tight", (nut, hub)) in facts:
            action("loosen", nut, hub, add=[("loose", (nut, hub))], remove=[("tight", (nut, hub))])
        action("jack-up", hub, add=[("not-on-ground", (hub,))],
               remove=[("on-ground", (hub,)), ("have", ("jack",))])
        action("undo", nut, hub, add=[("have", (nut,)), ("unfastened", (hub,))],
               remove=[("fastened", (hub,)), ("loose", (nut, hub))])
        action("remove-wheel", old, hub, add=[("have", (old,)), ("free", (hub,))],
               remove=[("on", (old, hub))])
        if replacement:
            fetch(replacement)
            action("put-on-wheel", replacement, hub, add=[("on", (replacement, hub))],
                   remove=[("have", (replacement,)), ("free", (hub,))])
            action("do-up", nut, hub, add=[("loose", (nut, hub)), ("fastened", (hub,))],
                   remove=[("have", (nut,)), ("unfastened", (hub,))])
        action("jack-down", hub, add=[("on-ground", (hub,)), ("have", ("jack",))],
               remove=[("not-on-ground", (hub,))])
        if replacement:
            action("tighten", nut, hub, add=[("tight", (nut, hub))], remove=[("loose", (nut, hub))])

    for name, args in sorted(goals):
        if (name, args) in facts:
            continue
        if name == "in":
            obj, container = args
            fetch(obj)
            ensure_open(container)
            action("put-away", obj, container, add=[("in", args)], remove=[("have", (obj,))])
        elif name == "inflated":
            wheel, = args
            fetch("pump")
            if any((p, (wheel,)) not in facts for p in ("intact", "not-inflated")):
                raise ValueError(f"tyreworld wheel cannot be inflated: {wheel}")
            action("inflate", wheel, add=[("inflated", args)], remove=[("not-inflated", args)])
    if not goals <= facts:
        raise ValueError("tyreworld constructive plan does not satisfy all symbolic goals")
    return plan
