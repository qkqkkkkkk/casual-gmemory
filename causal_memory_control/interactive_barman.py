"""Construct a symbolic barman plan; native replay remains the success test.

Use each destination shot to measure its recipe's two ingredients. Clean it
after each pour, mix in the shaker, then serve back into the clean destination.
Completed destination shots are never touched again. This avoids searching the
many equivalent permutations of spare shots and unrelated cocktails.
"""
from __future__ import annotations


def symbolic_plan(native):
    facts = {(literal.predicate.name, tuple(v.name for v in literal.variables))
             for literal in native.last_obs.literals}
    goals = sorted((literal.predicate.name, tuple(v.name for v in literal.variables))
                   for literal in native.goal_literals)
    if not goals or any(name != "contains" or len(args) != 2 for name, args in goals):
        raise ValueError("barman constructive expert requires positive contains goals")
    hands = sorted(args[0] for name, args in facts if name == "handempty")
    if len(hands) < 2:
        raise ValueError("barman constructive expert needs two initially free hands")
    hand, other = hands[:2]
    dispensers = {args[1]: args[0] for name, args in sorted(facts) if name == "dispenses"}
    parts = [{args[0]: args[1] for name, args in facts if name == key}
             for key in ("cocktail-part1", "cocktail-part2")]
    levels = sorted(args for name, args in facts if name == "shaker-empty-level")
    if not levels:
        raise ValueError("barman problem has no shaker")
    shaker, empty_level = levels[0]
    next_level = {args[0]: args[1] for name, args in facts if name == "next"}
    first_level = next_level[empty_level]
    second_level = next_level[first_level]
    for name in ("ontable", "clean", "empty"):
        if (name, (shaker,)) not in facts:
            raise ValueError("barman constructive expert needs an empty clean shaker on the table")
    if ("shaker-level", (shaker, empty_level)) not in facts:
        raise ValueError("barman shaker starts at an unexpected level")
    plan, destinations = [], set()

    def action(name, *args):
        plan.append(" ".join((name, *args)))

    for _, (shot, beverage) in goals:
        if ("contains", (shot, beverage)) in facts:
            continue
        if shot in destinations:
            raise ValueError("barman goals request different beverages in the same shot")
        destinations.add(shot)
        if any((name, (shot,)) not in facts for name in ("ontable", "clean", "empty")):
            raise ValueError("barman destination must start empty, clean and on the table")
        action("grasp", hand, shot)
        if beverage in dispensers:
            action("fill-shot", shot, beverage, hand, other, dispensers[beverage])
            action("leave", hand, shot)
            continue
        try:
            ingredient1, ingredient2 = parts[0][beverage], parts[1][beverage]
        except KeyError as exc:
            raise ValueError(f"barman goal lacks a complete recipe: {beverage}") from exc
        for ingredient, pour, level, after in (
                (ingredient1, "pour-shot-to-clean-shaker", empty_level, first_level),
                (ingredient2, "pour-shot-to-used-shaker", first_level, second_level)):
            action("fill-shot", shot, ingredient, hand, other, dispensers[ingredient])
            action(pour, shot, ingredient, shaker, hand, level, after)
            action("clean-shot", shot, ingredient, hand, other)
        action("leave", hand, shot)
        action("grasp", hand, shaker)
        action("shake", beverage, ingredient1, ingredient2, shaker, hand, other)
        action("pour-shaker-to-shot", beverage, shot, hand, shaker, second_level, first_level)
        action("empty-shaker", hand, shaker, beverage, first_level, empty_level)
        action("clean-shaker", hand, other, shaker)
        action("leave", hand, shaker)
    return plan
