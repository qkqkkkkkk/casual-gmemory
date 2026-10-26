"""Episode adapters with explicit success and reset parity; old envs are intact."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types

from .interactive_data import digest


def native_pddl_type(repo):
    # Load the unchanged native implementation under a private package. Importing
    # tasks.envs would eagerly load FEVER data and ALFWorld even for PDDL alone.
    package = "_cmc_native_envs"
    quiet_native_imports()
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(Path(repo) / "tasks/envs")]
        sys.modules[package] = module
    return importlib.import_module(package + ".pddl_env.pddl_env").PDDLEnv


def quiet_native_imports():
    # Legacy mas.llm prints its API key at import. Keep it out of new logs.
    import contextlib
    import io
    import os
    os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    with contextlib.redirect_stdout(io.StringIO()):
        importlib.import_module("mas.llm")


def parse_native_action(native, text):
    """Native operator/argument rules with hyphens kept intact, without NLTK.

    The legacy regex fallback splits fill-shot and jack-up into separate words,
    preventing otherwise legal actions when punkt is unavailable. This override
    is confined to the new episode adapter and uses actual domain predicates.
    """
    from pddlgym.structs import Literal

    tokens = re.findall(r"[A-Za-z0-9_-]+", text.lower())
    valid = native.env.action_space.all_ground_literals(native.last_obs)
    predicates = {a.predicate.name.lower(): a.predicate for a in valid}
    name = next((t for t in tokens if t in predicates), None)
    if name is None:
        return None
    predicate = predicates[name]
    objects = {o.name.lower(): o for o in native.last_obs.objects}
    arguments = [objects[t] for t in tokens if t in objects]
    if len(arguments) < predicate.arity:
        return None
    return Literal(predicate, arguments[:predicate.arity])


def planner_inputs(domain, problem):
    """Remove duplicate constant declarations for strict Fast Downward parsing.

    Native tyreworld files list wrench/jack/pump both as domain constants and
    problem objects. Their identity and all facts stay unchanged. The original
    files and the environment are untouched; the expert records this transform.
    """
    def declarations(body):
        tokens = body.split()
        pending, groups, index = [], [], 0
        while index < len(tokens):
            if tokens[index] == "-":
                if not pending or index + 1 >= len(tokens):
                    raise ValueError("invalid typed object declaration")
                groups.append((pending, tokens[index+1]))
                pending, index = [], index+2
            else:
                pending.append(tokens[index])
                index += 1
        if pending:
            groups.append((pending, None))
        return groups

    constants = re.search(r"\(:constants\s+([^()]*)\)", domain, re.I)
    objects = re.search(r"\(:objects\s+([^()]*)\)", problem, re.I)
    removed = []
    if constants and objects:
        declared = {name.lower(): kind for names, kind in declarations(constants[1]) for name in names}
        kept = []
        for names, kind in declarations(objects[1]):
            for name in names:
                if name.lower() in declared:
                    if kind and declared[name.lower()] and kind.lower() != declared[name.lower()].lower():
                        raise ValueError("constant/object types disagree; cannot normalize planner input")
                    removed.append(name)
            names = [name for name in names if name.lower() not in declared]
            if names:
                kept.append(" ".join(names) + (" - " + kind if kind else ""))
        if removed:
            problem = problem[:objects.start(1)] + "\n" + "\n".join(kept) + "\n" + problem[objects.end(1):]
    return domain, problem, {"version": "strict-planner-constant-dedup-v1", "removed_duplicate_objects": sorted(removed),
        "domain_text_digest": digest(domain), "problem_text_digest": digest(problem)}


def outcome(*, dataset, raw_reward, done, won, steps, max_steps, invalid_actions=0):
    if not isinstance(won, bool) or not isinstance(done, bool):
        raise ValueError("explicit boolean won/done required")
    if won:
        reason = "success"
    elif done:
        reason = "failed_terminal"
    elif steps >= max_steps:
        reason = "step_limit"
    else:
        reason = "incomplete"
    return {"metric_schema": "interactive-success-v1", "success": float(won),
            "raw_reward": float(raw_reward), "progress": float(raw_reward) if dataset == "pddl" else None,
            "done": done, "won": won, "termination_reason": reason,
            "steps": steps, "invalid_actions": invalid_actions}


class Episode:
    def __init__(self, task, repo, max_steps=30, *, expert=False):
        self.task, self.repo, self.max_trials = task, Path(repo), max_steps
        self.dataset, self.expert = task["dataset"], expert
        self.initial_fingerprint = None
        self.trace, self.resets = [], []
        if self.dataset == "pddl":
            self.native = native_pddl_type(repo)({}, max_steps)
            self.task_main, _ = self.native.set_env(task["config"])
            self.native._text_to_action = types.MethodType(parse_native_action, self.native)
            actual = sorted(" ".join(self.native._literal_to_text(g).lower().split())
                            for g in self.native.goal_literals)
            expected = sorted(" ".join(g.lower().split()) for g in task["annotation_subgoals"])
            if actual != expected:
                raise ValueError(f"PDDL annotation/registered problem mismatch: {task['id']}")
            self.env = self.native.env
        else:
            import textworld
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredExpert

            wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
            extras = ["gamefile"]
            if expert:
                wrappers.append(AlfredExpert())
                extras.append("expert_plan")
            game = self.repo / task["config"]["env_kwargs"]["gamefile"]
            self.env = textworld.start(str(game), textworld.EnvInfos(won=True,
                admissible_commands=True, extras=extras), wrappers=wrappers)
        self.reset()

    def reset(self):
        self.trace = []
        self.steps = self.invalid_actions = 0
        self.done = self.won = False
        self.raw_reward = 0.0
        if self.dataset == "pddl":
            # The native reset uses last_obs when rendering deltas. Always render
            # a full initial observation, including on schedule's second reset.
            self.native.last_obs = None
            self.native.reset()
            obs = self.native._get_obs()
            self.task_main = self.native._get_goal()
            self.task_description = f"Here is your initial observation: {obs}\n**Here is your task: {self.task_main}"
            state = self.native.last_obs
            identity = {"objects": sorted(map(str, state.objects)),
                        "literals": sorted(map(str, state.literals)), "goal": sorted(map(str, state.goal.literals))}
            self.won = all(g in state.literals for g in state.goal.literals)
            self.native.won = self.native.done = self.won
            self.native.reward = 1.0 if self.won else 0.0
            self.done = self.won
            self.raw_reward = self.native.reward
        else:
            self.state = self.env.reset()
            obs = str(self.state.feedback).strip()
            match = re.search(r"Your task is to:\s*(.+)", obs)
            if not match:
                raise ValueError("ALFWorld reset did not expose a task goal")
            self.task_main = self.task["task_type"] + "-" + match.group(1).strip()
            self.task_description = obs
            identity = {"observation": obs, "commands": sorted(self.state.admissible_commands),
                        "game_sha256": self.task["files"][self.task["config"]["env_kwargs"]["gamefile"]]}
        fingerprint = digest(identity)
        if self.initial_fingerprint and fingerprint != self.initial_fingerprint:
            raise ValueError(f"episode reset changed initial state: {self.task['id']}")
        self.initial_fingerprint = fingerprint
        self.resets.append(fingerprint)
        return obs

    def process_action(self, action):
        if self.dataset == "pddl":
            return self.native.process_action(action)
        return action.strip().replace("<", "").replace(">", "").split("\n")[0].strip()

    def step(self, action):
        action = self.process_action(action)
        if self.won:
            return "The goal is satisfied.", 1, True
        if self.dataset == "pddl":
            obs, reward, done = self.native.step(action)
            self.won, self.raw_reward = bool(self.native.won), float(self.native.reward)
            valid = reward != -1
        else:
            valid = action in self.state.admissible_commands
            self.state, raw, done = self.env.step(action)
            obs = str(self.state.feedback)
            self.won = bool(self.state.won)
            self.raw_reward = float(raw)
            reward = -1 if not valid or "think:" in action or obs == "Nothing happens." else int(self.won)
        self.done = bool(done)
        self.steps += 1
        self.invalid_actions += int(not valid)
        self.trace.append({"step": self.steps, "action": action, "observation": obs,
                           "step_reward": float(reward), "raw_reward": self.raw_reward,
                           "done": self.done, "won": self.won, "valid_action": valid})
        return obs, reward, bool(self.done or self.won)

    def result(self):
        return outcome(dataset=self.dataset, raw_reward=self.raw_reward, done=self.done,
                       won=self.won, steps=self.steps, max_steps=self.max_trials,
                       invalid_actions=self.invalid_actions)

    def feedback(self):
        return float(self.won), self.won, "You successfully finished this task!" if self.won else "You failed the task."

    def close(self):
        self.env.close()


def make_expert(task, repo, *, max_steps=200, timeout=120):
    """Generate and verify one expert plan in a fresh, bounded Python process.

    Fast Downward copies and dlopens its C++ library, then manually dlcloses it.
    Repeating that lifecycle in one Linux process can crash inside LoadLibrary.
    Keep native-library state out of callers, including the unittest process.
    """
    with tempfile.TemporaryDirectory(prefix="cmc-expert-") as directory:
        result_path = Path(directory) / "result.json"
        command = [sys.executable, "-X", "faulthandler", "-m", __package__ + ".interactive_envs",
                   "--expert-output", str(result_path)]
        try:
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                input=json.dumps({"task": task, "repo": str(Path(repo).resolve()), "max_steps": max_steps}),
                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"expert subprocess timed out after {timeout}s: {task['id']}") from exc
        if completed.returncode:
            raise RuntimeError(f"expert subprocess failed: {task['id']}; returncode={completed.returncode}\n"
                               + completed.stdout[-4000:] + completed.stderr[-8000:])
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"expert subprocess produced no valid result: {task['id']}\n"
                               + completed.stderr[-8000:]) from exc
    if (not isinstance(result, dict) or result.get("task_id") != task["id"]
            or result.get("validation", {}).get("success") != 1
            or result.get("validation", {}).get("invalid_actions") != 0
            or not result.get("actions")):
        raise ValueError(f"expert subprocess produced an invalid verified plan: {task['id']}")
    result["expert_execution"]["stdout"] = completed.stdout
    result["expert_execution"]["stderr"] = completed.stderr
    return result


_EXPERT_STARTED = False


def _make_expert_once(task, repo, *, max_steps=200):
    """Single-use entry for the CLI or an existing bounded expert_one worker."""
    global _EXPERT_STARTED
    if _EXPERT_STARTED:
        raise RuntimeError("an expert process can handle only one task; call make_expert for a fresh process")
    _EXPERT_STARTED = True
    episode = Episode(task, repo, max_steps, expert=task["dataset"] == "alfworld")
    try:
        actions = []
        planner_normalization = None
        if task["dataset"] == "pddl":
            from fast_downward import Operator, close_lib, load_lib, pddl2sas

            domain = task["domain"]
            root = Path(repo) / "tasks/envs/pddl_env/pddlgym/pddl"
            problem = next(Path(repo) / p for p in task["files"] if Path(p).parent.name == domain)
            domain_text, problem_text, planner_normalization = planner_inputs(
                (root / (domain + ".pddl")).read_text(), problem.read_text())
            _, sas = pddl2sas(domain_text, problem_text)
            lib = load_lib()
            try:
                if not lib.solve_sas(sas.encode(), False):
                    raise ValueError(f"planner failed: {task['id']}")
                plan = (Operator * lib.get_last_plan_length())()
                lib.get_last_plan(plan)
                symbolic = [p.name for p in plan]
            finally:
                close_lib(lib)
            for command in symbolic:
                tokens = command.strip("()").split()
                valid = sorted(episode.native.env.action_space.all_ground_literals(episode.native.last_obs), key=str)
                literal = next((a for a in valid if a.predicate.name.lower() == tokens[0].lower()
                    and [v.name.lower() for v in a.variables] == [s.lower() for s in tokens[1:]]), None)
                if literal is None:
                    raise ValueError(f"planner returned an invalid action: {command}")
                action = episode.native._literal_to_text(literal)
                actions.append(action)
                _, _, done = episode.step(action)
                if episode.invalid_actions:
                    raise ValueError(f"planner action failed native replay: {command}; rendered={action}")
                if done:
                    break
        else:
            for _ in range(max_steps):
                plan = episode.state["extra.expert_plan"]
                if not plan:
                    raise ValueError(f"expert returned no action: {task['id']}")
                action = plan[0]
                actions.append(action)
                _, _, done = episode.step(action)
                if done:
                    break
        if not actions or not episode.won or episode.invalid_actions:
            raise ValueError(f"expert trajectory failed validation: {task['id']}")
        return {"task_id": task["id"], "source": "fast_downward_symbolic" if task["dataset"] == "pddl"
                else "alfworld_handcoded_text_expert", "task_files": task["files"],
                "initial_fingerprint": episode.initial_fingerprint,
                "query": episode.task_main, "state": episode.task_description,
                "actions": actions, "target": "\n".join(actions), "trajectory": episode.trace,
                "validation": episode.result(), "supervision": "initial_public_context_full_successful_plan_ce_v1",
                "planner_input_normalization": planner_normalization,
                "expert_execution": {"process_policy": "one_expert_task_per_process_v1", "pid": os.getpid()},
                "hidden_expert_state_exposed_at_runtime": False}
    finally:
        episode.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="One isolated, verified expert episode")
    parser.add_argument("--expert-output", required=True, type=Path)
    arguments = parser.parse_args()
    request = json.load(sys.stdin)
    verified = _make_expert_once(request["task"], request["repo"], max_steps=request["max_steps"])
    arguments.expert_output.write_text(json.dumps(verified, ensure_ascii=False, allow_nan=False))
