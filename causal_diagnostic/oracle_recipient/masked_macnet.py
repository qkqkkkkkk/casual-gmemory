"""MacNet scheduler with one prompt-level GMemory candidate mask.

The original scheduler is preserved structurally.  The only experimental
change is that one retrieved trajectory or insight can be omitted from a
specified worker's prompt.  For receiver-level DROP, the decision node keeps
the full memory prompt; for GLOBAL_DROP it receives the pruned prompt too.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable

from mas.llm import Message
from mas.memory.common import AgentMessage, MASMessage
from tasks.mas_workflow.format import (
    format_task_context,
    format_task_prompt_with_insights,
)
from tasks.mas_workflow.macnet.graph_mas import MacNet


@dataclass(frozen=True)
class MemoryMaskPolicy:
    condition: str
    candidate_kind: str
    candidate_index: int
    drop_recipients: frozenset[str] = frozenset()
    drop_from_decision: bool = False

    def validate(self, worker_ids: Iterable[str]) -> None:
        workers = set(worker_ids)
        if self.candidate_kind not in {"trajectory", "insight"}:
            raise ValueError(f"unsupported candidate kind: {self.candidate_kind}")
        if self.candidate_index < 0:
            raise ValueError("candidate_index must be non-negative")
        unknown = set(self.drop_recipients) - workers
        if unknown:
            raise ValueError(f"unknown drop recipients: {sorted(unknown)}")


def _without_index(values: list[Any], index: int, label: str) -> list[Any]:
    if not 0 <= index < len(values):
        raise IndexError(
            f"{label} candidate index {index} is unavailable; found {len(values)}"
        )
    return [value for position, value in enumerate(values) if position != index]


class RecipientMaskedMacNet(MacNet):
    """Native MacNet with a branch-specific candidate exposure policy."""

    def set_memory_policy(self, policy: MemoryMaskPolicy) -> None:
        policy.validate(node.id for node in self._agent_nodes.values())
        self.memory_policy = policy
        self.execution_trace: list[dict[str, Any]] = []

    def set_sampling_temperature(self, temperature: float) -> None:
        for node in (*self._agent_nodes.values(), self._decision_node):
            node.reasoning_config.temperature = float(temperature)

    def _memory_for(
        self,
        successful: list[MASMessage],
        insights: list[str],
        *,
        recipient: str | None,
        decision: bool = False,
    ) -> tuple[list[MASMessage], list[str], bool]:
        policy = self.memory_policy
        should_drop = policy.drop_from_decision if decision else recipient in policy.drop_recipients
        selected_successful = list(successful)
        selected_insights = list(insights)
        if should_drop and policy.candidate_kind == "trajectory":
            selected_successful = _without_index(
                selected_successful, policy.candidate_index, "trajectory"
            )
        if should_drop and policy.candidate_kind == "insight":
            selected_insights = _without_index(
                selected_insights, policy.candidate_index, "insight"
            )
        return selected_successful, selected_insights, not should_drop

    @staticmethod
    def _render_prompt(
        few_shots: list[str],
        successful: list[MASMessage],
        insights: list[str],
        task_description: str,
    ) -> str:
        successful_shots = [
            format_task_context(
                trajectory.task_description,
                trajectory.task_trajectory,
                trajectory.get_extra_field("key_steps"),
            )
            for trajectory in successful
        ]
        return format_task_prompt_with_insights(
            few_shots=few_shots,
            memory_few_shots=successful_shots,
            insights=insights,
            task_description=task_description,
        )

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def schedule(self, task_config: dict) -> tuple[float, bool]:
        if not hasattr(self, "memory_policy"):
            raise RuntimeError("set_memory_policy must be called before schedule")
        if task_config.get("task_main") is None:
            raise ValueError("Missing required key task_main")
        if task_config.get("task_description") is None:
            raise ValueError("Missing required key task_description")

        task_main = str(task_config["task_main"])
        task_description = str(task_config["task_description"])
        few_shots = list(task_config.get("few_shots", []))
        env = self.env
        env.reset()
        self.meta_memory.init_task_context(task_main, task_description)
        successful, _, insights = self.meta_memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold,
        )
        successful = list(successful)
        insights = list(insights)

        # Validate candidate existence even in USE_ALL, where no list is pruned.
        candidate_values = successful if self.memory_policy.candidate_kind == "trajectory" else insights
        if not 0 <= self.memory_policy.candidate_index < len(candidate_values):
            raise IndexError(
                f"candidate index {self.memory_policy.candidate_index} is unavailable; "
                f"found {len(candidate_values)} {self.memory_policy.candidate_kind} values"
            )

        observer_prompt = self._render_prompt(
            few_shots, successful, insights, self.meta_memory.summarize(upstream_agent_ids=None)
        )
        self.notify_observers(observer_prompt)

        for step_index in range(env.max_trials):
            upstream_node_ids: dict[str, str] = {}
            in_degree = {
                node.id: len(node.spatial_predecessors)
                for node in self._agent_nodes.values()
            }
            queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
            step_trace: dict[str, Any] = {
                "step_index": step_index,
                "workers": {},
            }
            decision_full_prompt: str | None = None
            decision_pruned_prompt: str | None = None
            decision_exposed = True

            while queue:
                node_id = queue.pop(0)
                node = self._find_agent_node_by_uuid(node_id)
                current_description = self.meta_memory.summarize(
                    upstream_agent_ids=None
                )
                node_successful, node_insights, exposed = self._memory_for(
                    successful, insights, recipient=node.id
                )
                prompt = self._render_prompt(
                    few_shots,
                    node_successful,
                    node_insights,
                    current_description,
                )
                # The upstream scheduler passes the final worker's user message
                # to the decision node. Capture both full and globally-pruned
                # versions at exactly that point in the state trajectory.
                decision_full_prompt = self._render_prompt(
                    few_shots, successful, insights, current_description
                )
                decision_successful, decision_insights, decision_exposed = (
                    self._memory_for(
                        successful, insights, recipient=None, decision=True
                    )
                )
                decision_pruned_prompt = self._render_prompt(
                    few_shots,
                    decision_successful,
                    decision_insights,
                    current_description,
                )
                message = Message("user", prompt)
                raw_action = ""
                last_error: Exception | None = None
                for _ in range(3):
                    try:
                        raw_action = node.execute(message, use_critic=self._use_critic)
                        if raw_action:
                            break
                    except Exception as exc:  # pragma: no cover - model/runtime path
                        last_error = exc
                if not raw_action:
                    raise RuntimeError(f"worker {node.id} failed to produce an action") from last_error
                action = self.env.process_action(raw_action)
                agent_message = AgentMessage(
                    agent_name=node._agent.name,
                    system_instruction=node._agent.system_instruction,
                    user_instruction=prompt,
                    message=action,
                )
                upstream_ids = []
                for predecessor in node.spatial_predecessors:
                    if predecessor.id not in upstream_node_ids:
                        raise ValueError("upstream node was not executed")
                    upstream_ids.append(upstream_node_ids[predecessor.id])
                current_id = self.meta_memory.add_agent_node(
                    agent_message, upstream_agent_ids=upstream_ids
                )
                upstream_node_ids[node.id] = current_id
                step_trace["workers"][node.id] = {
                    "candidate_exposed": exposed,
                    "prompt_hash": self._prompt_hash(prompt),
                    "raw_output": raw_action,
                    "processed_action": action,
                }
                for successor in node.spatial_successors:
                    if successor not in self._agent_nodes.values():
                        raise RuntimeError("MacNet graph contains a non-worker successor")
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        queue.append(successor.id)

            self._update_memory()
            if decision_full_prompt is None or decision_pruned_prompt is None:
                raise RuntimeError("MacNet executed no worker nodes")
            decision_prompt = (
                decision_pruned_prompt
                if self.memory_policy.drop_from_decision
                else decision_full_prompt
            )
            decision_message = Message("user", decision_prompt)
            self._connect_decision_node()
            try:
                raw_decision = self._decision_node.execute(
                    decision_message, use_critic=False
                )
            finally:
                self._disconnect_dicision_node()
            action = env.process_action(raw_decision)
            observation, reward, done = env.step(action)
            self.notify_observers(f"Act {step_index + 1}: {action}\nObs {step_index + 1}: {observation}")
            self.meta_memory.move_memory_state(action, observation, reward=reward)
            step_trace["decision"] = {
                "candidate_exposed": decision_exposed,
                "prompt_hash": self._prompt_hash(decision_prompt),
                "raw_output": raw_decision,
                "processed_action": action,
                "observation": observation,
                "reward": float(reward),
                "done": bool(done),
            }
            self.execution_trace.append(step_trace)
            if done:
                break

        final_reward, final_done, feedback = self.env.feedback()
        self.notify_observers(feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=feedback)
        self.meta_memory.backward(final_done)
        return final_reward, final_done
