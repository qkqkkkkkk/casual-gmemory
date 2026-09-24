"""One-step, network-free HotpotQA environment with official answer metrics."""

from __future__ import annotations

import re
from typing import Any

from .metrics import score_answer


class OfflineHotpotQAEnv:
    def __init__(
        self, env_config: dict[str, Any] | None = None, max_trials: int = 1
    ) -> None:
        if int(max_trials) != 1:
            raise ValueError("OfflineHotpotQAEnv is a one-step environment")
        self.env_config = dict(env_config or {})
        self.max_trials = 1
        self.config: dict[str, Any] | None = None
        self.reset()

    def set_env(self, task_config: dict[str, Any]) -> tuple[str, str]:
        question = str(task_config.get("question", "")).strip()
        answer = str(task_config.get("answer", task_config.get("label", ""))).strip()
        if not question or not answer:
            raise ValueError("task question and answer must be non-empty")
        self.config = dict(task_config)
        self.config.update(question=question, answer=answer)
        return str(task_config["task_main"]), str(task_config["task_description"])

    def reset(self) -> None:
        self.reward = 0.0
        self.done = False
        self.last_prediction: str | None = None
        self.last_metrics: dict[str, Any] | None = None
        self.infos = {"steps": 0}

    @classmethod
    def process_action(cls, action: str) -> str:
        text = str(action).strip().replace("<", "").replace(">", "")
        matches = re.findall(r"Finish\s*\[([^\]]*)\]", text, flags=re.IGNORECASE)
        if not matches:
            return text.splitlines()[0].strip() if text else ""
        answer = matches[-1].strip()
        evidence = re.findall(r"Evidence\s*\[([^\]]*)\]", text, flags=re.IGNORECASE)
        if evidence:
            return f"Evidence[{evidence[-1].strip()}]\nFinish[{answer}]"
        return f"Finish[{answer}]"

    @staticmethod
    def _prediction(action: str) -> str | None:
        matches = re.findall(r"Finish\[([^\]]*)\]", str(action), flags=re.IGNORECASE)
        return matches[-1].strip() if matches and matches[-1].strip() else None

    def step(self, action: str) -> tuple[str, float, bool]:
        if self.config is None:
            raise RuntimeError("set_env must be called before step")
        normalized = self.process_action(action)
        prediction = self._prediction(normalized)
        self.infos["steps"] += 1
        self.last_prediction = prediction
        self.last_metrics = score_answer(prediction or "", self.config["answer"])
        self.reward = float(self.last_metrics["f1"])
        self.done = True
        if prediction is None:
            observation = "Invalid final answer. Expected Finish[answer]."
        else:
            observation = (
                f"Answer EM={self.last_metrics['em']:.0f}, "
                f"F1={self.last_metrics['f1']:.4f}."
            )
        return observation, self.reward, True

    def feedback(self) -> tuple[float, bool, str]:
        metrics = self.last_metrics or score_answer("", self.config["answer"] if self.config else "")
        exact = bool(metrics["em"])
        feedback = (
            f"HotpotQA answer EM={metrics['em']:.0f}, F1={metrics['f1']:.4f}."
        )
        return float(metrics["f1"]), exact, feedback
