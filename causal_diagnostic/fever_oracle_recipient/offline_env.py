"""One-step, network-free FEVER binary classification environment."""

from __future__ import annotations

import re
from typing import Any

from .data import BINARY_LABELS


class OfflineFeverEnv:
    """Score an exact SUPPORTS/REFUTES label without Wikipedia access."""

    def __init__(
        self, env_config: dict[str, Any] | None = None, max_trials: int = 1
    ) -> None:
        if int(max_trials) != 1:
            raise ValueError("OfflineFeverEnv is a one-step environment")
        self.env_config = dict(env_config or {})
        self.max_trials = 1
        self.config: dict[str, Any] | None = None
        self.reset()

    def set_env(self, task_config: dict[str, Any]) -> tuple[str, str]:
        label = str(task_config.get("label", "")).upper()
        claim = str(task_config.get("claim", "")).strip()
        if label not in BINARY_LABELS:
            raise ValueError("task label must be SUPPORTS or REFUTES")
        if not claim:
            raise ValueError("task claim must be non-empty")
        self.config = dict(task_config)
        self.config.update(label=label, claim=claim)
        task_main = f"FEVER claim: {claim}"
        task_description = (
            f"Classify this claim as SUPPORTS or REFUTES.\nClaim: {claim}\n"
            "Return exactly Finish[SUPPORTS] or Finish[REFUTES]."
        )
        return task_main, task_description

    def reset(self) -> None:
        self.reward = 0.0
        self.done = False
        self.last_prediction: str | None = None
        self.infos = {"steps": 0}

    @classmethod
    def process_action(cls, action: str) -> str:
        text = str(action).strip().replace("<", "").replace(">", "")
        explicit = re.findall(
            r"Finish\s*\[\s*(SUPPORTS|REFUTES)\s*\]", text, flags=re.IGNORECASE
        )
        labels = explicit or re.findall(
            r"\b(SUPPORTS|REFUTES)\b", text, flags=re.IGNORECASE
        )
        unique = {value.upper() for value in labels}
        if len(unique) == 1:
            label = unique.pop()
            evidence = re.findall(
                r"Evidence\s*\[([^\]]*)\]", text, flags=re.IGNORECASE
            )
            if evidence:
                pages = " | ".join(
                    value.strip()
                    for value in evidence[-1].split("|")
                    if value.strip()
                )
                return f"Evidence[{pages}]\nFinish[{label}]"
            return f"Finish[{label}]"
        return text.splitlines()[0].strip() if text else ""

    @staticmethod
    def _prediction(action: str) -> str | None:
        match = re.search(r"Finish\[(SUPPORTS|REFUTES)\]", action)
        return match.group(1) if match else None

    def step(self, action: str) -> tuple[str, float, bool]:
        if self.config is None:
            raise RuntimeError("set_env must be called before step")
        normalized = self.process_action(action)
        prediction = self._prediction(normalized)
        self.infos["steps"] += 1
        self.last_prediction = prediction
        self.reward = float(prediction == self.config["label"])
        self.done = True
        if prediction is None:
            observation = "Invalid final label. Expected Finish[SUPPORTS] or Finish[REFUTES]."
        elif self.reward == 1.0:
            observation = "Answer is CORRECT."
        else:
            observation = "Answer is INCORRECT."
        return observation, self.reward, self.done

    def feedback(self) -> tuple[float, bool, str]:
        feedback = (
            "The FEVER label was correct."
            if self.reward == 1.0
            else "The FEVER label was incorrect."
        )
        return self.reward, bool(self.reward), feedback
