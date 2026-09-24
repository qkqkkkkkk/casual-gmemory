"""MacNet generation settings for the two-line HotpotQA protocol."""

from __future__ import annotations

from typing import Any


def configure_hotpotqa_sampling(mas: Any, temperature: float) -> None:
    for node in (*mas._agent_nodes.values(), mas._decision_node):
        node.reasoning_config.temperature = float(temperature)
        node.reasoning_config.stop_strs = None
