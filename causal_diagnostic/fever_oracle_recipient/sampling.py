"""MacNet sampling configuration specific to the FEVER output protocol."""

from __future__ import annotations

from typing import Any


def configure_fever_sampling(mas: Any, temperature: float) -> None:
    """Allow FEVER's Evidence line followed by its scored Finish line."""
    for node in (*mas._agent_nodes.values(), mas._decision_node):
        node.reasoning_config.temperature = float(temperature)
        # Native MacNet stops at the first newline. That would retain only
        # Evidence[...] and remove the Finish[...] label from every response.
        node.reasoning_config.stop_strs = None
