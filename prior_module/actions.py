"""Canonical environment actions and a configurable semantic distance."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Mapping, Optional


@dataclass(frozen=True)
class CanonicalAction:
    tool_name: Optional[str]
    action_type: Optional[str]
    target_object: Optional[str]
    destination: Optional[str]
    arguments: tuple[tuple[str, str], ...] = ()
    parse_status: str = "valid"
    raw_action: str = ""

    def key(self) -> tuple[Any, ...]:
        return (
            self.tool_name,
            self.action_type,
            self.target_object,
            self.destination,
            self.arguments,
        )


@dataclass(frozen=True)
class DistanceResult:
    distance: float
    reason: str
    left: CanonicalAction
    right: CanonicalAction


EnvironmentActionParser = Callable[[str], Optional[CanonicalAction]]
CriticalChecker = Callable[
    [CanonicalAction, CanonicalAction, Optional[Mapping[str, Any]]], bool
]


class ActionCanonicalizer:
    """Parse ALFWorld-style actions or delegate to an environment parser.

    The environment parser is preferred because only it knows the authoritative
    object identifiers and aliases. The built-in parser is deliberately narrow:
    it is a safe fallback for common textual action formats, not a replacement
    for an environment grammar.
    """

    _ACTION_ALIASES = {
        "navigate": "go",
        "move": "go",
        "pick up": "take",
        "look at": "examine",
        "turn on": "toggle_on",
        "turn off": "toggle_off",
        "done": "finish",
    }

    def __init__(self, environment_parser: Optional[EnvironmentActionParser] = None):
        self._environment_parser = environment_parser

    def canonicalize(self, raw_action: str) -> CanonicalAction:
        raw_action = raw_action or ""
        if self._environment_parser is not None:
            parsed = self._environment_parser(raw_action)
            if parsed is not None:
                return parsed
        return self._parse_fallback(raw_action)

    @staticmethod
    def _normalize_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = re.sub(r"\s+", " ", value.strip().lower())
        value = re.sub(r"^(?:the|a|an)\s+", "", value)
        return value or None

    def _build(
        self,
        raw_action: str,
        action_type: str,
        target: Optional[str] = None,
        destination: Optional[str] = None,
        arguments: Optional[Mapping[str, str]] = None,
    ) -> CanonicalAction:
        canonical_type = self._ACTION_ALIASES.get(action_type, action_type)
        pairs = tuple(
            sorted(
                (self._normalize_text(key) or "", self._normalize_text(value) or "")
                for key, value in (arguments or {}).items()
            )
        )
        return CanonicalAction(
            tool_name=None,
            action_type=canonical_type,
            target_object=self._normalize_text(target),
            destination=self._normalize_text(destination),
            arguments=pairs,
            raw_action=raw_action,
        )

    def _parse_fallback(self, raw_action: str) -> CanonicalAction:
        text = self._normalize_text(raw_action) or ""
        text = re.sub(r"^(action|command)\s*:\s*", "", text)

        match = re.fullmatch(r"(?:go|navigate|move)\s+to\s+(.+)", text)
        if match:
            return self._build(raw_action, "go", destination=match.group(1))

        match = re.fullmatch(r"(open|close|examine|look at)\s+(.+)", text)
        if match:
            return self._build(raw_action, match.group(1), target=match.group(2))

        match = re.fullmatch(r"(?:take|pick up)\s+(.+?)(?:\s+from\s+(.+))?", text)
        if match:
            arguments = {"source": match.group(2)} if match.group(2) else {}
            return self._build(raw_action, "take", target=match.group(1), arguments=arguments)

        match = re.fullmatch(r"put\s+(.+?)\s+(?:in|into|on|onto)\s+(.+)", text)
        if match:
            return self._build(
                raw_action, "put", target=match.group(1), destination=match.group(2)
            )

        match = re.fullmatch(r"(heat|cool|clean)\s+(.+?)(?:\s+(?:with|using|in)\s+(.+))?", text)
        if match:
            return self._build(
                raw_action,
                match.group(1),
                target=match.group(2),
                destination=match.group(3),
            )

        match = re.fullmatch(r"(?:turn on|turn off|toggle)\s+(.+)", text)
        if match:
            action = "toggle"
            if text.startswith("turn on"):
                action = "toggle_on"
            elif text.startswith("turn off"):
                action = "toggle_off"
            return self._build(raw_action, action, target=match.group(1))

        if text in {"finish", "done", "submit", "terminate"}:
            return self._build(raw_action, text)

        return CanonicalAction(
            tool_name=None,
            action_type=None,
            target_object=None,
            destination=None,
            parse_status="invalid",
            raw_action=raw_action,
        )


@dataclass
class ActionDistancePolicy:
    """Domain-aware distance for two canonical executable actions."""

    material_threshold: float = 0.5
    invalid_distance: float = 0.5
    irreversible_action_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"finish", "submit", "terminate", "delete", "consume"}
        )
    )
    critical_checker: Optional[CriticalChecker] = None

    def distance(
        self,
        left: CanonicalAction,
        right: CanonicalAction,
        context: Optional[Mapping[str, Any]] = None,
    ) -> DistanceResult:
        if left.parse_status != "valid" or right.parse_status != "valid":
            if left.raw_action.strip() == right.raw_action.strip():
                return DistanceResult(0.0, "same_unparsed_text", left, right)
            return DistanceResult(self.invalid_distance, "unparsed_action", left, right)

        if left.key() == right.key():
            return DistanceResult(0.0, "same_canonical_action", left, right)

        if self._is_critical_difference(left, right, context):
            return DistanceResult(1.0, "critical_or_irreversible_difference", left, right)

        if left.tool_name != right.tool_name or left.action_type != right.action_type:
            return DistanceResult(0.75, "different_tool_or_action_type", left, right)

        if (
            left.target_object != right.target_object
            or left.destination != right.destination
        ):
            return DistanceResult(0.5, "different_target_or_destination", left, right)

        return DistanceResult(0.25, "noncritical_argument_difference", left, right)

    def _is_critical_difference(
        self,
        left: CanonicalAction,
        right: CanonicalAction,
        context: Optional[Mapping[str, Any]],
    ) -> bool:
        if left.action_type in self.irreversible_action_types:
            return True
        if right.action_type in self.irreversible_action_types:
            return True
        if self.critical_checker is not None:
            return self.critical_checker(left, right, context)

        context = context or {}
        irreversible = set(context.get("irreversible_action_types", ()))
        if left.action_type in irreversible or right.action_type in irreversible:
            return True

        critical_objects = {
            self._normalize(value) for value in context.get("critical_objects", ())
        }
        changed_objects = {left.target_object, right.target_object}
        return bool(critical_objects.intersection(changed_objects))

    @staticmethod
    def _normalize(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value).strip().lower())
