"""Seeded, persistent OpenAI-compatible client for matched causal branches."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, Sequence

from openai import OpenAI


class SeededCachedChat:
    """LLM callable compatible with ``ReasoningIO`` and GMemory.

    The upstream ``GPTChat`` does not send an inference seed.  Here the request
    seed is a stable function of the experiment seed and prompt, so identical
    prompts in matched branches receive identical stochastic draws regardless
    of call order. Responses are cached in SQLite for resume and branch reuse.
    """

    def __init__(
        self,
        model_name: str,
        cache_path: Path,
        *,
        experiment_seed: int,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model_name = model_name
        self.experiment_seed = int(experiment_seed)
        self.timeout = float(timeout)
        base_url = api_base or os.environ.get("OPENAI_API_BASE")
        if not base_url:
            raise RuntimeError("OPENAI_API_BASE is required")
        key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=self.timeout)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path), timeout=60)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        # Keep probability probes separate from the legacy text cache.  This
        # leaves existing v3 caches readable and makes resume safe when the
        # optional continuous-label outcome is enabled.
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS binary_scores "
            "(key TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
        )
        self.db.commit()
        self.calls = 0
        self.cache_hits = 0

    @staticmethod
    def _messages(messages: Sequence[Any]) -> list[dict[str, str]]:
        return [
            {
                "role": str(
                    message.role if hasattr(message, "role") else message["role"]
                ),
                "content": str(
                    message.content
                    if hasattr(message, "content")
                    else message["content"]
                ),
            }
            for message in messages
        ]

    def _request_seed(self, messages: Sequence[dict[str, str]]) -> int:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "experiment_seed": self.experiment_seed,
                    "model": self.model_name,
                    "messages": messages,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return int(digest[:12], 16) % 2_147_483_647

    def __call__(
        self,
        messages: Sequence[Any],
        temperature: float | None = 0.1,
        max_tokens: int | None = 512,
        stop_strs: Sequence[str] | None = None,
        num_comps: int | None = 1,
    ) -> str:
        rendered = self._messages(messages)
        request_seed = self._request_seed(rendered)
        payload = {
            "model": self.model_name,
            "messages": rendered,
            "temperature": 0.1 if temperature is None else float(temperature),
            "max_tokens": 512 if max_tokens is None else int(max_tokens),
            "stop": list(stop_strs) if stop_strs else None,
            "n": 1 if num_comps is None else int(num_comps),
            "seed": request_seed,
        }
        cache_key = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cached = self.db.execute(
            "SELECT response FROM responses WHERE key=?", (cache_key,)
        ).fetchone()
        if cached is not None:
            self.cache_hits += 1
            return str(cached[0])

        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(**payload)
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("LLM returned an empty response")
                answer = str(content)
                self.db.execute(
                    "INSERT OR REPLACE INTO responses VALUES (?, ?)",
                    (cache_key, answer),
                )
                self.db.commit()
                self.calls += 1
                return answer
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt < 3:
                    time.sleep(2**attempt)
        raise RuntimeError("LLM request failed after four attempts") from last_error

    @staticmethod
    def _binary_top_logprobs(response: Any) -> dict[str, float]:
        """Extract A/B logprobs and fail closed if either token is absent."""
        try:
            content = response.choices[0].logprobs.content
            first_token = content[0]
            alternatives = first_token.top_logprobs
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "endpoint did not return Chat Completions token logprobs"
            ) from exc

        result: dict[str, float] = {}
        rendered: list[str] = []
        for item in alternatives or []:
            if isinstance(item, Mapping):
                raw_token = item.get("token")
                raw_logprob = item.get("logprob")
            else:
                raw_token = getattr(item, "token", None)
                raw_logprob = getattr(item, "logprob", None)
            rendered.append(repr(raw_token))
            token = str(raw_token).strip().upper() if raw_token is not None else ""
            if token in {"A", "B"} and raw_logprob is not None:
                value = float(raw_logprob)
                if token not in result or value > result[token]:
                    result[token] = value
        missing = sorted({"A", "B"} - set(result))
        if missing:
            raise RuntimeError(
                "top_logprobs did not contain both A and B; missing "
                f"{missing}, returned tokens={rendered}"
            )
        return result

    def binary_label_probabilities(
        self,
        messages: Sequence[Any],
        *,
        top_logprobs: int = 5,
    ) -> dict[str, Any]:
        """Return probabilities normalized over A=SUPPORTS and B=REFUTES.

        These are conditional binary probabilities from a separate one-token
        scoring completion.  They are not reconstructed from generated text.
        """
        if top_logprobs < 2:
            raise ValueError("top_logprobs must be at least 2")
        rendered = self._messages(messages)
        request_seed = self._request_seed(rendered)
        payload = {
            "model": self.model_name,
            "messages": rendered,
            "temperature": 0.0,
            "max_tokens": 1,
            "n": 1,
            "seed": request_seed,
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
        }
        cache_key = hashlib.sha256(
            json.dumps(
                {"schema": "binary-label-logprobs-v1", "payload": payload},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cached = self.db.execute(
            "SELECT payload_json FROM binary_scores WHERE key=?", (cache_key,)
        ).fetchone()
        if cached is not None:
            self.cache_hits += 1
            return dict(json.loads(str(cached[0])))

        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(**payload)
                logprobs = self._binary_top_logprobs(response)
                maximum = max(logprobs.values())
                a_weight = math.exp(logprobs["A"] - maximum)
                b_weight = math.exp(logprobs["B"] - maximum)
                denominator = a_weight + b_weight
                result = {
                    "method": "single_token_A_B_logprobs",
                    "normalization": "conditional_on_A_or_B",
                    "token_mapping": {"A": "SUPPORTS", "B": "REFUTES"},
                    "supports_probability": a_weight / denominator,
                    "refutes_probability": b_weight / denominator,
                    "supports_logprob": logprobs["A"],
                    "refutes_logprob": logprobs["B"],
                    "supports_log_odds": logprobs["A"] - logprobs["B"],
                    "top_logprobs": int(top_logprobs),
                }
                self.db.execute(
                    "INSERT OR REPLACE INTO binary_scores VALUES (?, ?)",
                    (cache_key, json.dumps(result, ensure_ascii=False, sort_keys=True)),
                )
                self.db.commit()
                self.calls += 1
                return result
            except Exception as exc:  # pragma: no cover - network path
                last_error = exc
                # Missing logprob capability/data is deterministic; retries do
                # not make it valid and could hide a server incompatibility.
                if isinstance(exc, RuntimeError) and (
                    "logprobs" in str(exc) or "both A and B" in str(exc)
                ):
                    break
                if attempt < 3:
                    time.sleep(2**attempt)
        raise RuntimeError(
            "binary A/B logprob request failed; the endpoint must expose both "
            "tokens in Chat Completions top_logprobs"
        ) from last_error

    def close(self) -> None:
        self.db.close()
