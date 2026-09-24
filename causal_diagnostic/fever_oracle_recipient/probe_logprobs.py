"""Probe an OpenAI-compatible endpoint for strict A/B chat log probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Sequence

from causal_diagnostic.oracle_recipient.seeded_client import SeededCachedChat


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--cache", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.top_logprobs < 2:
        raise SystemExit("--top-logprobs must be at least 2")
    temporary = None
    cache = args.cache
    if cache is None:
        temporary = tempfile.TemporaryDirectory(prefix="gmemory-logprob-probe-")
        cache = Path(temporary.name) / "probe.sqlite"
    client = SeededCachedChat(
        args.model,
        cache,
        experiment_seed=0,
        api_base=args.endpoint,
        api_key=args.api_key,
    )
    try:
        score = client.binary_label_probabilities(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a binary classifier. Return exactly one "
                        "character: A or B."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Choose the true statement. A: water is wet. B: a week "
                        "has nine days. Return only A or B."
                    ),
                },
            ],
            top_logprobs=args.top_logprobs,
        )
    finally:
        client.close()
        if temporary is not None:
            temporary.cleanup()
    payload = {
        "status": "supported",
        "endpoint": args.endpoint,
        "model": args.model,
        "method": "single_token_A_B_logprobs",
        "top_logprobs": args.top_logprobs,
        "score": score,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
