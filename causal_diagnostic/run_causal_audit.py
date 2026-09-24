"""Dispatch the isolated FEVER or HotpotQA causal-audit pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("benchmark", choices=("fever", "hotpotqa"))
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args, remaining = parse_args(argv)
    if args.benchmark == "fever":
        from causal_diagnostic.fever_oracle_recipient.run_all import (
            main as run_pipeline,
        )
    else:
        from causal_diagnostic.hotpotqa_oracle_recipient.run_all import (
            main as run_pipeline,
        )
    return run_pipeline(remaining)


if __name__ == "__main__":
    main()
