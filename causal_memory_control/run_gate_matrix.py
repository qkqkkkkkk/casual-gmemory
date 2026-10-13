"""Select a dataset and run the shared resumable gate comparison matrix."""

from __future__ import annotations

import argparse
import sys

from .dataset_adapters import DATASET_ADAPTERS, load_dataset_adapter
from .run_hotpotqa_gate_matrix import main as run_matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--dataset", default="hotpotqa", help="Built in: hotpotqa, fever")
    parser.add_argument("--dataset-adapter", help="Optional Python module:factory for another dataset")
    parser.add_argument("--list-datasets", action="store_true")
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.list_datasets:
        for name, factory in DATASET_ADAPTERS.items():
            adapter = factory()
            print(f"{name}: {adapter.display_name}; metrics={','.join(metric.label for metric in adapter.metrics)}")
        return
    adapter = load_dataset_adapter(args.dataset.lower(), args.dataset_adapter)
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
        print("\nMatrix and selected dataset options:")
    return run_matrix(remaining, adapter=adapter)


if __name__ == "__main__":
    main()
