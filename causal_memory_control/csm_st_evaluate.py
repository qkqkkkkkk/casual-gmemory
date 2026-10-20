"""CSM-only evaluation entry, with a scoped host-runner dependency binding.

No source edits or permanent monkey patches to old ST/runtime modules. The
existing host runner continues to own task isolation, cache replay and audits.
"""
from __future__ import annotations

import argparse
import importlib

from .csm_st_runtime import CSMExposureGate


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", choices=("fever", "hotpotqa"), required=True)
    parser.add_argument("--csm-device", default="cpu")
    args, remaining = parser.parse_known_args(argv)
    runner = importlib.import_module(f"causal_memory_control.{args.dataset}_experiment")
    original = runner.GMemoryExposureGate

    class DeviceGate(CSMExposureGate):
        device = args.csm_device

    try:
        runner.GMemoryExposureGate = DeviceGate
        return runner.main(remaining)
    finally:
        runner.GMemoryExposureGate = original


if __name__ == "__main__":
    main()
