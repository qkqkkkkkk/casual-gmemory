"""Resume an existing ST decision experiment with a recorded numeric audit fix.

The original hash-pinned modules/manifests, dev predictions, calibrated policies
and completed test rows stay intact. Only unfinished workers use the new audit.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from . import run_st_decision_calibration as original
from .interactive_data import digest, locked, sha
from .run_st_encoder_benchmark import offline_models
from .st_single_drop_audit_evaluate import AUDIT_PROTOCOL


@contextmanager
def audited_workers():
    from . import interactive_progress

    run = interactive_progress.run_logged

    def dispatch(command, *args, **kwargs):
        if len(command) < 4 or command[3] != "causal_memory_control.st_single_drop_evaluate":
            raise ValueError("numeric audit resume refuses an unrelated worker command")
        updated = [*command[:3], "causal_memory_control.st_single_drop_audit_evaluate", *command[4:]]
        return run(updated, *args, **kwargs)

    try:
        interactive_progress.run_logged = dispatch
        yield
    finally:
        interactive_progress.run_logged = run


def main(argv=None):
    args = original.parse_args(argv)
    if args.stage != "evaluate":
        return original.main(argv)
    jobs, manifest = original.prepare(args)
    with original.tuning.output_lock(args.output_dir), offline_models():
        # Validate the old run exactly. Never rewrite an old source hash or
        # replace a selection to make it appear compatible with changed code.
        selection = original.require_selection(args, manifest)
        checkpoint_hashes = {name: {variant: sha(model["checkpoint"]) for variant, model in job["models"].items()}
                             for name, job in jobs.items()}
        locked(args.output_dir / "numeric_audit_resume.json", {
            "schema": "semantic-st-single-drop-numeric-audit-resume-v1",
            "original_manifest_sha256": digest(manifest),
            "selection_sha256": sha(args.output_dir / "selection.json"),
            "checkpoint_sha256": checkpoint_hashes, "audit": AUDIT_PROTOCOL,
            "implementation_sha256": {name: sha(Path(__file__).with_name(name)) for name in (
                "resume_st_single_drop_evaluation.py", "st_single_drop_audit_evaluate.py")},
            "completed_test_rows_reused": True, "dev_recomputed": False, "retraining": False})
        with audited_workers():
            return original.evaluate(args, jobs, manifest, selection)


if __name__ == "__main__":
    main()
