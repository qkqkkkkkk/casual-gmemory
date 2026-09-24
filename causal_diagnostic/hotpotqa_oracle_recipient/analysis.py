"""HotpotQA-labelled reports over the shared event-clustered causal analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from causal_diagnostic.oracle_recipient.analysis import (
    _fmt,
    _report_policy_table,
    _write_rq4_matrix,
    write_analysis as _write_shared_analysis,
)


ANALYSIS_SCHEMA = "gmemory-macnet-hotpotqa-rq234-analysis-v1"


def _patch_hotpot_labels(run: dict[str, Any] | None) -> None:
    if run is None:
        return
    run["analysis_schema"] = ANALYSIS_SCHEMA
    rq4 = run.get("rq4", {})
    if rq4.get("status") != "ok":
        return
    rq4["local_metric"] = "HotpotQA supporting-fact F1"
    rq4["team_metric"] = "official normalized HotpotQA answer token-F1"
    for row in rq4.get("events", []):
        row["local_metric"] = "hotpotqa_supporting_fact_f1"


def _report(payload: Mapping[str, Any]) -> str:
    current = payload["current"]
    rq3 = current["rq3"]["primary_team_score"]
    rq4 = current.get("rq4", {})
    retest = payload.get("independent_retest")
    lines = [
        "# Native GMemory + MacNet HotpotQA causal audit",
        "",
        f"- Held-out question-memory events: {current['event_count']}",
        f"- Recipients: {', '.join(current['recipients'])}",
        "- Primary team outcome: official normalized answer token-F1",
        "- Sensitivity outcome: normalized answer exact match",
        "- Local outcome: supporting-fact F1 over (title, sentence index)",
        "- Evaluation mode: offline with identical supplied context in every branch",
        "",
        "## RQ2: Global selective-memory value",
        "",
    ]
    lines.extend(
        _report_policy_table(
            payload["rq2_policy_tables"]["team_score"],
            score_name="Answer F1",
            percent_score=True,
        )
    )
    lines.extend(["", "### Answer exact-match sensitivity", ""])
    lines.extend(
        _report_policy_table(
            payload["rq2_policy_tables"]["success"],
            score_name="Answer EM",
            percent_score=True,
        )
    )
    if "decision_evidence_f1" in payload["rq2_policy_tables"]:
        lines.extend(["", "### Final-decision supporting-fact sensitivity", ""])
        lines.extend(
            _report_policy_table(
                payload["rq2_policy_tables"]["decision_evidence_f1"],
                score_name="Supporting-fact F1",
                percent_score=True,
            )
        )
    lines.extend(
        [
            "",
            "CIs use event-paired percentile bootstrap. Oracle Selective Drop is "
            "a same-sample upper bound; Cross-run Selective Drop is the independent-seed check.",
            "",
            "## RQ3: Recipient heterogeneity",
            "",
            "- Intervention: isolated recipient exposure minus the same global-drop control",
            "- Direct positive/negative recipient sign-flip rate: "
            + _fmt(rq3["direct_sign_flip_rate"], percent=True),
            "- Any recipient sign heterogeneity rate: "
            + _fmt(rq3["any_sign_heterogeneity_rate"], percent=True),
            "- Mean within-event answer-F1 utility range: "
            + _fmt(rq3["mean_utility_range"]),
            "",
            "| Recipient | Mean answer-F1 utility [CI] | Positive | Neutral | Negative |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for recipient in current["recipients"]:
        row = rq3["per_recipient"][recipient]
        lines.append(
            f"| {recipient} | {_fmt(row['mean_utility'])} | "
            f"{row['positive_events']} | {row['neutral_events']} | "
            f"{row['negative_events']} |"
        )
    stable = rq3.get("split_half_reproducible_sign_flip_rate")
    lines.extend(
        [
            "",
            "- Split-half reproducible recipient sign-flip rate: "
            + (
                _fmt(stable, percent=True)
                if stable is not None
                else "unavailable (at least four paired repeats required)"
            ),
            "",
            "## RQ4: Supporting-fact-to-answer gap",
            "",
        ]
    )
    if rq4.get("status") != "ok":
        lines.append("Not available: " + str(rq4.get("reason", rq4.get("status"))))
    else:
        lines.extend(
            [
                "- Local/team sign-mismatch rate: "
                + _fmt(rq4["local_team_mismatch_rate"], percent=True),
                "- Split-half reproducible mismatch rate: "
                + (
                    _fmt(
                        rq4["reproducible_local_team_mismatch_rate"],
                        percent=True,
                    )
                    if rq4.get("reproducible_local_team_mismatch_rate") is not None
                    else "unavailable"
                ),
            ]
        )
        correlation = rq4["local_team_utility_correlation"]
        lines.append(
            "- Local/team utility correlation: "
            + (
                _fmt(correlation)
                if correlation.get("status") == "ok"
                else str(correlation.get("status"))
            )
        )
        labels = {
            "local_positive_team_positive": "Local +, Team +",
            "local_positive_team_negative": "Local +, Team -",
            "local_negative_team_positive": "Local -, Team +",
            "local_negative_team_negative": "Local -, Team -",
            "neutral_involved": "Neutral involved",
        }
        lines.extend(["", "| Pattern | Event-clustered fraction [CI] |", "|---|---:|"])
        for key, label in labels.items():
            lines.append(f"| {label} | {_fmt(rq4['pattern_rates'][key], percent=True)} |")
    lines.extend(["", "## Independent retest", ""])
    if retest is None:
        lines.append("Not available; run a disjoint inference-seed retest.")
    else:
        lines.extend(
            [
                f"- Shared events: {retest['shared_events']}",
                "- RQ2 global-utility sign consistency: "
                + _fmt(retest["rq2"]["global_utility_sign_consistency"], percent=True),
                "- RQ2 bidirectional cross-run gain: "
                + _fmt(retest["rq2"]["bidirectional_cross_run_gain"]),
                "- RQ3 recipient sign consistency: "
                + _fmt(retest["rq3"]["recipient_sign_consistency"], percent=True),
                "- RQ3 reproducible directional flip rate: "
                + _fmt(
                    retest["rq3"]["reproducible_directional_flip_rate"],
                    percent=True,
                ),
            ]
        )
        if retest.get("rq4", {}).get("status") == "ok":
            lines.append(
                "- RQ4 local/team pattern agreement: "
                + _fmt(retest["rq4"]["pattern_agreement"], percent=True)
            )
    return "\n".join(lines) + "\n"


def write_analysis(
    results_dir: Path,
    *,
    retest_results: Path | None = None,
    delta: float = 0.0,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Path:
    output = _write_shared_analysis(
        results_dir,
        retest_results=retest_results,
        delta=delta,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["analysis_schema"] = ANALYSIS_SCHEMA
    payload["benchmark"] = "HotpotQA_distractor_offline"
    _patch_hotpot_labels(payload.get("current"))
    _patch_hotpot_labels(payload.get("previous"))
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rq4_matrix(results_dir / "rq4_local_team_matrix.csv", payload["current"])
    (results_dir / "run_report.md").write_text(_report(payload), encoding="utf-8")
    return output
