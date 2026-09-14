# Causal Memory Control

This package implements the final event-level method alongside G-Memory.  It
does not change G-Memory storage, graph construction, retrieval, or write
policy.  The causal unit is one candidate memory used by one receiver at one
frozen state:

```text
x = (query, task state, receiver agent, memory, retrieved candidate set)
```

The implementation has four gated stages:

1. `CounterfactualAudit` replays matched-seed `USE` versus receiver-level
   `DROP`. `PLACEBO` and `GLOBAL_DROP` are optional robustness arms.
2. `OracleControllability` tests whether the oracle policy
   `1[Q_use - Q_drop > delta]` improves final team return before any predictor
   is trained. Similarity, reliability, judge, and local policies are compared
   at the oracle's acceptance budget.
3. `AmortizedUtilityEstimator` learns the two potential outcomes `Q_use` and
   `Q_drop` with a deterministic bootstrap ensemble. Team utility is their
   difference. Receiver behavior change stays a diagnostic signal.
4. `RelianceController` emits only `ACCEPT`, `REJECT`, or `VERIFY`:

```text
ACCEPT  when U_hat - kappa * sigma >  delta
REJECT  when U_hat + kappa * sigma < -delta
VERIFY  otherwise
```

Shapley/PID attribution, propagation actions, GNNs, RL, LoRA, and changes to
the memory write policy are intentionally outside V1.

## G-Memory insertion point

In the current MacNet scheduler, retrieval is performed in
`tasks/mas_workflow/macnet/graph_mas.py`, then `successful_shots` and
`raw_rules` are passed to `format_task_prompt_with_insights()`.  Capture the
checkpoint and run/gate this module between those two operations.

```python
from causal_memory_control import GMemoryRetrievalAdapter

adapter = GMemoryRetrievalAdapter()
adapted = adapter.adapt(meta_memory.retrieve_memory(...))

event = adapter.build_event(
    adapted,
    target_id=adapted.candidates[0].memory_id,
    query=task_main,
    task_state=meta_memory.summarize(upstream_agent_ids=None),
    receiver_agent_id=curr_node.id,
    receiver_role=curr_node._agent.profile,
    recipient_agent_ids=[node.id for node in self._agent_nodes.values()],
)
```

The current MacNet path gives every worker the same retrieved trajectories and
rules.  Passing all worker IDs above therefore records the real `|O(m)|`
instead of silently assuming one recipient.  If role projection makes the
sets differ, pass `candidate_ids_by_recipient` explicitly.

For an audit, the host supplies a `BranchRunner` that restores
`AuditCheckpoint.upstream_state`, fixes the request seed/sampling config,
renders only `request.recipient_contexts`, executes the receiver and downstream
suffix, and returns a `BranchOutcome`.  `CallableBranchRunner` adapts an
ordinary function. `GMemoryRetrievalAdapter.render_prompt_inputs()` turns a
branch request back into `memory_few_shots` and `insights`.

G-Memory does not currently expose a receiver-level suffix checkpoint API.
Until the host provides one, its existing frozen whole-run diagnostic can be
wrapped as a higher-cost fallback, but it must be labeled as whole-run replay;
it is not equivalent to the primary receiver-level intervention.

## Existing P2 data checks

The cheap pre-training checks are public functions:

- `stratify_pivotality`: mismatch rate by the other agents' vote margin.
- `observation_count_distribution`: empirical distribution of `|O(m)|`.
- `oracle_noise_floor`: per-event and aggregate sign consistency across exact
  repeated `(memory, receiver)` configurations.

If only final binary answers were logged, continuous team vote probabilities
cannot be reconstructed by this package; that signal needs probability or
vote-distribution logging at collection time.

`evaluate_pretraining_gate` combines the oracle-headroom check with repeated
sign consistency. `CausalMemoryControlMethod.fit()` requires a passing report
by default, so low oracle consistency cannot silently flow into predictor
training. The consistency threshold is configurable and reported explicitly;
it is not hidden inside the estimator.

## Tests

From the repository root:

```bash
python3 -m unittest discover -s causal_memory_control/tests -v
```

All unit tests are dependency-free and do not open or mutate a G-Memory
database.
