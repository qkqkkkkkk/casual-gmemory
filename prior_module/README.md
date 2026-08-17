# Standalone Prior Module

This package implements the repository-root prior module design without
importing, editing, or writing to G-Memory.

## What is implemented

- Duck-typed G-Memory adapters for trajectory MASMessage objects and insight
  dictionaries. They produce stable CandidateRecord objects without changing
  the source memory.
- A feature builder over task text, online state text, Agent role text,
  candidate text, retrieval metadata, graph-derived fields, and explicit
  missingness masks.
- Replay-backed empirical-Bayes reliability prior, r0.
- Canonical environment actions and ordered behavior distance, d0.
- A five-level ordinal d predictor and a three-class team utility, u0,
  predictor. Both have transparent uniform cold starts and a dependency-free
  linear softmax baseline for supervised training.
- Source replay, paired action intervention, paired team rollout label helpers,
  and append-only JSONL decision logging.

## Minimal integration

    from prior_module import (
        GMemoryAdapter, GMemoryEmbeddingAdapter, DecisionContext, PriorModule
    )

    adapter = GMemoryAdapter()
    candidates = [
        adapter.adapt_trajectory(
            message,
            source_metadata={
                "source_environment_family": "alfworld",
                "source_model_id": "gpt-...",
                "source_prompt_version": "v1",
            },
        )
        for message in successful_trajectories
    ]
    candidates += [adapter.adapt_insight(rule) for rule in insights]

    prior = PriorModule(
        embedder=GMemoryEmbeddingAdapter(gmemory.embedding_func),
        decision_log_path="prior_data/decisions.jsonl",
    )
    scores = prior.score(decision_context, candidates, retrieval_by_id)

Construct decision_context at the workflow boundary from the current task,
receiving Agent, latest observation, and recent trajectory. The caller remains
responsible for choosing which USE, VERIFY, WARN, or IGNORE recommendations to
put into an Agent prompt.

## Action parser boundary

Pass the environment's own parser to ActionCanonicalizer whenever possible:

    canonicalizer = ActionCanonicalizer(
        environment_parser=environment.parse_action_to_canonical
    )

The included parser is only a fallback for common ALFWorld-like commands. It
does not know environment-specific aliases or object identities. A plan
commitment expressed only in natural language is intentionally excluded from
D_env; record it separately as communication or plan influence unless it is
automatically executed downstream.

## Collecting labels

- verify_source_replay(artifact, executor) returns a replay label for r. The
  caller provides an executor that resets the original task and executes raw
  source actions.
- collect_dependency_pair(...) returns
  y_d = D_env(action_with, action_control) after matched prompt branches.
- collect_team_rollout_pair(...) returns the reward delta and
  negative/neutral/positive class for u.

Train only after preserving feature rows for the exact decision context:

    prior.fit_dependency(dependency_feature_rows, distance_labels)
    prior.fit_team_utility(utility_feature_rows, utility_class_labels)

## Tests

From the repository root:

    python3 -m unittest discover -s prior_module/tests -v

The core tests are pure Python and do not open or mutate a G-Memory database.
The direct MASMessage compatibility test runs when the optional G-Memory
runtime dependencies are installed.
