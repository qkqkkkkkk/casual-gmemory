"""Lightweight policy/stage definitions; importing a plan never loads models."""

POLICIES = (
    ("always_use", "Always Use baseline"), ("global_drop", "Always Global Drop"),
    ("random_drop", "Random Drop"), ("semantic_linear", "Semantic Linear Gate"),
    ("semantic_mlp", "Semantic MLP (three heads)"), ("semantic_st", "Semantic ST direct U + InfoNCE"),
    ("csm_st", "CSM-ST Local–Global CE proxy"),
    ("oracle", "Oracle Selective Drop (post hoc)"),
)
TEAM_U_POLICY = ("local_global_team_u", "Local–Global + Team U + semantic dedup")
STAGES = ("experts", "support", "collect", "labels", "train_reward", "train_csm", "evaluate", "report")


def policies(config):
    return POLICIES[:-1] + (TEAM_U_POLICY,) + POLICIES[-1:] if config.get("include_local_global_team_u") else POLICIES


def learned_methods(config):
    return tuple(m for m, _ in policies(config) if m.startswith("semantic_") or m in ("csm_st", "local_global_team_u"))


def stages(config):
    if not config.get("include_local_global_team_u"):
        return STAGES
    return (*STAGES[:4], "labels_team_u", *STAGES[4:6], "train_team_u", *STAGES[6:])


def training_options(config, *, team_u=False):
    options = dict(seed=2024, device=config["train_device"], epochs=10, batch_tasks=16,
        encoder_learning_rate=2e-5, head_learning_rate=1e-4, weight_decay=0.01, beta=0.1,
        temperature=1.0, positive_epsilon=0.1, negative_epsilon=0.3, rarity_alpha=10.0,
        trivial_threshold=5.0, hard_threshold=5.0, rarity_bin_width=1.0, trivial_keep_fraction=0.2,
        local_files_only=True)
    if team_u:
        # Same reward-scale settings as run_local_global_team_u_benchmark.settings.
        options.update(positive_epsilon=0.05, negative_epsilon=0.25,
            trivial_threshold=0.1, hard_threshold=0.1, rarity_bin_width=0.25, trivial_keep_fraction=1.0)
    return options
