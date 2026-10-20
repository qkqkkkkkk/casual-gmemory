"""Fit-only rebalancing and Local--Global CSM-ST training (new experiment)."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import random

from .csm_st_data import frequency_weights, rarity, save_json, write_rows


def rebalance(rows, *, seed, trivial_threshold, trivial_keep_fraction, rarity_alpha):
    if not 0 < trivial_keep_fraction <= 1:
        raise ValueError("trivial keep fraction must be in (0,1]")
    rng = random.Random(seed)
    selected = [r for r in rows if rarity(r["ci"], rarity_alpha) >= trivial_threshold
                or rng.random() < trivial_keep_fraction]
    return selected or [rows[rng.randrange(len(rows))]]


def validate_label_rows(rows, *, fit_ids, dev_ids):
    fit, dev = [], []
    if fit_ids & dev_ids:
        raise ValueError("fit/dev task leakage")
    for row in rows:
        if len(row["ci"]) != len(row["candidates"]) or not row["ci"]:
            raise ValueError("misaligned CE labels")
        if not all(math.isfinite(v) for v in row["ci"]):
            raise ValueError("nonfinite CE label")
        origins = set(row["origin_task_ids"])
        if origins <= fit_ids:
            fit.append(row)
        elif origins <= dev_ids and "intervention" not in row:
            dev.append(row)
        else:
            raise ValueError("augmentation donor/query crosses fit/dev boundary")
    if {r["task_id"] for r in dev} != dev_ids or not fit:
        raise ValueError("incomplete CSM fit/dev labels")
    return fit, dev


def train(rows, root, *, model_config, options, fit_ids, dev_ids, metadata,
          dedup_config, label_provenance, network=None, tokenizer=None):
    import numpy as np
    import torch
    from torch import nn
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from .compare_gate_runs import file_sha256
    from .csm_st_model import LocalGlobalCSM, contrastive_loss, tokenize_lists
    from .runtime_gate import CHECKPOINT_SCHEMA

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    seed, device = options["seed"], options["device"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose a valid --train-device")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    fit, dev = validate_label_rows(rows, fit_ids=fit_ids, dev_ids=dev_ids)
    fit = rebalance(fit, seed=seed, trivial_threshold=options["trivial_threshold"],
        trivial_keep_fraction=options["trivial_keep_fraction"], rarity_alpha=options["rarity_alpha"])
    weights = frequency_weights(fit, alpha=options["rarity_alpha"], width=options["rarity_bin_width"])
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_config.encoder, revision=model_config.revision,
        trust_remote_code=False, local_files_only=options["local_files_only"])
    network = network or LocalGlobalCSM(model_config, local_files_only=options["local_files_only"])
    network.to(device)
    optimizer = torch.optim.AdamW([
        {"params": network.encoder.parameters(), "lr": options["encoder_learning_rate"]},
        {"params": [*network.global_layer.parameters(), *network.output.parameters()],
         "lr": options["head_learning_rate"]}], weight_decay=options["weight_decay"])
    artifact = root / "model"
    artifact.mkdir(exist_ok=True)
    network.encoder.config.save_pretrained(artifact / "encoder")
    tokenizer.save_pretrained(artifact / "tokenizer")
    history, best, best_epoch, best_predictions = [], math.inf, None, None
    batch_size = options["batch_tasks"]
    for epoch in tqdm(range(options["epochs"]), desc="Train Local-Global CSM-ST", unit="epoch"):
        network.train()
        order = list(range(len(fit)))
        random.Random(seed + epoch).shuffle(order)
        total, total_mse, total_cts, pairs, n = 0.0, 0.0, 0.0, 0, 0
        for start in tqdm(range(0, len(order), batch_size), desc=f"CSM epoch {epoch + 1}", unit="batch", leave=False):
            indices = order[start:start + batch_size]
            batch = [fit[i] for i in indices]
            tokens, counts = tokenize_lists(tokenizer, batch, model_config, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                                dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                scores, global_vectors, _ = network(tokens, counts)
                regression, contrastive = [], []
                for i, (index, row) in enumerate(zip(indices, batch)):
                    target = torch.tensor(row["ci"], device=device, dtype=torch.float32)
                    regression.append(weights[index] * (scores[i, :counts[i]].float() - target).square().mean())
                    if rarity(row["ci"], options["rarity_alpha"]) > options["hard_threshold"]:
                        value, count = contrastive_loss(global_vectors[i, :counts[i]], row["ci"],
                            temperature=options["temperature"], positive_epsilon=options["positive_epsilon"],
                            negative_epsilon=options["negative_epsilon"])
                        contrastive.append(value)
                        pairs += count
                mse = torch.stack(regression).mean()
                cts = torch.stack(contrastive).mean() if contrastive else scores.sum() * 0
                loss = mse + options["beta"] * cts
            if not torch.isfinite(loss):
                raise ValueError("nonfinite CSM training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            total_mse += float(mse.detach()) * len(batch)
            total_cts += float(cts.detach()) * len(batch)
            n += len(batch)
        network.eval()
        error, sign_correct, candidate_count, predictions = 0.0, 0, 0, []
        with torch.inference_mode():
            for row in dev:
                tokens, counts = tokenize_lists(tokenizer, [row], model_config, device)
                output, _, _ = network(tokens, counts)
                values = output[0, :counts[0]].float().cpu().tolist()
                error += sum((a - b) ** 2 for a, b in zip(values, row["ci"])) / len(values)
                sign_correct += sum((a > 0) == (b > 0) for a, b in zip(values, row["ci"]))
                candidate_count += len(values)
                predictions.append({"id": row["id"], "ci": row["ci"], "predicted_ci": values})
        dev_mse = error / len(dev)
        if not math.isfinite(dev_mse):
            raise ValueError("nonfinite CSM dev loss")
        history.append({"epoch": epoch + 1, "loss": total / n, "weighted_mse": total_mse / n,
            "infonce": total_cts / n, "contrastive_positive_pairs": pairs,
            "dev_list_mean_mse": dev_mse, "dev_ci_sign_accuracy": sign_correct / candidate_count})
        save_json(root / "loss_history.json", history)
        if dev_mse < best:
            best, best_epoch, best_predictions = dev_mse, epoch + 1, predictions
            temporary = artifact / "weights.pt.tmp"
            torch.save(network.state_dict(), temporary)
            temporary.replace(artifact / "weights.pt")
    write_rows(root / "dev_predictions.jsonl", best_predictions)
    all_ids = sorted(fit_ids | dev_ids)
    metadata = {**metadata, "training_task_ids": all_ids, "fit_task_ids": sorted(fit_ids),
        "dev_task_ids": sorted(dev_ids), "training_samples": sum(len(r["ci"]) for r in fit),
        "policy_calibration": {"enabled": False, "recommended_kappa": 0.0,
            "recommended_max_drops": -1, "selection_rule": "keep_iff_ci_gt_zero",
            "note": "dev selects epoch only; no kappa search/KEEP ALL fallback"}}
    diagnostics = {"fit_lists_after_rebalancing": len(fit), "original_fit_tasks": len(fit_ids),
        "dev_tasks": len(dev_ids), "augmented_fit_lists": sum("intervention" in r for r in fit),
        "hard_lists": sum(rarity(r["ci"], options["rarity_alpha"]) > options["hard_threshold"] for r in fit),
        "selected_epoch": best_epoch, "best_dev_list_mse": best,
        "training_list_ids": [r["id"] for r in fit], "loss_history": history,
        "encoder_resolved_commit": getattr(network.encoder.config, "_commit_hash", None),
        "local_pair_truncation": "longest_first; full task text supplied, bounded BERT window"}
    if not diagnostics["hard_lists"] or not any(r["contrastive_positive_pairs"] for r in history):
        import warnings

        warnings.warn("No valid hard-list contrastive pairs: InfoNCE is inactive. Inspect CE scale/fit-only "
                      "thresholds before claiming the contrastive component was tested.", RuntimeWarning)
    hashes = {str(p.relative_to(artifact)): file_sha256(p) for p in artifact.rglob("*") if p.is_file()}
    payload = {"schema": CHECKPOINT_SCHEMA, "training_metadata": metadata,
        "csm_st": {"schema": "csm-st-local-global-v1", "artifact_dir": "model",
            "artifact_sha256": hashes, "config": asdict(model_config), "options": options,
            "dedup": dedup_config, "label_provenance": label_provenance}, "training_diagnostics": diagnostics}
    save_json(root / "training_summary.json", {"options": options, "diagnostics": diagnostics,
        "label_provenance": label_provenance, "training_metadata": metadata})
    save_json(root / "checkpoint.json", payload)
    return payload
