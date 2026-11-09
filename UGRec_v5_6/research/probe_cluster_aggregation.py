"""Training-item algebra checks; uniform-assignment intervention is not a rank ablation."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from compression.geometry_ae.model import GeometryAE
from joint_training.checkpoint import read_checkpoint
from joint_training.config import JointConfig
from joint_training.data import initialize_random_ae
from joint_training.model import JointModel
from misc.artifacts import array_hash, json_hash, write_json
from protocol import ROOT, SELECTION_SEED, corpus, raw_features
from run import seed_run


def relative_difference(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


@torch.no_grad()
def inspect(model, raw):
    tokenizer = model.ae.cluster_tokenizer
    aggregation = model.ae.cluster_aggregation
    latent = GeometryAE.latent(model.ae, raw)
    expanded = tokenizer.fc0(latent)
    grouped = expanded.reshape(len(raw), tokenizer.G, tokenizer.dim)
    scores = tokenizer.fc_gk(expanded).reshape(len(raw), tokenizer.G, tokenizer.K)
    assignment = (scores / tokenizer.assignment_temperature).softmax(-1)
    gate = tokenizer.fc_g(expanded).sigmoid()

    def components(probability):
        activation = probability * gate[..., None]
        content = torch.einsum("ngk,ngd->nkd", activation, grouped)
        center = activation.sum(1)[..., None] * tokenizer.centers
        return content - center, content, center

    residual, content, center = components(assignment)
    uniform, uniform_content, _ = components(torch.full_like(assignment, 1. / tokenizer.K))
    actual = model.ae.components(raw)
    torch.testing.assert_close(residual, actual, rtol=1e-5, atol=1e-7)
    # Sum of assignment probabilities is one: content sum is independent of them.
    content_sum = (gate[..., None] * grouped).sum(1)
    torch.testing.assert_close(content.sum(1), content_sum, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(uniform_content.sum(1), content_sum, rtol=1e-5, atol=1e-7)
    summed = residual.sum(1)
    aggregate = aggregation(residual)
    aggregate_uniform = aggregation(uniform)
    torch.testing.assert_close(summed, content_sum - center.sum(1), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(aggregate, model.ae.latent(raw), rtol=1e-5, atol=1e-7)
    normalized = model.normalizer(aggregate)
    normalized_uniform = model.normalizer(aggregate_uniform)
    entropy = -(assignment * assignment.clamp_min(1e-12).log()).sum(-1).mean()
    result = {
        "content_sum_assignment_cancellation_verified": True,
        "implementation_and_formula_match": True,
        "content_sum_max_abs_error": float((content.sum(1) - content_sum).abs().max()),
        "center_term_norm_over_content_sum_norm": float(center.sum(1).norm() / content_sum.norm()),
        "normalized_assignment_entropy": float(entropy / torch.tensor(tokenizer.K).log()),
        "assignment_std_across_items_mean": float(assignment.std(0, unbiased=False).mean()),
        "uniform_assignment_intervention": {
            "raw_components_relative_l2": relative_difference(residual, uniform),
            "raw_sum_relative_l2": relative_difference(summed, uniform.sum(1)),
            "actual_aggregate_relative_l2": relative_difference(aggregate, aggregate_uniform),
            "normalized_catalog_latent_cosine_mean": float(F.cosine_similarity(normalized, normalized_uniform).mean()),
            "normalized_catalog_latent_relative_l2": relative_difference(normalized, normalized_uniform),
        },
    }
    if aggregation.mode == "linear":
        blocks = aggregation.weight.reshape(tokenizer.dim, tokenizer.K, tokenizer.dim).permute(1, 0, 2)
        common = blocks.mean(0)
        identity = torch.eye(tokenizer.dim, device=blocks.device)
        result["learned_blocks"] = {
            "between_block_deviation_over_weight_norm": float((blocks - common).norm() / blocks.norm()),
            "common_block_distance_from_identity_relative": float((common - identity).norm() / identity.norm()),
            "bias_norm": float(aggregation.bias.norm()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    directory = ROOT / "outputs/item_cluster_followup/runs/Beauty" / f"beauty_{args.method}_s2024"
    output = ROOT / "outputs/item_cluster_followup/mechanisms" / args.method
    output.mkdir(parents=True, exist_ok=True)
    if (output / "aggregation_probe.json").exists():
        raise FileExistsError(output)
    config = JointConfig.from_dict(json.loads((directory / "config.json").read_text()))
    config.training.device = args.device
    torch.set_num_threads(config.training.cpu_threads)
    # Algebra checks compare different reduction forms, so avoid TF32 rounding.
    # Training remains at its original "high" setting.
    torch.set_float32_matmul_precision("highest")
    cache, mapping, train, _, _ = corpus(config.data.category)
    raw = raw_features(cache, len(mapping["id2item"]) - 1)
    train_ids = torch.tensor(sorted({item for seq in train for item in seq}))
    item_ids = train_ids[torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(SELECTION_SEED))[:256]]
    seed_run(config.training.seed, config.training.deterministic)
    model = JointModel(config, raw, initialize_random_ae(config, raw, train_ids)).to(args.device)
    experiment = json.loads((directory / "experiment.json").read_text())
    state = {name: array_hash(value.detach().cpu().numpy()) for name, value in model.state_dict().items()}
    assert json_hash(state) == experiment["initial_state_sha256"]
    model.eval()
    sampled_raw = model.raw_items(item_ids.to(args.device))
    initial = inspect(model, sampled_raw)
    checkpoint_path = directory / "best_flow.pt"
    checkpoint = read_checkpoint(checkpoint_path)
    assert checkpoint["source"] == experiment["source"]
    model.load_state_dict(checkpoint["model"], strict=True)
    trained = inspect(model, sampled_raw)
    source = Path(__file__).read_bytes()
    with (output / "probe_source.py").open("xb") as stream:
        stream.write(source)
    result = {"method": args.method, "initial": initial, "trained": trained,
              "initial_state_reproduced": True, "test_accessed": False,
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(source).hexdigest(),
              "training_item_ids_sha256": array_hash(item_ids.numpy()),
              "matmul_precision": "highest (FP32 algebra checks only; training unchanged)",
              "scope": "256 fixed training items; no ranking evaluation or checkpoint selection"}
    write_json(output / "aggregation_probe.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
