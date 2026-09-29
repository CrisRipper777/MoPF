from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _config(seed: int, use_relation_condition: bool):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "seed": seed,
        "model": {
            "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
            "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
            "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
            "num_residual_adapters": 3, "edge_chunk_size": 128,
            "rcfa_hidden_dim": 256, "rcfa_node_chunk_size": 32,
            "use_rcfa": True, "use_relation_condition": use_relation_condition,
        },
    })


def _capture_rcfa_inputs(model, x, edge_index):
    captured = {"text": [], "visual": []}
    handles = []
    for modality in captured:
        def hook(_module, args, name=modality):
            captured[name].append(tuple(value.detach().clone() for value in args))
        handles.append(getattr(model, f"rcfa_{modality}").register_forward_pre_hook(hook))
    try:
        result = model.analyze(x, edge_index)
    finally:
        for handle in handles:
            handle.remove()
    return result, captured


def audit() -> dict:
    import torch
    from src.models.crsa_rcfa import Model

    seed = 1729
    info = {"text_dim": 7, "visual_dim": 5, "input_dim": 12, "num_classes": 3}
    x = torch.linspace(-1.0, 1.0, steps=19 * 12).reshape(19, 12)
    nodes = torch.arange(19)
    edge_index = torch.stack((torch.cat((nodes[:-1], nodes[1:])),
                              torch.cat((nodes[1:], nodes[:-1]))))
    torch.manual_seed(seed)
    full = Model(_config(seed, True), info).eval()
    torch.manual_seed(seed)
    no_rse = Model(_config(seed, False), info).eval()

    full_state, no_rse_state = full.state_dict(), no_rse.state_dict()
    same_keys = set(full_state) == set(no_rse_state)
    init_equal = same_keys and all(torch.equal(full_state[key], no_rse_state[key])
                                   for key in full_state)
    full_count = sum(p.numel() for p in full.parameters())
    no_rse_count = sum(p.numel() for p in no_rse.parameters())
    full_params = sum(p.numel() for p in full.parameters() if p.requires_grad)
    no_rse_params = sum(p.numel() for p in no_rse.parameters() if p.requires_grad)
    full_head = torch.nn.Linear(full.out_dim, info["num_classes"])
    no_rse_head = torch.nn.Linear(no_rse.out_dim, info["num_classes"])
    full_total_params = full_params + sum(p.numel() for p in full_head.parameters() if p.requires_grad)
    no_rse_total_params = no_rse_params + sum(p.numel() for p in no_rse_head.parameters() if p.requires_grad)

    with torch.no_grad():
        full_result, full_inputs = _capture_rcfa_inputs(full, x, edge_index)
        no_rse_result, no_rse_inputs = _capture_rcfa_inputs(no_rse, x, edge_index)

    comparisons = {}
    for modality in ("text", "visual"):
        comparisons[f"H0_{modality}"] = torch.equal(
            full_result[f"H0_{modality}"], no_rse_result[f"H0_{modality}"]
        )
        for hop, (state_a, state_b) in enumerate(zip(
            full_result[f"C_{modality}"], no_rse_result[f"C_{modality}"]
        )):
            comparisons[f"C{hop}_{modality}"] = torch.equal(state_a, state_b)
        for hop, (delta_a, delta_b, rse_a, rse_b) in enumerate(zip(
            full_result[f"delta_{modality}"], no_rse_result[f"delta_{modality}"],
            full_result[f"rse_{modality}"], no_rse_result[f"rse_{modality}"],
        ), start=1):
            comparisons[f"delta{hop}_{modality}"] = torch.equal(delta_a, delta_b)
            comparisons[f"R{hop}_{modality}"] = torch.equal(rse_a, rse_b)

        full_calls, no_rse_calls = full_inputs[modality], no_rse_inputs[modality]
        comparisons[f"condition_call_count_{modality}"] = len(full_calls) == 3 and len(no_rse_calls) == 3
        for hop in range(min(len(full_calls), len(no_rse_calls), 3)):
            h_full, delta_full, r_full = full_calls[hop]
            h_zero, delta_zero, r_zero = no_rse_calls[hop]
            expected_rse = full_result[f"rse_{modality}"][hop]
            comparisons[f"gate_H0_{hop + 1}_{modality}"] = torch.equal(h_full, h_zero)
            comparisons[f"gate_delta_{hop + 1}_{modality}"] = torch.equal(delta_full, delta_zero)
            comparisons[f"full_condition_R{hop + 1}_{modality}"] = torch.equal(r_full, expected_rse)
            comparisons[f"noRSE_condition_zero_{hop + 1}_{modality}"] = torch.equal(
                r_zero, torch.zeros_like(expected_rse)
            )
            comparisons[f"gate_input_diff_only_R{hop + 1}_{modality}"] = (
                torch.equal(h_full, h_zero) and torch.equal(delta_full, delta_zero)
                and torch.equal(r_full, expected_rse) and torch.equal(r_zero, torch.zeros_like(expected_rse))
            )

    try:
        model_hashes = {
            name: subprocess.check_output(["git", "hash-object", str(ROOT / name)],
                                          cwd=ROOT, text=True).strip()
            for name in ("src/models/crsa_rcfa.py", "configs/model/crsa_rcfa.yaml")
        }
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        model_hashes, commit = {}, "unknown"

    passed = (
        full.use_rcfa and no_rse.use_rcfa and full.use_relation_condition
        and not no_rse.use_relation_condition and same_keys and init_equal
        and full_count == no_rse_count and full_params == no_rse_params
        and full_total_params == no_rse_total_params
        and bool(comparisons) and all(comparisons.values())
    )
    return {
        "status": "passed" if passed else "failed",
        "seed": seed,
        "variant_controls": {
            "full_use_rcfa": full.use_rcfa,
            "full_use_relation_condition": full.use_relation_condition,
            "noRSE_use_rcfa": no_rse.use_rcfa,
            "noRSE_use_relation_condition": no_rse.use_relation_condition,
        },
        "state_dict_keys_identical": same_keys,
        "initialization_tensors_identical": init_equal,
        "parameter_count_full": full_count,
        "parameter_count_noRSE": no_rse_count,
        "trainable_parameter_count_full": full_params,
        "trainable_parameter_count_noRSE": no_rse_params,
        "parameter_count_including_nc_head_full": full_total_params,
        "parameter_count_including_nc_head_noRSE": no_rse_total_params,
        "all_parameter_tensors_equal": init_equal,
        "stage_i_state_and_effect_checks": comparisons,
        "model_and_config_blob_hashes_at_preflight": model_hashes,
        "preflight_source_commit": commit,
        "difference_description": "RCFA conditioner R input is R_k in Full and zeros_like(R_k) in noRSE; H0, delta, C, and R computation are identical.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the explicit RSE conditioning ablation")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rcfa_relation_condition_ablation/preflight_audit.json")
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
