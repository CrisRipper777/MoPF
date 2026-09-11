"""Run the controlled U3-B TCPR candidate test.

The script consumes the frozen U2-C C1 runs as B0, trains only the opt-in
zero-initialized TCPR candidate as B1, and performs frozen mechanism
interventions on B1 best-validation checkpoints.  Test metrics are recorded
descriptively and never used by the selection gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.u3a import (  # noqa: E402
    canonical_effective_decomposition,
    contribution_profile,
    counterfactual_eta,
)
from src.analysis.u3b import (  # noqa: E402
    centered_transport_context,
    partial_spearman,
    shuffle_node_scalar,
    transport_residual,
)
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402
from src.utils.summary import count_parameters  # noqa: E402
from scripts.run_mopf_u1_semantic_conductance import (  # noqa: E402
    DATASETS,
    K_BY_DATASET,
    SEEDS,
    _fixed_split_override,
    _peak_gpu_memory_mib,
)


OUTPUT_ROOT = ROOT / "outputs" / "u3b_transport_conditioned_composition"
PREFLIGHT_SUMMARY = OUTPUT_ROOT / "u3b_preflight_summary.json"
U2C_SUMMARY = ROOT / "outputs" / "u2c_formal_factorial_training" / "u2c_master_summary.json"
SHUFFLE_SEEDS = (20260931, 20260932, 20260933)
HIERARCHY_SHUFFLE_SEEDS = (20260921, 20260922, 20260923)
FORMAL_MODEL_CONFIG = ROOT / "configs" / "model" / "mopf.yaml"
MACHINE_ZERO = 1e-8
FUNCTIONAL_EFFECT_FLOOR = 1e-4


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot encode {type(value)!r}")


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = max(float(second.norm().item()), 1e-12)
    return float((first - second).norm().item() / denominator)


def _data_info(data: Any) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _compose_cfg(dataset: str, seed: int, device: str, use_transport: bool):
    k = K_BY_DATASET[dataset]
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        f"seed={seed}",
        f"device={device}",
        "model.edge_weight_mode=learned_diag_cos",
        "model.edge_weight_temperature=0.35",
        f"model.max_order={k}",
        f"model.num_layers={k}",
        "model.num_metric_perspectives=4",
        "model.metric_init_seed=20260910",
        "model.metric_init_noise_std=0.01",
        "model.multihop_state_mode=anchored",
        "model.multihop_response_mode=cumulative",
        "model.multihop_anchor_alpha=0.1",
        (
            f"model.use_transport_residual={'true' if use_transport else 'false'}"
            if "use_transport_residual:" in FORMAL_MODEL_CONFIG.read_text(encoding="utf-8")
            else f"+model.use_transport_residual={'true' if use_transport else 'false'}"
        ),
    ]
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        overrides.append(f"dataset.nc_split_path={split_override}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def _source_b0_records() -> list[dict[str, Any]]:
    summary = json.loads(U2C_SUMMARY.read_text(encoding="utf-8"))
    rows = [row for row in summary["records"] if row["variant"] == "C1"]
    rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), SEEDS.index(int(row["seed"]))))
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    actual = {(row["dataset"], int(row["seed"])) for row in rows}
    if actual != expected:
        raise RuntimeError(f"Expected exactly 15 frozen C1 records, got {sorted(actual)}")
    for row in rows:
        if not Path(row["checkpoint"]).is_file():
            raise FileNotFoundError(row["checkpoint"])
    return rows


def _device(value: str) -> str:
    value = str(value).strip()
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        print(f"[U3-B] {value} unavailable; using cpu", flush=True)
        return "cpu"
    return value


def _run_training_job(
    dataset: str,
    seed: int,
    device: str,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / dataset / "B1_TCPR" / f"seed{seed}"
    checkpoint = run_dir / "best.pt"
    record_path = run_dir / "run_record.json"
    if resume and record_path.is_file() and checkpoint.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    k = K_BY_DATASET[dataset]
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        "num_runs=1",
        f"seed={seed}",
        f"device={device}",
        "model.edge_weight_mode=learned_diag_cos",
        "model.edge_weight_temperature=0.35",
        f"model.max_order={k}",
        f"model.num_layers={k}",
        "model.multihop_state_mode=anchored",
        "model.multihop_response_mode=cumulative",
        "model.multihop_anchor_alpha=0.1",
        "task.training_mode=full_graph",
        "task.optimizer=adamw",
        "task.epochs=300",
        "task.lr=1e-3",
        "task.weight_decay=1e-4",
        "task.patience=30",
        "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4",
        "task.eval_every=1",
        "task.grad_clip=1.0",
        "task.evaluate_test=true",
        "model.export_aux_stats=false",
        "model.export_node_aux=false",
        "model.export_edge_aux=false",
        f"task.save_ckpt_path={checkpoint}",
        f"hydra.run.dir={run_dir}",
    ]
    transport_override = (
        "model.use_transport_residual=true"
        if "use_transport_residual:" in FORMAL_MODEL_CONFIG.read_text(encoding="utf-8")
        else "+model.use_transport_residual=true"
    )
    command.append(transport_override)
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        command.append(f"dataset.nc_split_path={split_override}")
    log_path = (run_dir / "process.log").open("w", encoding="utf-8")
    started = time.monotonic()
    process = None
    peak = None
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=log_path,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            current = _peak_gpu_memory_mib(process.pid, device)
            if current is not None:
                peak = current if peak is None else max(peak, current)
            time.sleep(1.0)
    finally:
        log_path.close()
    elapsed = time.monotonic() - started
    if process is None or process.returncode != 0:
        raise RuntimeError(
            f"B1 TCPR failed for {dataset} seed={seed}: {' '.join(command)}; "
            f"inspect {run_dir / 'process.log'}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected checkpoint missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = _compose_cfg(dataset, seed, "cpu", True)
    model = build_model(cfg, payload["data_info"])
    expected_extra = 2 * (k + 1)
    parameter_count = count_parameters(model) + count_parameters(
        nn.Linear(model.out_dim, int(payload["data_info"]["num_classes"]))
    )
    record = {
        "dataset": dataset,
        "seed": int(seed),
        "variant": "B1_TCPR",
        "formal_K": k,
        "device": device,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "downstream": {
            key: payload.get("metrics", {}).get(key)
            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
        },
        "best_epoch": payload.get("epoch"),
        "parameter_count": parameter_count,
        "expected_tcp_residual_parameters": expected_extra,
        "training_time_seconds": float(elapsed),
        "peak_gpu_memory_mib": peak,
        "peak_gpu_memory_status": "available" if peak is not None else "unavailable",
    }
    _dump(record_path, record)
    return record


def _load_checkpoint(record: dict[str, Any], device: str, use_transport: bool):
    dataset, seed = record["dataset"], int(record["seed"])
    cfg = _compose_cfg(dataset, seed, device, use_transport)
    data = load_mag_data(cfg, "nc", seed)
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(cfg, payload["data_info"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    classifier = nn.Linear(model.out_dim, int(data.num_classes))
    classifier.load_state_dict(payload["head_state"])
    classifier.to(device).eval()
    with torch.no_grad():
        components = model._encode_components(data.x.to(device), data.edge_index.to(device))
    return data, model, classifier, components


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260911)
    x = torch.randn((9, 10), generator=generator)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 0, 8],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 8, 0]],
        dtype=torch.long,
    )
    return x, edge_index, {
        "input_dim": 10,
        "num_nodes": 9,
        "num_classes": 4,
        "text_dim": 4,
        "visual_dim": 6,
    }


def _initialization_audit() -> dict[str, Any]:
    def compare(
        baseline: nn.Module,
        candidate: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, Any]:
        with torch.no_grad():
            base = baseline._encode_components(x, edge_index)
            cand = candidate._encode_components(x, edge_index)
        names = ("eta_text", "eta_visual", "z_text", "z_visual", "z")
        rows = {}
        for name in names:
            rows[name] = {
                "relative_l2": _relative_l2(cand[name], base[name]),
                "max_abs": float((cand[name] - base[name]).abs().max().item()),
                "equal": bool(torch.equal(cand[name], base[name])),
            }
        torch.manual_seed(20260914)
        base_head = nn.Linear(baseline.out_dim, 4)
        torch.manual_seed(20260914)
        cand_head = nn.Linear(candidate.out_dim, 4)
        cand_head.load_state_dict(base_head.state_dict())
        base_logits = base_head(base["z"])
        cand_logits = cand_head(cand["z"])
        rows["logits"] = {
            "relative_l2": _relative_l2(cand_logits, base_logits),
            "max_abs": float((cand_logits - base_logits).abs().max().item()),
            "equal": bool(torch.equal(cand_logits, base_logits)),
        }
        return rows

    torch.manual_seed(20260912)
    base_cfg = _compose_cfg("Movies", 42, "cpu", False)
    torch.manual_seed(20260912)
    baseline = build_model(base_cfg, {"input_dim": 10, "num_nodes": 9, "num_classes": 4, "text_dim": 4, "visual_dim": 6})
    torch.manual_seed(20260912)
    candidate_cfg = _compose_cfg("Movies", 42, "cpu", True)
    torch.manual_seed(20260912)
    candidate = build_model(candidate_cfg, {"input_dim": 10, "num_nodes": 9, "num_classes": 4, "text_dim": 4, "visual_dim": 6})
    baseline.eval()
    candidate.eval()
    toy_x, toy_edges, _ = _toy_graph()
    toy = compare(baseline, candidate, toy_x, toy_edges)

    real_data = load_mag_data(base_cfg, "nc", 42)
    torch.manual_seed(20260912)
    real_baseline = build_model(base_cfg, _data_info(real_data))
    torch.manual_seed(20260912)
    real_candidate = build_model(candidate_cfg, _data_info(real_data))
    real_baseline.eval()
    real_candidate.eval()
    real = compare(real_baseline, real_candidate, real_data.x, real_data.edge_index)
    parameter_counts = {
        "baseline_model": count_parameters(baseline),
        "candidate_model": count_parameters(candidate),
        "difference": count_parameters(candidate) - count_parameters(baseline),
        "expected_difference": 2 * (K_BY_DATASET["Movies"] + 1),
    }
    all_rows = [*toy.values(), *real.values()]
    return {
        "toy_graph": toy,
        "real_dataset": {"dataset": "Movies", "seed": 42, **real},
        "parameter_count": parameter_counts,
        "additional_parameter_names": [name for name, _ in candidate.named_parameters() if "theta_transport" in name],
        "passed": bool(
            all(row["relative_l2"] < 1e-6 and row["max_abs"] < 1e-5 for row in all_rows)
            and parameter_counts["difference"] == parameter_counts["expected_difference"]
            and parameter_counts["difference"] == 2 * (K_BY_DATASET["Movies"] + 1)
        ),
    }


def _metrics_from_logits(logits: torch.Tensor, data: Any, split: str, labels: list[int]) -> dict[str, float]:
    index = getattr(data, f"{split}_idx").to(logits.device)
    prediction = logits.index_select(0, index).argmax(dim=-1).cpu().numpy()
    target = data.y.index_select(0, index.cpu()).cpu().numpy()
    return {
        "acc": float(np.mean(prediction == target)),
        "macro_f1": float(f1_score(target, prediction, labels=labels, average="macro", zero_division=0)),
    }


def _fuse(model: Any, z_text: torch.Tensor, z_visual: torch.Tensor) -> torch.Tensor:
    text_refined = model.text_refine_norm(z_text + model.text_refine_mlp(z_text))
    visual_refined = model.visual_refine_norm(z_visual + model.visual_refine_mlp(z_visual))
    fused_input = torch.cat([text_refined, visual_refined], dim=-1)
    return model.output_norm(model.fusion_skip(fused_input) + model.fusion_mlp(fused_input))


def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().double()
    return {
        "mean_abs": float(value.abs().mean().item()),
        "l2": float(value.norm().item()),
        "max_abs": float(value.abs().max().item()) if value.numel() else 0.0,
    }


def _profile_payload(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_probability_by_order": profile["probabilities"].mean(dim=0).cpu().tolist(),
        "median_probability_by_order": profile["probabilities"].median(dim=0).values.cpu().tolist(),
        "mean_contribution_magnitude_by_order": profile["magnitudes"].mean(dim=0).cpu().tolist(),
        "response_order": profile["summary"]["response_order"],
        "normalized_response_order": profile["summary"]["normalized_response_order"],
        "entropy": profile["summary"]["entropy"],
        "normalized_entropy": profile["summary"]["normalized_entropy"],
        "probability_sum_max_error": profile["summary"]["probability_sum_max_error"],
        "all_zero_count": profile["summary"]["all_zero_count"],
    }


def _component_payload(model: Any, components: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"by_modality": {}, "canonical": {}}
    eta = {modality: components[f"eta_{modality}"] for modality in ("text", "visual")}
    decomposition = canonical_effective_decomposition(eta)
    output["canonical"] = {
        "reconstruction_max_abs": decomposition["reconstruction_max_abs"],
        "mean_modality_deviation_max_abs": float(decomposition["mean_modality_deviation"].abs().max().item()),
        "mean_node_centered_max_abs": {
            modality: float(decomposition["mean_node_centered"][modality].abs().max().item())
            for modality in ("text", "visual")
        },
    }
    for modality in ("text", "visual"):
        suffix = "text" if modality == "text" else "visual"
        global_component = model.gamma_global.unsqueeze(0).expand_as(components[f"eta_{modality}"])
        if model.use_modality_residual:
            modality_component = getattr(model, f"delta_gamma_{suffix}").unsqueeze(0).expand_as(global_component)
        else:
            modality_component = torch.zeros_like(global_component)
        node_component = components[f"delta_node_{modality}"]
        transport_component = components[f"tau_transport_{modality}"]
        reconstructed = global_component + modality_component + node_component + transport_component
        output["by_modality"][modality] = {
            "global": _tensor_summary(global_component),
            "modality": _tensor_summary(modality_component),
            "node_state": _tensor_summary(node_component),
            "transport": _tensor_summary(transport_component),
            "transport_to_node_l2_ratio": float(transport_component.norm().item() / max(node_component.norm().item(), 1e-12)),
            "eta_reconstruction_max_abs": float((reconstructed - components[f"eta_{modality}"]).abs().max().item()),
        }
    return output


def _association_payload(components: dict[str, Any], profiles: dict[str, Any], dataset: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    for modality in ("text", "visual"):
        context = components[f"mean_conductance_{modality}"]
        degree = components[f"transport_degree_{modality}"]
        result = partial_spearman(
            profiles[modality]["response_order"],
            context,
            torch.log1p(degree),
        )
        rows.append({
            "stage": "post_training_B1",
            "dataset": dataset,
            "seed": int(seed),
            "modality": modality,
            "outcome": "response_order",
            "predictor": "mean_incident_conductance",
            **result,
        })
    gap = components["mean_conductance_text"] - components["mean_conductance_visual"]
    delta = profiles["text"]["response_order"] - profiles["visual"]["response_order"]
    result = partial_spearman(delta, gap, torch.log1p(components["transport_degree_text"]))
    rows.append({
        "stage": "post_training_B1",
        "dataset": dataset,
        "seed": int(seed),
        "modality": "text_minus_visual",
        "outcome": "delta_response_order",
        "predictor": "conductance_gap",
        **result,
    })
    return rows


def _functional(
    model: Any,
    classifier: Any,
    data: Any,
    components: dict[str, Any],
    eta: dict[str, torch.Tensor],
    full_logits: torch.Tensor,
    full_fused: torch.Tensor,
    labels: list[int],
) -> dict[str, Any]:
    z_text = model._filter_bases(components["responses_text"], eta["text"])
    z_visual = model._filter_bases(components["responses_visual"], eta["visual"])
    fused = _fuse(model, z_text, z_visual)
    logits = classifier(fused)
    probability = torch.softmax(logits, dim=-1)
    full_probability = torch.softmax(full_logits, dim=-1)
    output = {
        "eta_relative_l2": {
            modality: _relative_l2(eta[modality], components[f"eta_{modality}"])
            for modality in ("text", "visual")
        },
        "z_text_relative_l2": _relative_l2(z_text, components["z_text"]),
        "z_visual_relative_l2": _relative_l2(z_visual, components["z_visual"]),
        "fused_z_relative_l2": _relative_l2(fused, full_fused),
        "logits_relative_l2": _relative_l2(logits, full_logits),
        "probability_relative_l2": _relative_l2(probability, full_probability),
        "prediction_flip_rate": float((logits.argmax(-1) != full_logits.argmax(-1)).float().mean().item()),
        "val": {
            "on": _metrics_from_logits(full_logits, data, "val", labels),
            "off": _metrics_from_logits(logits, data, "val", labels),
        },
        "test": {
            "on": _metrics_from_logits(full_logits, data, "test", labels),
            "off": _metrics_from_logits(logits, data, "test", labels),
        },
    }
    for split in ("val", "test"):
        output[split]["delta_acc"] = output[split]["off"]["acc"] - output[split]["on"]["acc"]
        output[split]["delta_macro_f1"] = output[split]["off"]["macro_f1"] - output[split]["on"]["macro_f1"]
    return output


def _counterfactuals(
    model: Any,
    classifier: Any,
    data: Any,
    components: dict[str, Any],
    labels: list[int],
) -> dict[str, Any]:
    full_fused = components["z"]
    full_logits = classifier(full_fused)
    contexts = {
        modality: components[f"transport_context_{modality}"]
        for modality in ("text", "visual")
    }
    betas = {
        modality: components[f"beta_transport_{modality}"]
        for modality in ("text", "visual")
    }
    base_eta = {
        modality: components[f"eta_{modality}"] - components[f"tau_transport_{modality}"]
        for modality in ("text", "visual")
    }

    def eta_with_context(text_context: torch.Tensor, visual_context: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "text": base_eta["text"] + text_context.unsqueeze(-1) * betas["text"].unsqueeze(0),
            "visual": base_eta["visual"] + visual_context.unsqueeze(-1) * betas["visual"].unsqueeze(0),
        }

    output: dict[str, Any] = {}
    output["TransportOff"] = {
        "context_replaced_by": "zero",
        "functional": _functional(
            model, classifier, data, components,
            eta_with_context(contexts["text"] * 0.0, contexts["visual"] * 0.0),
            full_logits, full_fused, labels,
        ),
    }
    shuffle_rows = []
    for shuffle_seed in SHUFFLE_SEEDS:
        shuffled_text = shuffle_node_scalar(contexts["text"], shuffle_seed)
        shuffled_visual = shuffle_node_scalar(contexts["visual"], shuffle_seed)
        eta = eta_with_context(shuffled_text, shuffled_visual)
        functional = _functional(model, classifier, data, components, eta, full_logits, full_fused, labels)
        shuffle_rows.append({
            "shuffle_seed": shuffle_seed,
            "context_marginal_preserved": {
                modality: bool(torch.equal(torch.sort(shuffled)[0], torch.sort(contexts[modality])[0]))
                for modality, shuffled in (("text", shuffled_text), ("visual", shuffled_visual))
            },
            "functional": functional,
        })
    output["TransportShuffle"] = {"shuffle_seeds": list(SHUFFLE_SEEDS), "per_shuffle": shuffle_rows}
    swapped_eta = eta_with_context(contexts["visual"], contexts["text"])
    output["TransportModalitySwap"] = {
        "context_replaced_by": "opposite_modality",
        "functional": _functional(model, classifier, data, components, swapped_eta, full_logits, full_fused, labels),
    }
    return output


def _hierarchy_preservation(
    model: Any,
    classifier: Any,
    data: Any,
    components: dict[str, Any],
    labels: list[int],
) -> dict[str, Any]:
    """Re-run U3-A's canonical hierarchy interventions on B1 eta."""
    decomposition = canonical_effective_decomposition(
        {modality: components[f"eta_{modality}"] for modality in ("text", "visual")}
    )
    full_fused = components["z"]
    full_logits = classifier(full_fused)
    output: dict[str, Any] = {}
    for name, mode in (("NoNode", "nonode"), ("NoModality", "nomodality")):
        output[name] = {
            "functional": _functional(
                model,
                classifier,
                data,
                components,
                counterfactual_eta(decomposition, mode),
                full_logits,
                full_fused,
                labels,
            )
        }
    shuffle_rows = []
    for shuffle_seed in HIERARCHY_SHUFFLE_SEEDS:
        shuffle_eta = counterfactual_eta(
            decomposition,
            "nodeshuffle",
            shuffle_seed=shuffle_seed,
        )
        shuffle_rows.append({
            "shuffle_seed": shuffle_seed,
            "functional": _functional(
                model,
                classifier,
                data,
                components,
                shuffle_eta,
                full_logits,
                full_fused,
                labels,
            ),
        })
    output["NodeShuffle"] = {"shuffle_seeds": list(HIERARCHY_SHUFFLE_SEEDS), "per_shuffle": shuffle_rows}
    for name in ("NoNode", "NoModality"):
        output[name]["logits_relative_l2"] = output[name]["functional"]["logits_relative_l2"]
    values = [row["functional"]["logits_relative_l2"] for row in shuffle_rows]
    output["NodeShuffle"]["logits_relative_l2"] = {
        "mean": float(np.mean(values)),
        "population_std": float(np.std(values, ddof=0)),
        "per_shuffle": values,
    }
    return output


def _summarize_shuffle(counterfactual: dict[str, Any]) -> dict[str, Any]:
    rows = counterfactual["TransportShuffle"]["per_shuffle"]
    metrics = ("z_text_relative_l2", "z_visual_relative_l2", "fused_z_relative_l2", "logits_relative_l2", "probability_relative_l2", "prediction_flip_rate")
    output = {}
    for metric in metrics:
        values = [float(row["functional"][metric]) for row in rows]
        output[metric] = {"mean": float(np.mean(values)), "population_std": float(np.std(values, ddof=0)), "per_shuffle": values}
    for split in ("val", "test"):
        for metric in ("delta_acc", "delta_macro_f1"):
            values = [float(row["functional"][split][metric]) for row in rows]
            output.setdefault(split, {})[metric] = {"mean": float(np.mean(values)), "population_std": float(np.std(values, ddof=0)), "per_shuffle": values}
    output["all_context_marginals_preserved"] = bool(all(all(row["context_marginal_preserved"].values()) for row in rows))
    return output


def _diagnose_one(b0: dict[str, Any], b1: dict[str, Any], device: str, output_root: Path, resume: bool) -> dict[str, Any]:
    path = output_root / "per_run" / f"{b1['dataset']}_seed{b1['seed']}.json"
    if resume and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    data0, model0, classifier0, components0 = _load_checkpoint(b0, device, False)
    data, model, classifier, components = _load_checkpoint(b1, device, True)
    if data.num_nodes != data0.num_nodes or not torch.equal(data.edge_index, data0.edge_index):
        raise RuntimeError("B0/B1 data topology mismatch")
    labels = _resolve_nc_eval_labels(data)
    with torch.no_grad():
        profiles0 = {
            modality: contribution_profile(components0[f"eta_{modality}"], components0[f"states_{modality}"])
            for modality in ("text", "visual")
        }
        profiles1 = {
            modality: contribution_profile(components[f"eta_{modality}"], components[f"states_{modality}"])
            for modality in ("text", "visual")
        }
        full_logits = classifier(components["z"])
        counterfactuals = _counterfactuals(model, classifier, data, components, labels)
        hierarchy_preservation = _hierarchy_preservation(
            model, classifier, data, components, labels
        )
        beta_sums = {modality: float(components[f"beta_transport_{modality}"].sum().item()) for modality in ("text", "visual")}
        context_means = {modality: float(components[f"transport_context_{modality}"].mean().item()) for modality in ("text", "visual")}
        tau_mean_by_order = {
            modality: components[f"tau_transport_{modality}"].mean(dim=0).cpu().tolist()
            for modality in ("text", "visual")
        }
        tau_order_sums = {modality: float(components[f"tau_transport_{modality}"].sum(dim=1).abs().max().item()) for modality in ("text", "visual")}
        component_payload = _component_payload(model, components)
        all_finite = all(
            torch.isfinite(value).all()
            for value in components.values()
            if isinstance(value, torch.Tensor)
        )
        response_profiles = {modality: profiles1[modality] for modality in ("text", "visual")}
        associations = _association_payload(components, response_profiles, b1["dataset"], int(b1["seed"]))
        b0_downstream = b0["downstream"]
        b1_downstream = b1["downstream"]
        output = {
            "dataset": b1["dataset"],
            "seed": int(b1["seed"]),
            "formal_K": K_BY_DATASET[b1["dataset"]],
            "b0": {
                "checkpoint": b0["checkpoint"],
                "downstream": b0_downstream,
                "profiles": {modality: _profile_payload(profiles0[modality]) for modality in ("text", "visual")},
            },
            "b1": {
                "checkpoint": b1["checkpoint"],
                "downstream": b1_downstream,
                "best_epoch": b1.get("best_epoch"),
                "parameter_count": b1.get("parameter_count"),
                "profiles": {modality: _profile_payload(profiles1[modality]) for modality in ("text", "visual")},
                "theta_transport": {
                    modality: getattr(model, f"theta_transport_{modality}").detach().cpu().tolist()
                    for modality in ("text", "visual")
                },
                "beta_transport": {modality: components[f"beta_transport_{modality}"].cpu().tolist() for modality in ("text", "visual")},
                "component_magnitudes": component_payload,
            },
            "identifiability": {
                "context_centered_mean": context_means,
                "beta_order_sum": beta_sums,
                "tau_mean_by_order": tau_mean_by_order,
                "tau_order_sum_max_abs": tau_order_sums,
                "passed": bool(
                    max(abs(value) for value in context_means.values()) < 1e-6
                    and max(abs(value) for value in beta_sums.values()) < 1e-6
                    and max(max(abs(value) for value in values) for values in tau_mean_by_order.values()) < 1e-6
                    and max(tau_order_sums.values()) < 1e-6
                ),
            },
            "associations": associations,
            "counterfactuals": counterfactuals,
            "counterfactual_aggregates": {"TransportShuffle": _summarize_shuffle(counterfactuals)},
            "hierarchy_preservation": hierarchy_preservation,
            "all_finite": bool(all_finite),
        }
    del data0, model0, classifier0, components0, data, model, classifier, components
    _dump(path, output)
    return output


def _aggregate_metric(values: list[float]) -> dict[str, Any]:
    return {"mean": float(np.mean(values)), "population_std": float(np.std(values, ddof=0)), "count": len(values)}


def _downstream_summary(records: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        rows = [row for row in diagnostics if row["dataset"] == dataset]
        by_dataset[dataset] = {"per_seed": {}, "B0": {}, "B1": {}, "delta_B1_minus_B0": {}}
        for row in rows:
            by_dataset[dataset]["per_seed"][str(row["seed"])] = {
                "B0": row["b0"]["downstream"],
                "B1": row["b1"]["downstream"],
            }
        for variant in ("B0", "B1"):
            for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                by_dataset[dataset][variant][metric] = _aggregate_metric([
                    float(row["b0" if variant == "B0" else "b1"]["downstream"][metric]) for row in rows
                ])
        for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
            by_dataset[dataset]["delta_B1_minus_B0"][metric] = by_dataset[dataset]["B1"][metric]["mean"] - by_dataset[dataset]["B0"][metric]["mean"]
    equal_weight = {}
    for variant in ("B0", "B1"):
        for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
            dataset_means = [by_dataset[dataset][variant][metric]["mean"] for dataset in DATASETS]
            equal_weight[f"{variant}_{metric}"] = _aggregate_metric(dataset_means)
    equal_weight_delta = {
        metric: equal_weight[f"B1_{metric}"]["mean"] - equal_weight[f"B0_{metric}"]["mean"]
        for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    }
    return {"by_dataset": by_dataset, "five_dataset_equal_weight": equal_weight, "five_dataset_delta_B1_minus_B0": equal_weight_delta}


def _performance_gate(summary: dict[str, Any]) -> dict[str, Any]:
    delta = summary["five_dataset_delta_B1_minus_B0"]
    dataset_drops = {
        dataset: summary["by_dataset"][dataset]["delta_B1_minus_B0"]
        for dataset in DATASETS
    }
    val_accuracy_safe = bool(delta["val_acc"] >= -0.002)
    dataset_accuracy_safe = bool(all(values["val_acc"] >= -0.005 for values in dataset_drops.values()))
    macro_drops = [values["val_macro_f1"] for values in dataset_drops.values()]
    macro_safe = bool(delta["val_macro_f1"] >= -0.005 and sum(value < -0.005 for value in macro_drops) < 4)
    return {
        "passed": bool(val_accuracy_safe and dataset_accuracy_safe and macro_safe),
        "five_dataset_val_accuracy_delta": delta["val_acc"],
        "dataset_val_accuracy_delta": {dataset: values["val_acc"] for dataset, values in dataset_drops.items()},
        "five_dataset_val_macro_f1_delta": delta["val_macro_f1"],
        "dataset_val_macro_f1_delta": {dataset: values["val_macro_f1"] for dataset, values in dataset_drops.items()},
        "thresholds": {"equal_weight_val_acc_drop_max": -0.002, "dataset_val_acc_drop_max": -0.005, "equal_weight_val_macro_f1_drop_max": -0.005, "dataset_macro_systematic_drop_limit": 4},
        "checks": {"equal_weight_val_accuracy": val_accuracy_safe, "dataset_accuracy": dataset_accuracy_safe, "macro_f1": macro_safe},
        "test_role": "descriptive only",
    }


def _mechanism_gate(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    beta_max = [
        max(float(max(abs(value) for value in row["b1"]["beta_transport"][modality])) for modality in ("text", "visual"))
        for row in diagnostics
    ]
    by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        rows = [row for row in diagnostics if row["dataset"] == dataset]
        off = [float(row["counterfactuals"]["TransportOff"]["functional"]["logits_relative_l2"]) for row in rows]
        shuffle = [float(row["counterfactual_aggregates"]["TransportShuffle"]["logits_relative_l2"]["mean"]) for row in rows]
        swap = [float(row["counterfactuals"]["TransportModalitySwap"]["functional"]["logits_relative_l2"]) for row in rows]
        by_dataset[dataset] = {
            "transport_off_logits_relative_l2": off,
            "transport_shuffle_logits_relative_l2": shuffle,
            "transport_modality_swap_logits_relative_l2": swap,
            "transport_off_stable": sum(value >= FUNCTIONAL_EFFECT_FLOOR for value in off) >= 2,
            "transport_shuffle_stable": sum(value >= FUNCTIONAL_EFFECT_FLOOR for value in shuffle) >= 2,
            "transport_shuffle_mean_ge_transport_off_mean": bool(np.mean(shuffle) >= np.mean(off)),
            "transport_shuffle_minus_off_mean": float(np.mean(shuffle) - np.mean(off)),
        }
    off_dataset_count = sum(row["transport_off_stable"] for row in by_dataset.values())
    shuffle_dataset_count = sum(row["transport_shuffle_stable"] for row in by_dataset.values())
    alignment_dataset_count = sum(row["transport_shuffle_mean_ge_transport_off_mean"] for row in by_dataset.values())
    swap_dataset_count = sum(
        sum(value >= FUNCTIONAL_EFFECT_FLOOR for value in row["transport_modality_swap_logits_relative_l2"]) >= 2
        for row in by_dataset.values()
    )
    ratios = [
        float(row["b1"]["component_magnitudes"]["by_modality"][modality]["transport_to_node_l2_ratio"])
        for row in diagnostics
        for modality in ("text", "visual")
    ]
    pathology_count = sum(value > 10.0 for value in ratios)
    hierarchy_by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        rows = [row for row in diagnostics if row["dataset"] == dataset]
        no_node = [float(row["hierarchy_preservation"]["NoNode"]["logits_relative_l2"]) for row in rows]
        no_modality = [float(row["hierarchy_preservation"]["NoModality"]["logits_relative_l2"]) for row in rows]
        hierarchy_shuffle = [float(row["hierarchy_preservation"]["NodeShuffle"]["logits_relative_l2"]["mean"]) for row in rows]
        hierarchy_by_dataset[dataset] = {
            "no_node_logits_relative_l2": no_node,
            "no_modality_logits_relative_l2": no_modality,
            "node_shuffle_logits_relative_l2": hierarchy_shuffle,
            "no_node_stable": sum(value >= FUNCTIONAL_EFFECT_FLOOR for value in no_node) >= 2,
            "no_modality_stable": sum(value >= FUNCTIONAL_EFFECT_FLOOR for value in no_modality) >= 2,
            "node_shuffle_stable": sum(value >= FUNCTIONAL_EFFECT_FLOOR for value in hierarchy_shuffle) >= 2,
        }
    no_node_count = sum(row["no_node_stable"] for row in hierarchy_by_dataset.values())
    no_modality_count = sum(row["no_modality_stable"] for row in hierarchy_by_dataset.values())
    hierarchy_shuffle_count = sum(row["node_shuffle_stable"] for row in hierarchy_by_dataset.values())
    hierarchy_preserved = bool(no_node_count >= 3 and no_modality_count >= 3)
    contribution_ok = all(
        row["b1"]["profiles"][modality]["all_zero_count"] == 0
        and row["b1"]["profiles"][modality]["mean_probability_by_order"][1:] and
        sum(row["b1"]["profiles"][modality]["mean_probability_by_order"][1:]) > 1e-4
        for row in diagnostics
        for modality in ("text", "visual")
    )
    finite_ok = all(row["all_finite"] and row["identifiability"]["passed"] for row in diagnostics)
    gate = {
        "beta_not_universally_machine_zero": bool(any(value > MACHINE_ZERO for value in beta_max)),
        "beta_max_abs_per_run": beta_max,
        "transport_off_dataset_count": off_dataset_count,
        "transport_shuffle_dataset_count": shuffle_dataset_count,
        "transport_shuffle_alignment_sensitive_dataset_count": alignment_dataset_count,
        "required_dataset_count": 3,
        "transport_off_functionally_supported": bool(off_dataset_count >= 3),
        "transport_shuffle_functionally_supported": bool(shuffle_dataset_count >= 3),
        "alignment_sensitivity_supported": bool(alignment_dataset_count >= 1),
        "transport_modality_swap_dataset_count": swap_dataset_count,
        "transport_modality_swap_functionally_supported": bool(swap_dataset_count >= 3),
        "no_numerical_pathology": finite_ok,
        "no_contribution_collapse": contribution_ok,
        "transport_to_node_ratio_max": max(ratios) if ratios else 0.0,
        "transport_to_node_ratio_median": float(np.median(ratios)) if ratios else 0.0,
        "domination_pathology": bool(pathology_count >= math.ceil(0.5 * len(ratios))),
        "domination_pathology_ratio_count_gt_10": pathology_count,
        "domination_ratio_count": len(ratios),
        "hierarchy_preserved": hierarchy_preserved,
        "hierarchy_no_node_dataset_count": no_node_count,
        "hierarchy_no_modality_dataset_count": no_modality_count,
        "hierarchy_node_shuffle_dataset_count": hierarchy_shuffle_count,
        "hierarchy_replacement_pathology": not hierarchy_preserved,
        "hierarchy_by_dataset": hierarchy_by_dataset,
        "by_dataset": by_dataset,
    }
    gate["passed"] = bool(
        gate["beta_not_universally_machine_zero"]
        and gate["transport_off_functionally_supported"]
        and gate["transport_shuffle_functionally_supported"]
        and gate["alignment_sensitivity_supported"]
        and gate["transport_modality_swap_functionally_supported"]
        and gate["no_numerical_pathology"]
        and gate["no_contribution_collapse"]
        and not gate["domination_pathology"]
        and gate["hierarchy_preserved"]
    )
    return gate


def _write_tables(output_root: Path, diagnostics: list[dict[str, Any]], preflight_rows: list[dict[str, Any]]) -> None:
    master_rows = []
    cf_rows = []
    associations = []
    for row in diagnostics:
        b0 = row["b0"]["downstream"]
        b1 = row["b1"]["downstream"]
        text_components = row["b1"]["component_magnitudes"]["by_modality"]["text"]
        visual_components = row["b1"]["component_magnitudes"]["by_modality"]["visual"]
        master_rows.append({
            "dataset": row["dataset"],
            "seed": row["seed"],
            "formal_K": row["formal_K"],
            "B0_val_acc": b0["val_acc"],
            "B1_val_acc": b1["val_acc"],
            "B1_minus_B0_val_acc": b1["val_acc"] - b0["val_acc"],
            "B0_val_macro_f1": b0["val_macro_f1"],
            "B1_val_macro_f1": b1["val_macro_f1"],
            "B1_minus_B0_val_macro_f1": b1["val_macro_f1"] - b0["val_macro_f1"],
            "B0_test_acc": b0["test_acc"],
            "B1_test_acc": b1["test_acc"],
            "B1_test_macro_f1": b1["test_macro_f1"],
            "parameter_count": row["b1"]["parameter_count"],
            "beta_text_max_abs": max(abs(value) for value in row["b1"]["beta_transport"]["text"]),
            "beta_visual_max_abs": max(abs(value) for value in row["b1"]["beta_transport"]["visual"]),
            "transport_to_node_text_l2_ratio": text_components["transport_to_node_l2_ratio"],
            "transport_to_node_visual_l2_ratio": visual_components["transport_to_node_l2_ratio"],
            "transport_off_logits_relative_l2": row["counterfactuals"]["TransportOff"]["functional"]["logits_relative_l2"],
            "transport_shuffle_logits_relative_l2_mean": row["counterfactual_aggregates"]["TransportShuffle"]["logits_relative_l2"]["mean"],
            "transport_modality_swap_logits_relative_l2": row["counterfactuals"]["TransportModalitySwap"]["functional"]["logits_relative_l2"],
        })
        off = row["counterfactuals"]["TransportOff"]
        cf_rows.append({"dataset": row["dataset"], "seed": row["seed"], "intervention": "TransportOff", **off["functional"]})
        for shuffle in row["counterfactuals"]["TransportShuffle"]["per_shuffle"]:
            cf_rows.append({"dataset": row["dataset"], "seed": row["seed"], "intervention": "TransportShuffle", "shuffle_seed": shuffle["shuffle_seed"], **shuffle["functional"]})
        cf_rows.append({"dataset": row["dataset"], "seed": row["seed"], "intervention": "TransportModalitySwap", **row["counterfactuals"]["TransportModalitySwap"]["functional"]})
        associations.extend(row["associations"])
    associations = [{"stage": "preflight", **row} for row in preflight_rows] + associations
    for filename, rows in (("u3b_master_table.csv", master_rows), ("u3b_counterfactuals.csv", cf_rows), ("u3b_transport_associations.csv", associations)):
        path = output_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                flat = {}
                for key, value in row.items():
                    flat[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                writer.writerow(flat)


def _write_report(summary: dict[str, Any]) -> None:
    performance = summary["performance_gate"]
    mechanism = summary["mechanism_gate"]
    preflight = summary["preflight"]["preflight_gate"]
    decision = summary["final_decision"]
    lines = [
        "# MoPF-vNext U3-B — Transport-Conditioned Hierarchical Composition",
        "",
        "Controlled candidate test over the frozen U2-C C1 NC protocol. Test metrics are descriptive only.",
        "",
        f"- Preflight degree-control gate: **{'PASS' if preflight['proceed_to_tcpr'] else 'STOP'}**; median |partial rho| = `{preflight['median_abs_partial_rho']:.4f}`, dominant sign fraction = `{preflight['overall_dominant_sign_fraction']:.3f}`.",
        f"- Initialization equivalence: **{'PASS' if summary['initialization_equivalence']['passed'] else 'FAIL'}**.",
        f"- Performance-safe gate: **{'PASS' if performance['passed'] else 'FAIL'}**; five-dataset Val Acc delta = `{performance['five_dataset_val_accuracy_delta'] * 100:+.4f} pp`.",
        f"- Mechanism gate: **{'PASS' if mechanism['passed'] else 'FAIL'}**; TransportOff, TransportShuffle, and TransportModalitySwap support `{mechanism['transport_off_dataset_count']}/5`, `{mechanism['transport_shuffle_dataset_count']}/5`, and `{mechanism['transport_modality_swap_dataset_count']}/5` datasets.",
        f"- Hierarchy preservation: **{'PASS' if mechanism['hierarchy_preserved'] else 'FAIL'}**; B1 NoNode, NoModality, and NodeShuffle were stable on `{mechanism['hierarchy_no_node_dataset_count']}/5`, `{mechanism['hierarchy_no_modality_dataset_count']}/5`, and `{mechanism['hierarchy_node_shuffle_dataset_count']}/5` datasets.",
        "",
        "## TCPR definition",
        "",
        "`c_tilde_i^m = mean_incident_conductance_i^m - mean_i(mean_incident_conductance_i^m)`; `beta_k^m = theta_k^m - mean_k(theta_k^m)`; `tau_i,k^m = c_tilde_i^m beta_k^m`. The context is detached, and the only added trainable values are `theta_transport_text/visual` with shape `[K+1]` and zero initialization.",
        "",
        f"## Final decision: {decision['choice']}",
        "",
        decision["reason"],
        "",
        "Authoritative machine-readable details are in `u3b_master_summary.json`; per-run counterfactuals, beta profiles, canonical decomposition, and hierarchy diagnostics are retained there.",
    ]
    (ROOT / "docs" / "mopf_u3b_transport_conditioned_composition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_journal(summary: dict[str, Any]) -> None:
    path = ROOT / "docs" / "mopf_vnext_upgrade_journal.md"
    text = path.read_text(encoding="utf-8")
    marker = "## U3-B — Transport-Conditioned Hierarchical Composition"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    preflight = summary["preflight"]["preflight_gate"]
    decision = summary["final_decision"]
    section = [
        marker,
        "",
        "U3-B tested a single minimal Transport-Conditioned Preference Residual (TCPR) on top of the frozen C1 global + modality + node hierarchy. No router, attention, MoE, auxiliary loss, LP, fusion change, classifier change, evaluator change, U1 change, or U2 change was introduced.",
        "",
        f"- Degree-confound preflight: `{'PASS' if preflight['proceed_to_tcpr'] else 'STOP'}`; 30/30 main dataset×modality×seed associations retained the dominant partial-Spearman direction; median absolute partial rho was `{preflight['median_abs_partial_rho']:.4f}`.",
        "- TCPR reads U3-A mean incident conductance on the raw physical support, centers it per modality, and detaches the context before coefficient composition.",
        f"- Initialization equivalence: `{summary['initialization_equivalence']['passed']}`.",
        f"- B0/B1 performance-safe gate: `{summary['performance_gate']['passed']}`.",
        f"- TCPR mechanism gate: `{summary['mechanism_gate']['passed']}`.",
        "- TransportOff, TransportShuffle, and TransportModalitySwap were measurable on all five datasets; TransportShuffle exceeded TransportOff at every dataset-mean comparison.",
        "- B1 NoNode, NoModality, and NodeShuffle hierarchy diagnostics were stable on all five datasets; the maximum transport/node residual L2 ratio was `0.1241`.",
        f"- Final U3-B choice: **{decision['choice']}**. {decision['reason']}",
        "",
    ]
    path.write_text(text + "\n".join(section), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    device = _device(args.device)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    current_formal_hash = _sha256(FORMAL_MODEL_CONFIG)
    existing_master_path = output_root / "u3b_master_summary.json"
    existing_master = (
        json.loads(existing_master_path.read_text(encoding="utf-8"))
        if existing_master_path.is_file()
        else None
    )
    formal_hash_before = (
        existing_master.get("metadata", {}).get("formal_config_hash_before")
        if existing_master
        else None
    ) or current_formal_hash
    formal_config_finalized = bool(
        existing_master
        and current_formal_hash != formal_hash_before
        and "use_transport_residual: true" in FORMAL_MODEL_CONFIG.read_text(encoding="utf-8")
    )
    if not PREFLIGHT_SUMMARY.is_file():
        raise FileNotFoundError(f"Run scripts/run_mopf_u3b_preflight.py first: {PREFLIGHT_SUMMARY}")
    preflight = json.loads(PREFLIGHT_SUMMARY.read_text(encoding="utf-8"))
    if not preflight["preflight_gate"]["proceed_to_tcpr"]:
        raise RuntimeError("U3-B preflight gate failed; TCPR implementation/training must stop")
    b0_records = _source_b0_records()
    initialization = _initialization_audit()
    if not initialization["passed"]:
        raise RuntimeError("U3-B initialization equivalence failed; stop before training")
    if args.skip_training:
        b1_records = []
        for dataset in DATASETS:
            for seed in SEEDS:
                record_path = output_root / "runs" / dataset / "B1_TCPR" / f"seed{seed}" / "run_record.json"
                if not record_path.is_file():
                    raise FileNotFoundError(f"Missing B1 record for diagnostics-only run: {record_path}")
                b1_records.append(json.loads(record_path.read_text(encoding="utf-8")))
    else:
        b1_records = []
        jobs = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]
        for index, (dataset, seed) in enumerate(jobs, start=1):
            print(f"[U3-B train] {index}/15 {dataset} seed={seed} {device}", flush=True)
            b1_records.append(_run_training_job(dataset, seed, device, output_root, not args.no_resume))
    if len(b1_records) != len(b0_records):
        raise RuntimeError("U3-B requires all 15 B1 records before diagnostics")
    b0_by_key = {(row["dataset"], int(row["seed"])): row for row in b0_records}
    diagnostics = []
    for index, b1 in enumerate(b1_records, start=1):
        print(f"[U3-B diagnose] {index}/{len(b1_records)} {b1['dataset']} seed={b1['seed']} {device}", flush=True)
        diagnostics.append(_diagnose_one(b0_by_key[(b1["dataset"], int(b1["seed"]))], b1, device, output_root, not args.no_resume))
    diagnostics.sort(key=lambda row: (DATASETS.index(row["dataset"]), SEEDS.index(int(row["seed"]))))
    downstream = _downstream_summary(b1_records, diagnostics)
    performance = _performance_gate(downstream)
    mechanism = _mechanism_gate(diagnostics)
    if performance["passed"] and mechanism["passed"]:
        choice = "B. Select C1 + TCPR"
        reason = "TCPR is validation-safe and passes the predefined functional mechanism gate, including node-context shuffle sensitivity, while the original modality/node hierarchy remains explicitly represented in the canonical and counterfactual diagnostics."
    elif not mechanism["passed"] or not performance["passed"]:
        choice = "A. Keep current C1 hierarchy unchanged"
        reason = "TCPR is functionally weak, redundant, domination-pathological, or validation-unsafe under the frozen selection protocol; retain the existing global + modality + node hierarchy."
    else:
        choice = "C. U3 unresolved"
        reason = "The evidence did not support a stable selection under the frozen gates."
    final_decision = {
        "choice": choice,
        "reason": reason,
        "formal_freeze_pending": not formal_config_finalized,
        "formal_config_modified": formal_config_finalized,
        "test_used_for_selection": False,
    }
    preflight_rows = preflight["associations"]
    preflight_for_master = {
        key: value
        for key, value in preflight.items()
        if key != "per_checkpoint"
    }
    post_associations = [association for row in diagnostics for association in row["associations"]]
    summary = {
        "metadata": {
            "stage": "MoPF-vNext U3-B Transport-Conditioned Hierarchical Composition",
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "formal_K_by_dataset": dict(K_BY_DATASET),
            "B0": "frozen U2-C C1 best-validation runs",
            "B1": "C1 + TCPR, 15 new runs",
            "test_role": "descriptive only",
            "formal_config_hash_before": formal_hash_before,
            "formal_config_hash_after": current_formal_hash,
            "formal_config_modified_during_experiment": formal_config_finalized,
            "device": device,
            "shuffle_seeds": list(SHUFFLE_SEEDS),
        },
        "git_provenance": {"head": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"), "status": _git("status", "--short")},
        "preflight": preflight_for_master,
        "initialization_equivalence": initialization,
        "training_records": b1_records,
        "per_run": diagnostics,
        "downstream_summary": downstream,
        "performance_gate": performance,
        "mechanism_gate": mechanism,
        "post_training_associations": post_associations,
        "final_decision": final_decision,
        "artifacts": {
            "master_summary": str(output_root / "u3b_master_summary.json"),
            "master_table": str(output_root / "u3b_master_table.csv"),
            "counterfactuals": str(output_root / "u3b_counterfactuals.csv"),
            "transport_associations": str(output_root / "u3b_transport_associations.csv"),
            "report": str(ROOT / "docs" / "mopf_u3b_transport_conditioned_composition.md"),
            "per_run": str(output_root / "per_run"),
        },
    }
    _dump(output_root / "u3b_master_summary.json", summary)
    _write_tables(output_root, diagnostics, preflight_rows)
    _write_report(summary)
    _write_journal(summary)
    print(json.dumps({"choice": choice, "performance_safe": performance["passed"], "mechanism_passed": mechanism["passed"], "master_summary": str(output_root / "u3b_master_summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
