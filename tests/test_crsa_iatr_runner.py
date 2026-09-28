from __future__ import annotations

import json

import torch

import scripts.run_crsa_iatr_nc as runner
from scripts.analyze_crsa_iatr_nc import _manifest_contexts, _paired_rows, _summary_rows


def test_formal_matrix_is_fixed_and_commands_are_validation_only():
    contexts = runner.formal_contexts("cuda:1")
    assert len(contexts) == 60
    assert len({context.run_key for context in contexts}) == 60
    assert {context.variant for context in contexts} == set(runner.VARIANTS)
    assert {context.dataset for context in contexts} == set(runner.DATASETS)
    assert {context.seed for context in contexts} == set(runner.SEEDS)
    command = runner.build_command(contexts[0])
    assert "task.evaluate_test=false" in command
    assert "task.eval_modality_masks.enabled=false" in command
    assert contexts[0].run_dir != contexts[1].run_dir
    assert contexts[0].checkpoint != contexts[1].checkpoint


def test_smoke_namespace_is_separate_from_formal(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "outputs")
    smoke = runner.smoke_contexts("Movies", ["base", "full"], 42, 1, "cuda:1")
    formal = runner.formal_contexts("cuda:1")
    assert smoke[0].run_dir.is_relative_to(tmp_path / "outputs" / "smoke")
    assert smoke[0].checkpoint.is_relative_to(tmp_path / "outputs" / "smoke")
    assert smoke[0].run_dir != formal[0].run_dir


def _smoke_artifacts(context):
    model = {
        "name": "crsa_iatr", "variant": context.variant,
        "use_crsa": False, "use_iatr": False,
        "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
        "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
        "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
        "num_residual_adapters": 3, "edge_chunk_size": context.edge_chunk_size,
        "trajectory_heads": 4, "trajectory_layers": 1, "trajectory_ffn_dim": 512,
        "anchor_heads": 4, "anchor_ffn_dim": 512,
        "iatr_node_chunk_size": context.iatr_node_chunk_size,
    }
    task = {
        "name": "nc", "protocol_version": "unified_full_graph_nc_v1",
        "training_mode": "full_graph", "optimizer": "adamw", "epochs": context.epochs,
        "patience": 30, "early_stop_min_epoch": 1, "early_stop_min_delta": 1e-4,
        "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0, "eval_every": 1,
        "inference_mode": "full", "scheduler": None, "evaluate_test": False,
        "eval_modality_masks": {"enabled": False},
    }
    config = {
        "model": model, "task": task, "dataset": {"name": context.dataset},
        "seed": context.seed, "num_runs": 1,
    }
    metrics = {
        "metrics": {
            "val_acc": {"mean": 0.5, "std": 0.0},
            "val_macro_f1": {"mean": 0.4, "std": 0.0},
        },
        "best_epoch": 1, "checkpoint_selection": "best_val_accuracy",
    }
    context.run_dir.mkdir(parents=True, exist_ok=True)
    context.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    (context.run_dir / "resolved_config.json").write_text(json.dumps(config))
    (context.run_dir / "metrics.json").write_text(json.dumps(metrics))
    (context.run_dir / "complete.marker").write_text(json.dumps({
        "task": "nc", "dataset": context.dataset, "seed": context.seed,
    }))
    torch.save({
        "task": "nc", "seed": context.seed, "selection": "best_val_accuracy", "epoch": 1,
        "metrics": {"val_acc": 0.5, "val_macro_f1": 0.4},
        "model_state": {"weight": torch.ones(1)},
        "head_state": {"weight": torch.ones(1)}, "data_info": {},
    }, context.checkpoint)
    return metrics


def test_completion_predicate_requires_test_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path / "outputs")
    context = runner.smoke_contexts("Movies", ["base"], 42, 1, "cuda:1")[0]
    metrics = _smoke_artifacts(context)
    assert runner.is_complete(context)
    metrics["metrics"]["test_acc"] = {"mean": 0.9, "std": 0.0}
    (context.run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert not runner.is_complete(context)


def test_summary_and_paired_deltas_are_descriptive():
    rows = []
    values = {
        "base": (0.60, 0.50), "crsa": (0.62, 0.52),
        "iatr": (0.61, 0.51), "full": (0.66, 0.55),
    }
    for variant, (acc, f1) in values.items():
        rows.append({
            "variant": variant, "dataset": "Movies", "seed": 42,
            "status": "complete", "val_acc": acc, "val_macro_f1": f1,
        })
    summary = _summary_rows(rows)
    assert len(summary) == 8
    assert any(item["dataset"] == "ALL_DATASETS" for item in summary)
    deltas = _paired_rows(rows)
    full_base = next(row for row in deltas if row["contrast"] == "Full-Base"
                     and row["metric"] == "val_acc")
    interaction = next(row for row in deltas if row["contrast"].startswith("Interaction-")
                       and row["metric"] == "val_acc")
    assert abs(full_base["delta"] - 0.06) < 1e-8
    assert abs(interaction["delta"] - 0.03) < 1e-8
    assert all("PASS" not in str(row) and "FAIL" not in str(row) for row in summary)


def test_formal_manifest_context_keeps_namespace_for_provenance():
    contexts = _manifest_contexts("formal", [])
    assert len(contexts) == 60
    assert contexts[0]["mode"] == "formal"
    assert contexts[0]["run_dir"].endswith("/runs/base/Movies/seed42")
