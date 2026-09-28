from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

from scripts.run_relation_residual_audit import (
    Context,
    VARIANTS,
    build_command,
    formal_contexts,
    is_complete,
    smoke_contexts,
)
from src.models.relation_operator_audit import Model as OldModel
from src.models.relation_residual_audit import Model
from src.tasks.common import summarize_aux_info_stats, update_aux_info_stats


def _cfg(variant: str, chunk: int = 4):
    return OmegaConf.create({
        "model": {
            "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
            "residual_variant": variant, "relation_dim": 64, "relation_heads": 4,
            "relation_ffn_dim": 128, "shared_bottleneck_dim": 256,
            "adapter_bottleneck_dim": 32, "num_residual_adapters": 3,
            "edge_chunk_size": chunk,
        }
    })


def _toy(seed: int = 7):
    torch.manual_seed(seed)
    info = {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_nodes": 8, "num_classes": 3}
    x = torch.randn(8, 6)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7, 0, 1],
    ])
    return info, x, edge_index


def _make(variant: str, seed: int = 7, chunk: int = 4):
    info, x, edges = _toy(seed)
    torch.manual_seed(seed)
    model = Model(_cfg(variant, chunk), info).eval()
    return model, x, edges


def test_t1_effective_physical_operator_exactly_matches_p0():
    info, x, edges = _toy()
    old = OldModel(OmegaConf.create({"model": {
        "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
        "operator_variant": "plain", "edge_chunk_size": 4,
    }}), info).eval()
    new = Model(_cfg("shared_prior"), info).eval()
    old_p = old._get_operator(edges, x.size(0), x.dtype)
    new_p = new._get_operator(edges, x.size(0), x.dtype)
    assert torch.equal(old_p.indices(), new_p.indices())
    assert torch.equal(old_p.values(), new_p.values())


def test_t2_shared_prior_initialization_is_near_plain():
    info, x, edges = _toy()
    torch.manual_seed(101)
    old = OldModel(OmegaConf.create({"model": {
        "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
        "operator_variant": "plain", "edge_chunk_size": 4,
    }}), info).eval()
    torch.manual_seed(101)
    new = Model(_cfg("shared_prior"), info).eval()
    old_z = old(x, edges)[0]
    new_z = new(x, edges)[0]
    assert (old_z - new_z).abs().max() < 0.01


def test_t3_self_loops_bypass_relation_shared_and_adapter_modules():
    model, x, edges = _make("context_residual")
    relation_edges, shared_batch_sizes = [], []
    adapter_batch_sizes = [[], [], []]
    handles = [
        model.relation_text.register_forward_pre_hook(
            lambda module, args: relation_edges.append((args[2].clone(), args[3].clone()))
        ),
        model.shared_text.register_forward_pre_hook(
            lambda module, args: shared_batch_sizes.append(args[0].size(0))
        ),
    ]
    for index, adapter in enumerate(model.adapters_text):
        handles.append(adapter.register_forward_pre_hook(
            lambda module, args, i=index: adapter_batch_sizes[i].append(args[0].size(0))
        ))
    with torch.no_grad():
        out = model.analyze(x, edges)
    for handle in handles:
        handle.remove()
    target, source = model._operator_edges(model._get_operator(edges, x.size(0), x.dtype))[:2]
    assert relation_edges and all(not torch.any(t == s) for t, s in relation_edges)
    assert sum(t.numel() for t, _ in relation_edges) == target.numel()
    assert sum(shared_batch_sizes) == 3 * target.numel()
    assert all(sum(sizes) == 3 * target.numel() for sizes in adapter_batch_sizes)
    assert out["num_self_edges"] == x.size(0)
    assert out["num_nonself_edges"] == target.numel()


def test_t4_residual_variants_have_exact_parameter_parity():
    models = [_make(variant)[0] for variant in VARIANTS[1:]]
    counts = [sum(p.numel() for p in model.parameters() if p.requires_grad) for model in models]
    assert counts[0] == counts[1] == counts[2]


def test_t5_structural_response_matches_dense_nonself_operator():
    model, x, edges = _make("context_residual")
    with torch.no_grad():
        result = model.analyze(x, edges)
    p = model._get_operator(edges, x.size(0), x.dtype).to_dense()
    p_nbr = p - torch.diag(torch.diag(p))
    expected = p_nbr @ result["H0_text"] - result["H0_text"]
    assert torch.allclose(result["D_text"], expected, atol=1e-6, rtol=1e-6)


def test_t6_attribute_variant_uses_only_learned_null_context():
    model, x, edges = _make("attribute_residual")
    prep = model._prepare_inputs(x, edges)
    target, source = prep["target"], prep["source"]
    h = prep["h0_text"]
    arbitrary_context = torch.randn_like(h) * 100
    real_name_route, _ = model._routes_for_modality(
        "text", h, None, target, source, context_mode="real"
    )
    ignored_d_route, _ = model._routes_for_modality(
        "text", h, arbitrary_context, target, source, context_mode="real"
    )
    assert torch.equal(real_name_route, ignored_d_route)
    assert prep["structural"]["text"] is None


def test_t7_real_structural_context_changes_relation_routes():
    model, x, edges = _make("context_residual")
    with torch.no_grad():
        real = model.route_analysis(x, edges, context_mode="real")
        null = model.route_analysis(x, edges, context_mode="null")
    assert not torch.allclose(real["routes_text"], null["routes_text"], atol=1e-9, rtol=0)


def test_t8_global_residual_uses_one_mean_route_on_every_edge():
    model, x, edges = _make("global_residual")
    with torch.no_grad():
        analysis = model.analyze(x, edges)
    for modality in ("text", "visual"):
        route = analysis[f"routes_{modality}"]
        assert torch.equal(route, route[:1].expand_as(route))
        assert torch.allclose(route[0], analysis[f"raw_routes_{modality}"].mean(dim=0))


def test_t9_r1_forced_global_route_matches_r0g_with_common_weights():
    r0g, x, edges = _make("global_residual", seed=19)
    r1, _, _ = _make("attribute_residual", seed=23)
    r1.load_state_dict(r0g.state_dict(), strict=True)
    with torch.no_grad():
        global_result = r0g.analyze(x, edges)
        forced_result = r1.analyze(x, edges, route_mode="global")
    assert torch.equal(global_result["fused_z"], forced_result["fused_z"])


def test_t10_no_adaptation_route_has_exactly_zero_residual():
    adapters = torch.randn(5, 3, 256)
    route = torch.zeros(5, 4)
    route[:, 0] = 1.0
    residual = Model._mix_adapters(adapters, route)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_t11_router_initialization_starts_near_half_no_adaptation():
    model, x, edges = _make("attribute_residual")
    with torch.no_grad():
        route = model.route_analysis(x, edges)["routes_text"]
    assert torch.allclose(route[:, 0].mean(), torch.tensor(0.5), atol=0.01)
    assert torch.allclose(route[:, 1:].mean(), torch.tensor(1.0 / 6), atol=0.01)


def test_t12_edge_chunk_sizes_preserve_routes_and_outputs():
    m1, x, edges = _make("context_residual", seed=31, chunk=2)
    m2, _, _ = _make("context_residual", seed=32, chunk=7)
    m2.load_state_dict(m1.state_dict(), strict=True)
    with torch.no_grad():
        a, b = m1.analyze(x, edges), m2.analyze(x, edges)
    assert torch.allclose(a["routes_text"], b["routes_text"], atol=1e-7, rtol=1e-6)
    assert torch.allclose(a["fused_z"], b["fused_z"], atol=1e-6, rtol=1e-6)


def test_t13_context_variant_all_required_paths_receive_finite_gradients():
    model, x, edges = _make("context_residual")
    model.train()
    z, _, _, _, _ = model(x, edges)
    z.square().mean().backward()
    prefixes = (
        "shared_text", "relation_text.self_projection", "relation_text.context_projection",
        "relation_text.attention", "relation_text.ffn", "relation_text.router",
        "adapters_text.0", "text_projector",
    )
    named = dict(model.named_parameters())
    for prefix in prefixes:
        grads = [p.grad for name, p in named.items() if name.startswith(prefix)]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), prefix


def test_t14_context_null_changes_only_evidence_and_does_not_mutate_parameters():
    model, x, edges = _make("context_residual")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        real = model.route_analysis(x, edges, context_mode="real")
        null = model.route_analysis(x, edges, context_mode="null")
    assert torch.equal(real["H0_text"], null["H0_text"])
    assert torch.equal(real["D_text"], null["D_text"])
    assert not torch.equal(real["routes_text"], null["routes_text"])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_t15_context_shuffle_preserves_node_marginal_and_is_deterministic():
    context = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    a = Model._shuffle_context(context, 501)
    b = Model._shuffle_context(context, 501)
    assert torch.equal(a, b)
    assert torch.equal(a.sort(dim=0).values, context.sort(dim=0).values)
    assert not torch.equal(a, context)


def test_t16_zero_shared_and_zero_residual_interventions_are_exact():
    model, x, edges = _make("attribute_residual", seed=67)
    zero_shared = Model(_cfg("attribute_residual"), {"input_dim": 6, "text_dim": 3, "visual_dim": 3}).eval()
    zero_shared.load_state_dict(model.state_dict(), strict=True)
    zero_residual = Model(_cfg("attribute_residual"), {"input_dim": 6, "text_dim": 3, "visual_dim": 3}).eval()
    zero_residual.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        for modality in ("text", "visual"):
            getattr(zero_shared, f"shared_{modality}").up.weight.zero_()
            getattr(zero_shared, f"shared_{modality}").up.bias.zero_()
            for adapter in getattr(zero_residual, f"adapters_{modality}"):
                adapter.up.weight.zero_()
                adapter.up.bias.zero_()
        expected_shared = zero_shared.analyze(x, edges)
        actual_shared = model.analyze(x, edges, shared_enabled=False)
        expected_residual = zero_residual.analyze(x, edges)
        actual_residual = model.analyze(x, edges, residual_enabled=False)
    assert torch.equal(expected_shared["fused_z"], actual_shared["fused_z"])
    assert torch.equal(expected_residual["fused_z"], actual_residual["fused_z"])


def test_t17_shrinkage_endpoints_and_formula():
    model, x, edges = _make("context_residual")
    with torch.no_grad():
        normal = model.route_analysis(x, edges)
        one = model.route_analysis(x, edges, shrinkage=1.0)
        zero = model.route_analysis(x, edges, shrinkage=0.0)
        half = model.route_analysis(x, edges, shrinkage=0.5)
    for modality in ("text", "visual"):
        route = normal[f"routes_{modality}"]
        assert torch.equal(route, one[f"routes_{modality}"])
        mean = route.mean(dim=0, keepdim=True)
        assert torch.allclose(zero[f"routes_{modality}"], mean.expand_as(route))
        assert torch.allclose(half[f"routes_{modality}"], mean + 0.5 * (route - mean))


def test_t18_launcher_and_eval_label_protocol_is_test_isolated():
    context = formal_contexts("cuda:1")[0]
    command = build_command(context)
    assert "task.evaluate_test=false" in command
    assert "model=relation_residual_audit" in command
    class Data:
        y = torch.tensor([0, 1, 0, 1, 2, 2])
        train_idx = torch.tensor([0, 1, 2])
        val_idx = torch.tensor([3, 4])
        num_classes = 3
        @property
        def test_idx(self):
            raise AssertionError("test split was accessed")
    from src.tasks.nc import _resolve_nc_eval_labels
    assert _resolve_nc_eval_labels(Data(), include_test=False) == [0, 1, 2]


def test_t19_formal_matrix_is_exactly_60_unique_contexts():
    contexts = formal_contexts("cuda:1")
    assert len(contexts) == 60
    assert len({context.run_key for context in contexts}) == 60
    assert len({context.checkpoint for context in contexts}) == 60


def test_t20_smoke_variants_are_isolated_and_unique():
    contexts = smoke_contexts("Movies", VARIANTS, 42, 1, "cuda:1")
    assert len(contexts) == 4
    assert {c.variant for c in contexts} == set(VARIANTS)
    assert all("/smoke/" in str(c.run_dir) for c in contexts)


def test_t21_resume_accepts_exact_config_and_rejects_changed_architecture(tmp_path, monkeypatch):
    import json
    from scripts import run_relation_residual_audit as launcher

    output = tmp_path / "outputs"
    monkeypatch.setattr(launcher, "OUTPUT", output)
    context = Context("formal", "context_residual", "Movies", 42, 300, "cuda:1", 65536)
    context.run_dir.mkdir(parents=True)
    context.checkpoint.parent.mkdir(parents=True)
    config = {
        "dataset": {"name": "Movies"}, "seed": 42, "num_runs": 1,
        "model": {
            "name": "relation_residual_audit", "residual_variant": "context_residual",
            "hidden_dim": 256, "max_order": 3, "dropout": 0.2, "relation_dim": 64,
            "relation_heads": 4, "relation_ffn_dim": 128, "shared_bottleneck_dim": 256,
            "adapter_bottleneck_dim": 32, "num_residual_adapters": 3,
            "edge_chunk_size": 65536,
        },
        "task": {
            "name": "nc", "protocol_version": "unified_full_graph_nc_v1",
            "training_mode": "full_graph", "optimizer": "adamw", "epochs": 300,
            "patience": 30, "early_stop_min_epoch": 30, "early_stop_min_delta": 1e-4,
            "lr": 1e-3, "weight_decay": 1e-4, "grad_clip": 1.0, "eval_every": 1,
            "inference_mode": "full", "evaluate_test": False, "scheduler": None,
        },
    }
    (context.run_dir / "resolved_config.json").write_text(json.dumps(config))
    (context.run_dir / "metrics.json").write_text(json.dumps({
        "best_epoch": 12, "checkpoint_selection": "best_val_accuracy",
        "metrics": {"val_acc": {}, "val_macro_f1": {}},
    }))
    (context.run_dir / "complete.marker").write_text("complete\n")
    torch.save({
        "task": "nc", "seed": 42, "selection": "best_val_accuracy", "epoch": 12,
        "metrics": {"val_acc": 0.5, "val_macro_f1": 0.4},
        "model_state": {"weight": torch.ones(1)},
        "head_state": {"weight": torch.ones(1)}, "data_info": {"num_classes": 2},
    }, context.checkpoint)
    assert is_complete(context)
    config["model"]["dropout"] = 0.3
    (context.run_dir / "resolved_config.json").write_text(json.dumps(config))
    assert not is_complete(context)


def test_t22_frozen_architecture_dimensions_reject_overrides():
    cfg = _cfg("shared_prior")
    cfg.model.relation_heads = 8
    with pytest.raises(ValueError, match="P0-R fixes"):
        Model(cfg, {"input_dim": 6, "text_dim": 3, "visual_dim": 3})


def test_t23_training_mechanism_log_parser_keeps_epoch_boundaries(tmp_path):
    from scripts.analyze_relation_residual_audit import _training_mechanism_stats

    (tmp_path / "main.log").write_text(
        "[INFO] Epoch 00001 | Train Loss 1.0\n"
        "[INFO] Aux rr_text_pi0_mean 0.5026 | rr_visual_pi0_mean 0.5031\n"
        "[INFO] Epoch 00002 | Train Loss 0.9\n"
        "[INFO] Aux rr_text_pi0_mean 0.6100 | rr_visual_pi0_mean 0.6200\n"
    )
    records = _training_mechanism_stats(tmp_path)
    assert records == [
        (1, {"rr_text_pi0_mean": 0.5026, "rr_visual_pi0_mean": 0.5031}),
        (2, {"rr_text_pi0_mean": 0.61, "rr_visual_pi0_mean": 0.62}),
    ]
