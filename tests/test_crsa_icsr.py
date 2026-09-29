from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.crsa_rcfa import Model as FrozenCRSA
from src.models.crsa_icsr import Model as ICSRModel, _ICSRModality


def _cfg(seed=17, *, use_icsr=True, edge_chunk=4, node_chunk=3):
    return OmegaConf.create({
        "seed": seed,
        "model": {
            "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
            "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
            "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
            "num_residual_adapters": 3, "edge_chunk_size": edge_chunk,
            "use_icsr": use_icsr, "icsr_bottleneck_dim": 32,
            "icsr_node_chunk_size": node_chunk,
        },
    })


def _old_cfg(seed=17, edge_chunk=4):
    return OmegaConf.create({
        "seed": seed,
        "model": {
            "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
            "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
            "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
            "num_residual_adapters": 3, "edge_chunk_size": edge_chunk,
            "use_rcfa": False, "use_relation_condition": True,
            "rcfa_hidden_dim": 256, "rcfa_node_chunk_size": 3,
        },
    })


def _toy():
    torch.manual_seed(5)
    info = {"input_dim": 10, "text_dim": 6, "visual_dim": 4,
            "num_nodes": 9, "num_classes": 3}
    x = torch.randn(9, 10)
    edges = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 0, 0, 3],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 8, 7, 0, 1, 3],
    ])
    return info, x, edges


def _new(seed=17, *, use_icsr=True, edge_chunk=4, node_chunk=3):
    info, x, edges = _toy()
    torch.manual_seed(seed)
    return ICSRModel(_cfg(seed, use_icsr=use_icsr, edge_chunk=edge_chunk,
                          node_chunk=node_chunk), info), info, x, edges


def test_legacy_crsa_parameter_initialization_and_propagation_parity():
    info, x, edges = _toy()
    seed = 31
    torch.manual_seed(seed)
    old = FrozenCRSA(_old_cfg(seed), info).eval()
    old_head = nn.Linear(256, info["num_classes"])
    torch.manual_seed(seed)
    new = ICSRModel(_cfg(seed, use_icsr=False), info).eval()
    new_head = nn.Linear(256, info["num_classes"])
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    assert all(torch.equal(a, b) for a, b in zip(old_head.parameters(), new_head.parameters()))

    with torch.no_grad():
        old_a, new_a = old.analyze(x, edges), new.analyze(x, edges)
    for key in ("H0_text", "H0_visual", "base_text", "base_visual", "fused_z"):
        assert torch.equal(old_a[key], new_a[key]), key
    for modality in ("text", "visual"):
        for group in ("C", "rse", "routes"):
            left, right = old_a[f"{group}_{modality}"], new_a[f"{group}_{modality}"]
            if isinstance(left, list):
                assert all(torch.equal(a, b) for a, b in zip(left, right)), f"{group}_{modality}"
            else:
                assert torch.equal(left, right), f"{group}_{modality}"
    assert (old(x, edges)[0] - new(x, edges)[0]).abs().max().item() < 1e-6


def test_zero_initialized_interaction_is_exact_identity_on_carrier():
    model, _, x, edges = _new()
    model.eval()
    with torch.no_grad():
        result = model.analyze(x, edges)
    for modality in ("text", "visual"):
        assert len(result[f"interaction_{modality}"]) == 3
        for residual in result[f"interaction_{modality}"]:
            assert torch.equal(residual, torch.zeros_like(residual))
        assert torch.equal(result[f"Z_{modality}"], result[f"base_{modality}"])


def test_descriptors_match_parameter_free_layer_norm_formulas():
    model, _, x, edges = _new()
    model.eval()
    with torch.no_grad():
        result = model.analyze(x, edges)
    for modality in ("text", "visual"):
        h0 = result[f"H0_{modality}"]
        for hop in range(1, 4):
            context = result[f"C_{modality}"][hop]
            expected_a = F.layer_norm(h0, (256,)) * F.layer_norm(context, (256,))
            expected_d = F.layer_norm(context, (256,)) - F.layer_norm(h0, (256,))
            assert (expected_a - result[f"agreement_{modality}"][hop - 1]).abs().max() < 1e-6
            assert (expected_d - result[f"deviation_{modality}"][hop - 1]).abs().max() < 1e-6


def test_adapter_shared_across_hops_and_parameter_delta_is_fixed():
    model, info, x, edges = _new()
    reference, _, _, _ = _new(use_icsr=False)
    assert sum(p.numel() for p in model.parameters()) - sum(
        p.numel() for p in reference.parameters()
    ) == 49_728
    for modality in ("text", "visual"):
        module = getattr(model, f"icsr_{modality}")
        assert isinstance(module, _ICSRModality)
        calls = []
        handle = module.register_forward_hook(lambda _m, _i, _o: calls.append(1))
        with torch.no_grad():
            model.analyze(x, edges)
        handle.remove()
        expected_chunk_calls = (x.size(0) + model.icsr_node_chunk_size - 1) // model.icsr_node_chunk_size
        assert len(calls) == 3 * expected_chunk_calls
    assert model.icsr_text is not model.icsr_visual
    assert not any("icsr_hop" in key for key in model.state_dict())


def test_modality_isolation_in_states_descriptors_and_interactions():
    model, info, x, edges = _new()
    model.eval()
    changed_visual, changed_text = x.clone(), x.clone()
    torch.manual_seed(71)
    changed_visual[:, info["text_dim"]:] = torch.randn_like(changed_visual[:, info["text_dim"]:]) * 4
    changed_text[:, :info["text_dim"]] = torch.randn_like(changed_text[:, :info["text_dim"]]) * 4
    with torch.no_grad():
        original = model.analyze(x, edges)
        visual_changed = model.analyze(changed_visual, edges)
        text_changed = model.analyze(changed_text, edges)
    for key in ("H0_text", "Z_text", "base_text"):
        assert torch.equal(original[key], visual_changed[key]), key
    for suffix in ("C_text", "agreement_text", "deviation_text", "interaction_text"):
        assert all(torch.equal(a, b) for a, b in zip(original[suffix], visual_changed[suffix])), suffix
    for suffix in ("C_visual", "agreement_visual", "deviation_visual", "interaction_visual"):
        assert all(torch.equal(a, b) for a, b in zip(original[suffix], text_changed[suffix])), suffix
    assert torch.equal(original["Z_visual"], text_changed["Z_visual"])


def test_full_v3_backward_has_finite_nonzero_up_gradients():
    model, info, x, edges = _new()
    head = nn.Linear(256, info["num_classes"])
    labels = torch.arange(x.size(0)) % info["num_classes"]
    z, _, _, aux_loss, aux_info = model(x, edges)
    loss = F.cross_entropy(head(z), labels) + aux_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert aux_loss.item() == 0.0 and aux_info == {}
    grads = [p.grad for p in list(model.parameters()) + list(head.parameters()) if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    for modality in ("text", "visual"):
        grad = getattr(model, f"icsr_{modality}").up.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0


def test_stage_two_has_only_the_fixed_bottleneck_and_strict_reload_is_exact():
    model, info, x, edges = _new()
    for modality in ("text", "visual"):
        module = getattr(model, f"icsr_{modality}")
        assert [type(layer) for layer in module.children()] == [nn.Linear, nn.Linear]
        assert not any(isinstance(layer, (nn.LayerNorm, nn.MultiheadAttention))
                       for layer in module.modules())
        assert not hasattr(module, "gate") and not hasattr(module, "router")
    source = inspect.getsource(_ICSRModality).lower()
    assert "sigmoid" not in source and "softmax" not in source and "attention" not in source

    model.eval()
    with torch.no_grad():
        expected = model(x, edges)[0]
    restored = ICSRModel(_cfg(17), info)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(x, edges)[0]
    assert torch.equal(expected, actual)


def test_equal_mean_interaction_aggregation_and_rse_diagnostic():
    model, _, x, edges = _new(seed=63)
    model.eval()
    with torch.no_grad():
        for modality in ("text", "visual"):
            adapter = getattr(model, f"icsr_{modality}")
            adapter.up.weight.normal_(mean=0.0, std=1e-4)
            adapter.up.bias.normal_(mean=0.0, std=1e-4)
        result = model.analyze(x, edges)
        operator = model._get_operator(edges, x.size(0), x.dtype)
        for modality in ("text", "visual"):
            states = result[f"C_{modality}"]
            rse = result[f"rse_{modality}"]
            expected_u = [getattr(model, f"icsr_{modality}")(result[f"H0_{modality}"], states[k])
                          for k in range(1, 4)]
            expected_z = result[f"base_{modality}"] + torch.stack(expected_u).mean(dim=0)
            assert torch.allclose(result[f"Z_{modality}"], expected_z, atol=1e-6, rtol=0.0)
            for k in range(1, 4):
                physical = torch.sparse.mm(operator, states[k - 1])
                assert (rse[k - 1] - (states[k] - physical)).abs().max() < 1e-5
