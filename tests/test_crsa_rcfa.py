from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.crsa_iatr import Model as LegacyModel
from src.models.crsa_rcfa import Model


def _cfg(*, seed=17, use_rcfa=True, use_relation_condition=True, edge_chunk=4, node_chunk=3):
    return OmegaConf.create({
        "seed": seed,
        "model": {
            "hidden_dim": 256,
            "max_order": 3,
            "dropout": 0.2,
            "relation_dim": 64,
            "relation_heads": 4,
            "relation_ffn_dim": 128,
            "shared_bottleneck_dim": 256,
            "adapter_bottleneck_dim": 32,
            "num_residual_adapters": 3,
            "edge_chunk_size": edge_chunk,
            "use_rcfa": use_rcfa,
            "use_relation_condition": use_relation_condition,
            "rcfa_hidden_dim": 256,
            "rcfa_node_chunk_size": node_chunk,
        },
    })


def _toy():
    torch.manual_seed(5)
    info = {"input_dim": 10, "text_dim": 6, "visual_dim": 4, "num_nodes": 9, "num_classes": 3}
    x = torch.randn(9, 10)
    edges = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 0, 0, 3],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 8, 7, 0, 1, 3],
    ])
    return info, x, edges


def _make(*, seed=17, use_rcfa=True, use_relation_condition=True, edge_chunk=4, node_chunk=3):
    info, x, edges = _toy()
    torch.manual_seed(seed)
    model = Model(
        _cfg(seed=seed, use_rcfa=use_rcfa, use_relation_condition=use_relation_condition,
             edge_chunk=edge_chunk, node_chunk=node_chunk),
        info,
    )
    return model, info, x, edges


def _legacy_cfg(seed):
    return OmegaConf.create({
        "seed": seed,
        "model": {
            "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
            "variant": "crsa", "use_crsa": True, "use_iatr": False,
            "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
            "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
            "num_residual_adapters": 3, "edge_chunk_size": 4,
            "trajectory_heads": 4, "trajectory_layers": 1,
            "trajectory_ffn_dim": 512, "anchor_heads": 4,
            "anchor_ffn_dim": 512, "iatr_node_chunk_size": 3,
        },
    })


def test_a_legacy_crsa_initialization_and_output_parity():
    info, x, edges = _toy()
    seed = 31
    torch.manual_seed(seed)
    old = LegacyModel(_legacy_cfg(seed), info).eval()
    old_head = nn.Linear(256, info["num_classes"])
    torch.manual_seed(seed)
    new = Model(_cfg(seed=seed, use_rcfa=False), info).eval()
    new_head = nn.Linear(256, info["num_classes"])

    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    assert all(torch.equal(a, b) for a, b in zip(old_head.parameters(), new_head.parameters()))
    with torch.no_grad():
        old_z = old(x, edges)[0]
        new_z = new(x, edges)[0]
    max_error = (old_z - new_z).abs().max().item()
    assert max_error < 1e-6


def test_b_rse_decomposition_matches_plain_physical_propagation():
    model, _, x, edges = _make()
    result = model.analyze(x, edges)
    operator = model._get_operator(edges, x.size(0), x.dtype)
    errors = []
    for modality in ("text", "visual"):
        states = result[f"C_{modality}"]
        rse_values = result[f"rse_{modality}"]
        assert len(states) == 4 and len(rse_values) == 3
        for hop in range(1, 4):
            plain = torch.sparse.mm(operator, states[hop - 1])
            error = (rse_values[hop - 1] - (states[hop] - plain)).abs().max().item()
            errors.append(error)
            assert error < 1e-5, (modality, hop, error)
            assert rse_values[hop - 1].requires_grad
    assert max(errors) < 1e-5


def test_c_zero_init_gate_and_readout_identity_for_both_modalities():
    model, _, x, edges = _make()
    model.eval()
    result = model.analyze(x, edges)
    identity_errors = []
    for modality in ("text", "visual"):
        states = result[f"C_{modality}"]
        base = torch.stack(states, dim=0).mean(dim=0)
        gates = result[f"gate_{modality}"]
        assert len(gates) == 3
        assert all(torch.equal(gate, torch.ones_like(gate)) for gate in gates)
        error = (result[f"Z_{modality}"] - base).abs().max().item()
        identity_errors.append(error)
        assert error < 1e-6
    assert max(identity_errors) < 1e-6


def test_d_gate_and_output_shapes_are_feature_wise_and_finite():
    model, _, x, edges = _make()
    with torch.no_grad():
        result = model.analyze(x, edges)
        z = model(x, edges)[0]
    assert z.shape == (x.size(0), 256)
    for modality in ("text", "visual"):
        assert result[f"Z_{modality}"].shape == (x.size(0), 256)
        for gate in result[f"gate_{modality}"]:
            assert gate.shape == (x.size(0), 256)
            assert torch.isfinite(gate).all()
        assert torch.isfinite(result[f"Z_{modality}"]).all()
    assert torch.isfinite(z).all()


def test_e_full_v2_nc_loss_backpropagates_finite_gradients_to_rcfa():
    model, info, x, edges = _make()
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
        final = getattr(model, f"rcfa_{modality}").conditioner[-1]
        assert final.weight.grad is not None
        assert torch.isfinite(final.weight.grad).all()
        assert final.weight.grad.abs().sum().item() > 0


def test_f_rcfa_inputs_are_modality_isolated():
    model, info, x, edges = _make()
    model.eval()
    changed_visual = x.clone()
    changed_visual[:, info["text_dim"]:] = torch.randn_like(changed_visual[:, info["text_dim"]:]) * 4
    changed_text = x.clone()
    changed_text[:, :info["text_dim"]] = torch.randn_like(changed_text[:, :info["text_dim"]]) * 4
    with torch.no_grad():
        original = model.analyze(x, edges)
        visual_changed = model.analyze(changed_visual, edges)
        text_changed = model.analyze(changed_text, edges)
    for key in ("H0_text", "Z_text"):
        assert torch.equal(original[key], visual_changed[key]), key
    for key in ("rse_text", "gate_text"):
        assert all(torch.equal(a, b) for a, b in zip(original[key], visual_changed[key])), key
    for key in ("rse_visual", "gate_visual"):
        assert all(torch.equal(a, b) for a, b in zip(original[key], text_changed[key])), key
    assert torch.equal(original["Z_visual"], text_changed["Z_visual"])


def test_g_zero_relation_condition_path_keeps_architecture_and_backpropagates():
    full, info, x, edges = _make(seed=51, use_relation_condition=True)
    ablated, _, _, _ = _make(seed=52, use_relation_condition=False)
    ablated.load_state_dict(full.state_dict(), strict=True)
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in ablated.parameters())
    # Make relation conditioning observable while preserving identical Stage I weights.
    with torch.no_grad():
        for modality in ("text", "visual"):
            layer = getattr(full, f"rcfa_{modality}").conditioner[-1]
            layer.weight.normal_(mean=0.0, std=1e-4)
            layer.bias.normal_(mean=0.0, std=1e-4)
        ablated.load_state_dict(full.state_dict(), strict=True)
    full.eval()
    ablated.eval()
    full_result = full.analyze(x, edges)
    ablated_result = ablated.analyze(x, edges)
    for modality in ("text", "visual"):
        for a, b in zip(full_result[f"C_{modality}"], ablated_result[f"C_{modality}"]):
            assert torch.equal(a, b)
        for a, b in zip(full_result[f"rse_{modality}"], ablated_result[f"rse_{modality}"]):
            assert torch.equal(a, b)
    z, _, _, _, _ = ablated(x, edges)
    loss = F.cross_entropy(nn.Linear(256, info["num_classes"])(z), torch.arange(x.size(0)) % info["num_classes"])
    loss.backward()
    assert torch.isfinite(loss)
    assert ablated.rcfa_text.conditioner[-1].weight.grad is not None
    assert torch.isfinite(ablated.rcfa_text.conditioner[-1].weight.grad).all()


def test_h_model_forward_has_no_label_or_split_inputs():
    signature = inspect.signature(Model.forward)
    assert list(signature.parameters) == ["self", "x", "edge_index"]

    class GuardedInfo(dict):
        def get(self, key, default=None):
            if any(token in key.lower() for token in ("label", "train_idx", "val_idx", "test_idx")):
                raise AssertionError(f"model read forbidden metadata: {key}")
            return super().get(key, default)

    info, x, edges = _toy()
    model = Model(_cfg(), GuardedInfo(info)).eval()
    with torch.no_grad():
        assert torch.isfinite(model(x, edges)[0]).all()


def test_i_strict_checkpoint_reload_is_deterministic(tmp_path):
    model, info, x, edges = _make(seed=61)
    model.eval()
    with torch.no_grad():
        expected = model(x, edges)[0]
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = Model(_cfg(seed=61), info).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        actual = restored(x, edges)[0]
    assert torch.equal(expected, actual)
