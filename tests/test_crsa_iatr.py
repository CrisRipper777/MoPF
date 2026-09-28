from __future__ import annotations

import inspect

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from src.models.crsa_iatr import Model
from src.models.relation_residual_audit import Model as FrozenP0R


FLAGS = {
    "base": (False, False),
    "crsa": (True, False),
    "iatr": (False, True),
    "full": (True, True),
}


def _cfg(variant: str, *, node_chunk: int = 3, edge_chunk: int = 4, seed: int = 17):
    use_crsa, use_iatr = FLAGS[variant]
    return OmegaConf.create({
        "seed": seed,
        "model": {
            "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
            "variant": variant, "use_crsa": use_crsa, "use_iatr": use_iatr,
            "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
            "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
            "num_residual_adapters": 3, "edge_chunk_size": edge_chunk,
            "trajectory_heads": 4, "trajectory_layers": 1,
            "trajectory_ffn_dim": 512, "anchor_heads": 4,
            "anchor_ffn_dim": 512, "iatr_node_chunk_size": node_chunk,
        },
    })


def _toy():
    torch.manual_seed(5)
    info = {"input_dim": 10, "text_dim": 6, "visual_dim": 4, "num_nodes": 9, "num_classes": 3}
    x = torch.randn(9, 10)
    # Deliberately asymmetric input with loops and duplicate coordinates.
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 0, 0, 3],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 8, 7, 0, 1, 3],
    ])
    return info, x, edge_index


def _make(variant: str, seed: int = 17, node_chunk: int = 3, edge_chunk: int = 4):
    info, x, edges = _toy()
    cfg = _cfg(variant, node_chunk=node_chunk, edge_chunk=edge_chunk, seed=seed)
    torch.manual_seed(seed)
    model = Model(cfg, info).eval()
    return model, x, edges


def _count(hook_calls):
    return len(hook_calls)


def test_all_four_variants_forward_backward_and_nc_head():
    info, x, edges = _toy()
    labels = torch.arange(x.size(0)) % info["num_classes"]
    for variant in FLAGS:
        model, _, _ = _make(variant)
        model.train()
        z, _, _, aux_loss, aux_info = model(x, edges)
        head = nn.Linear(256, info["num_classes"])
        loss = F.cross_entropy(head(z), labels) + aux_loss
        loss.backward()
        assert z.shape == (x.size(0), 256)
        assert torch.isfinite(z).all() and torch.isfinite(loss)
        assert aux_loss.item() == 0.0 and aux_info == {}
        assert len(model.analyze(x, edges)["C_text"]) == 4
        assert len(model.analyze(x, edges)["C_visual"]) == 4
        assert head(z).shape == (x.size(0), info["num_classes"])
        active = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert active and all(torch.isfinite(grad).all() for grad in active)


def test_ablation_semantics_do_not_collapse_to_one_computation_path():
    info, x, edges = _toy()
    observations = {}
    for variant in FLAGS:
        model, _, _ = _make(variant)
        relation_calls, shared_calls, adapter_calls, trajectory_calls, anchor_calls = [], [], [], [], []
        handles = []
        if model.use_crsa:
            handles.append(model.relation_text.register_forward_hook(
                lambda *args: relation_calls.append(1)
            ))
            handles.append(model.shared_text.register_forward_hook(
                lambda *args: shared_calls.append(1)
            ))
            for adapter in model.adapters_text:
                handles.append(adapter.register_forward_hook(lambda *args: adapter_calls.append(1)))
        if model.use_iatr:
            handles.extend([
                model.iatr_text.trajectory_encoder.register_forward_hook(
                    lambda *args: trajectory_calls.append(1)
                ),
                model.iatr_text.anchor_attention.register_forward_hook(
                    lambda *args: anchor_calls.append(1)
                ),
            ])
        with torch.no_grad():
            result = model.analyze(x, edges)
        for handle in handles:
            handle.remove()
        observations[variant] = (
            _count(relation_calls), _count(shared_calls), _count(adapter_calls),
            _count(trajectory_calls), _count(anchor_calls),
        )
        if not model.use_iatr:
            expected = (torch.stack(result["C_text"]).mean(0) + torch.stack(result["C_visual"]).mean(0))
            expected = model.plain_fusion(torch.cat((
                torch.stack(result["C_text"]).mean(0),
                torch.stack(result["C_visual"]).mean(0),
            ), dim=-1))
            assert torch.allclose(result["fused_z"], expected)
        else:
            assert result["trajectory_tokens_text"].shape == (x.size(0), 3, 256)
            assert result["trajectory_tokens_visual"].shape == (x.size(0), 3, 256)
    assert observations["base"] == (0, 0, 0, 0, 0)
    assert observations["crsa"][0] > 0 and observations["crsa"][1] > 0
    assert observations["crsa"][2] > 0 and observations["crsa"][3:] == (0, 0)
    assert observations["iatr"] == (0, 0, 0, 3, 3)
    assert observations["full"][0] > 0 and observations["full"][1] > 0
    assert observations["full"][2] > 0 and observations["full"][3:] == (3, 3)


def test_crsa_physical_operator_matches_frozen_p0r_exactly():
    info, x, edges = _toy()
    p0r = FrozenP0R(OmegaConf.create({"model": {
        "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
        "residual_variant": "context_residual", "relation_dim": 64,
        "relation_heads": 4, "relation_ffn_dim": 128,
        "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
        "num_residual_adapters": 3, "edge_chunk_size": 4,
    }}), info).eval()
    new, _, _ = _make("crsa")
    old_p = p0r._get_operator(edges, x.size(0), x.dtype)
    new_p = new._get_operator(edges, x.size(0), x.dtype)
    assert torch.equal(old_p.indices(), new_p.indices())
    assert torch.equal(old_p.values(), new_p.values())


def test_iatr_increment_tokens_and_intrinsic_anchor_shapes():
    model, x, edges = _make("full")
    with torch.no_grad():
        result = model.analyze(x, edges)
    assert len(result["C_text"]) == len(result["C_visual"]) == 4
    for modality in ("text", "visual"):
        states = result[f"C_{modality}"]
        tokens = result[f"trajectory_tokens_{modality}"]
        module = getattr(model, f"iatr_{modality}")
        expected_state = torch.stack(states[1:], dim=1)
        expected_delta = torch.stack([
            states[1] - states[0], states[2] - states[1], states[3] - states[2],
        ], dim=1)
        expected = module.token_norm(
            module.state_projection(expected_state)
            + module.increment_projection(expected_delta)
            + module.order_embedding.unsqueeze(0)
        )
        assert tokens.shape == (x.size(0), 3, 256)
        assert torch.allclose(tokens, expected, atol=1e-6, rtol=1e-6)
        assert result[f"trajectory_encoded_{modality}"].shape == (x.size(0), 3, 256)
        assert result[f"anchor_{modality}"].shape == (x.size(0), 256)
    assert result["D_text"] is not None and result["D_visual"] is not None


def test_text_visual_stages_are_independent_and_factor_initialization_is_matched():
    crsa, _, _ = _make("crsa", seed=29)
    full, _, _ = _make("full", seed=29)
    iatr, _, _ = _make("iatr", seed=29)
    assert crsa.shared_text.down.weight.data_ptr() != crsa.shared_visual.down.weight.data_ptr()
    assert crsa.relation_text.router.weight.data_ptr() != crsa.relation_visual.router.weight.data_ptr()
    assert full.iatr_text.state_projection.weight.data_ptr() != full.iatr_visual.state_projection.weight.data_ptr()
    assert all(torch.equal(crsa.state_dict()[key], full.state_dict()[key])
               for key in crsa.state_dict() if key in full.state_dict())
    assert all(torch.equal(iatr.state_dict()[key], full.state_dict()[key])
               for key in iatr.state_dict() if key in full.state_dict())


def test_exact_node_chunking_preserves_iatr_outputs():
    a, x, edges = _make("full", seed=61, node_chunk=2, edge_chunk=3)
    b, _, _ = _make("full", seed=62, node_chunk=7, edge_chunk=8)
    b.load_state_dict(a.state_dict(), strict=True)
    with torch.no_grad():
        z_a, z_b = a(x, edges)[0], b(x, edges)[0]
    assert torch.allclose(z_a, z_b, atol=2e-6, rtol=2e-6)


def test_forward_api_has_no_label_or_split_inputs():
    signature = inspect.signature(Model.forward)
    assert list(signature.parameters) == ["self", "x", "edge_index"]
    class GuardedInfo(dict):
        def get(self, key, default=None):
            if any(token in key.lower() for token in ("label", "mask", "test", "split")):
                raise AssertionError(f"model read forbidden metadata: {key}")
            return super().get(key, default)
    info, x, edges = _toy()
    guarded = GuardedInfo(info)
    model = Model(_cfg("full"), guarded).eval()
    with torch.no_grad():
        z = model(x, edges)[0]
    assert torch.isfinite(z).all()


def test_variant_name_must_match_flags():
    info, _, _ = _toy()
    cfg = _cfg("base")
    cfg.model.use_iatr = True
    with pytest.raises(ValueError, match="conflicts"):
        Model(cfg, info)
