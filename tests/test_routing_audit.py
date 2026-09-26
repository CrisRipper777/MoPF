from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.cmrf_probe import Model as CMRFModel
from src.models.routing_audit import Model


def cfg(variant: str, hidden_dim: int = 8):
    return OmegaConf.create(
        {
            "model": {
                "hidden_dim": hidden_dim,
                "max_order": 3,
                "dropout": 0.0,
                "context_dim": 128,
                "latent_dim": 4,
                "routing_variant": variant,
            }
        }
    )


def inputs():
    torch.manual_seed(19)
    x = torch.randn(9, 11)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8],
         [1, 0, 2, 1, 3, 2, 5, 4, 7, 6, 8]],
        dtype=torch.long,
    )
    return x, edge_index


def make_model(variant: str, hidden_dim: int = 8):
    return Model(
        cfg(variant, hidden_dim),
        {"input_dim": 11, "text_dim": 5, "visual_dim": 6},
    )


def test_uniform_state_mixture_reproduces_e3_c0_forward():
    x, edge_index = inputs()
    old_cfg = OmegaConf.create(
        {"model": {
            "hidden_dim": 8,
            "max_order": 3,
            "dropout": 0.0,
            "composition_mode": "uniform",
            "controller_latent_dim": 4,
            "controller_hidden_dim": 8,
            "controller_lambda": 0.25,
        }}
    )
    old = CMRFModel(old_cfg, {"input_dim": 11, "text_dim": 5, "visual_dim": 6}).eval()
    new = make_model("uniform").eval()
    new.load_state_dict(
        {
            key: value
            for key, value in old.state_dict().items()
            if key in new.state_dict() and new.state_dict()[key].shape == value.shape
        },
        strict=False,
    )
    expected = old(x, edge_index)[0]
    actual = new(x, edge_index)[0]
    assert float((expected - actual).abs().max()) < 1e-6
    assert torch.allclose(
        new.analyze(x, edge_index)["alpha_text"],
        torch.full((x.size(0), 4), 0.25),
        atol=0,
        rtol=0,
    )


def test_state_simplex_expresses_near_self_only():
    states = [torch.randn(5, 7) for _ in range(4)]
    alpha = torch.tensor([[0.9999997, 0.0000001, 0.0000001, 0.0000001]]).expand(5, -1)
    mixed = Model.mix_states(states, alpha)
    assert torch.allclose(mixed, states[0], atol=1e-5, rtol=0)


def test_global_simplex_logits_are_shared_across_nodes():
    x, edge_index = inputs()
    model = make_model("global_simplex").eval()
    with torch.no_grad():
        model.global_text_logits.copy_(torch.tensor([1.0, 0.0, -1.0, 0.5]))
    result = model.analyze(x, edge_index)
    assert torch.equal(result["alpha_text"], result["alpha_text"][:1].expand_as(result["alpha_text"]))


def test_node_residual_router_is_zero_initialized():
    for variant in ("node_flat_simplex", "hierarchical_flat_simplex"):
        result = make_model(variant).eval().analyze(*inputs())
        assert torch.count_nonzero(result["delta_text"]) == 0
        assert torch.count_nonzero(result["delta_visual"]) == 0


def test_hierarchical_initialization_is_uniform():
    result = make_model("hierarchical_flat_simplex").eval().analyze(*inputs())
    expected = torch.full_like(result["alpha_text"], 0.25)
    assert torch.equal(result["alpha_text"], expected)
    assert torch.equal(result["alpha_visual"], expected)


def test_relation_encoder_leaves_physical_edges_and_operator_unchanged():
    x, edge_index = inputs()
    saved_edges = edge_index.clone()
    baseline = make_model("trajectory")
    relation = make_model("relation")
    p0 = baseline._build_propagation_operator(edge_index, x.size(0), x.dtype).to_dense()
    p1 = relation._build_propagation_operator(edge_index, x.size(0), x.dtype).to_dense()
    relation.eval().analyze(x, edge_index)
    assert torch.equal(edge_index, saved_edges)
    assert torch.equal(p0, p1)


def test_relation_shuffle_changes_only_router_path_inputs():
    x, edge_index = inputs()
    model = make_model("relation").eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.router_text[-1].weight, std=0.1)
        torch.nn.init.normal_(model.router_visual[-1].weight, std=0.1)
    analysis = model.analyze(x, edge_index)
    states_before = [value.clone() for value in analysis["S_text"]]
    relation_before = analysis["relation_text"].clone()
    permutation = torch.tensor([3, 0, 8, 2, 1, 7, 4, 6, 5])
    routed = model.reroute(
        analysis,
        relation_text_override=analysis["relation_text"][permutation],
        relation_visual_override=analysis["relation_visual"][permutation],
    )
    assert all(torch.equal(a, b) for a, b in zip(states_before, analysis["S_text"]))
    assert torch.equal(relation_before, analysis["relation_text"])
    assert not torch.allclose(analysis["alpha_text"], routed["alpha_text"])
    assert torch.equal(analysis["S_visual"][0], model.analyze(x, edge_index)["S_visual"][0])


def test_optional_cross_modal_relation_context_only_changes_router_logits():
    x, edge_index = inputs()
    model = make_model("cross_relation").eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.router_text[-1].weight, std=0.1)
        torch.nn.init.normal_(model.router_visual[-1].weight, std=0.1)
    analysis = model.analyze(x, edge_index)
    states_before = [value.clone() for value in analysis["S_text"]]
    relation_text_before = analysis["relation_text"].clone()
    relation_visual_before = analysis["relation_visual"].clone()
    before_edges = edge_index.clone()
    permutation = torch.tensor([3, 0, 8, 2, 1, 7, 4, 6, 5])
    changed = model.reroute(
        analysis, joint_relation_override=analysis["joint_relation"][permutation]
    )
    assert torch.equal(edge_index, before_edges)
    assert torch.equal(relation_text_before, analysis["relation_text"])
    assert torch.equal(relation_visual_before, analysis["relation_visual"])
    assert all(torch.equal(a, b) for a, b in zip(states_before, analysis["S_text"]))
    assert not torch.allclose(analysis["alpha_text"], changed["alpha_text"])


def test_runner_disables_test_evaluation_and_analysis_excludes_test_indices():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts" / "run_routing_audit.py").read_text(encoding="utf-8")
    analysis = (root / "scripts" / "analyze_routing_audit.py").read_text(encoding="utf-8")
    assert '"task.evaluate_test=false"' in runner
    assert "test_idx" not in analysis
    assert "test_acc" not in analysis


def test_forward_backward_finite_for_all_routing_variants():
    x, edge_index = inputs()
    for variant in Model.MODES:
        model = make_model(variant)
        output = model(x, edge_index)[0]
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()



def test_relation_and_capacity_control_parameters_within_five_percent():
    info = {"input_dim": 16, "text_dim": 8, "visual_dim": 8}
    counts = {}
    for variant in ("relation", "capacity_control"):
        model = Model(
            OmegaConf.create({"model": {
                "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
                "context_dim": 128, "latent_dim": 32,
                "routing_variant": variant,
            }}),
            info,
        )
        counts[variant] = sum(parameter.numel() for parameter in model.parameters())
    difference_pct = abs(counts["relation"] - counts["capacity_control"]) / counts["relation"]
    assert difference_pct <= 0.05


def test_trajectory_encoder_node_chunking_matches_unchunked_eval():
    model = make_model("trajectory").eval()
    encoder = model.trajectory_text
    torch.manual_seed(23)
    states = [torch.randn(11, 8) for _ in range(4)]
    chunked = encoder(states)
    tokens = torch.stack([encoder.state_proj(value) for value in states], dim=1)
    tokens = tokens + encoder.order_embedding.unsqueeze(0)
    full = encoder.pool_proj(encoder.encoder(tokens).mean(dim=1))
    assert torch.allclose(chunked, full, atol=1e-6, rtol=1e-6)

def test_checkpoint_reload_equivalence():
    x, edge_index = inputs()
    model = make_model("relation").eval()
    expected = model(x, edge_index)[0]
    clone = make_model("relation").eval()
    clone.load_state_dict(model.state_dict())
    assert torch.equal(expected, clone(x, edge_index)[0])

