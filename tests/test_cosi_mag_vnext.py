from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from src.models.cosi_filter import chebyshev_bank
from src.models.cosi_mag_vnext import Model
from src.models.cosi_relation import CrossModalRelationContext
from src.tasks.common import resolve_num_neighbors
from src.tasks.lp import (
    _build_message_edge_lookup,
    _exclude_positive_label_edges_by_global_eid,
)


def make_cfg(**updates):
    model = {
        "hidden_dim": 256,
        "dropout": 0.1,
        "norm": "layernorm",
        "max_order": 3,
        "relation_hidden_dim": 32,
        "relation_context_dim": 32,
        "relation_weight_scale": 0.3,
        "eps": 1e-8,
        "attention_dim": 64,
        "attention_dropout": 0.0,
        "attention_gate_init": 0.1,
        "eta_initial": [0.2, 0.0, 0.0],
        "semantic_gate_init": 0.1,
        "diffusion_add_self_loops": True,
        "fusion": "concat_residual_mlp",
    }
    model.update(updates)
    return OmegaConf.create({"model": model, "task": {"num_neighbors": [5, 5]}})


def make_model(dtype=torch.float32):
    cfg = make_cfg()
    model = Model(
        cfg,
        {"input_dim": 22, "text_dim": 12, "visual_dim": 10, "num_nodes": 6},
    )
    return model.to(dtype=dtype)


def small_graph(device="cpu"):
    x = torch.randn(6, 22, device=device)
    # Undirected 0--1--2--3 path, isolated node 4, and explicit loop on node 5.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 5], [1, 0, 2, 1, 3, 2, 5]],
        dtype=torch.long,
        device=device,
    )
    return x, edge_index


def test_forward_backward_cpu_shapes_and_finiteness():
    torch.manual_seed(7)
    model = make_model()
    x, edge_index = small_graph()
    model.train()
    z, h, logits, aux_loss, aux_info = model(x, edge_index)
    assert z.shape == (6, 256)
    assert h is None and logits is None
    assert aux_loss.shape == () and aux_loss.item() == 0.0
    assert aux_info == {}
    assert torch.isfinite(z).all()
    (z * torch.randn_like(z)).sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_crc_initial_weights_symmetry_and_normalized_operator():
    torch.manual_seed(11)
    crc = CrossModalRelationContext()
    h_text = torch.randn(4, 16)
    h_visual = torch.randn(4, 16)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    relation = crc(h_text, h_visual, edge_index)
    assert torch.equal(relation["weight_text"], torch.ones(6))
    assert torch.equal(relation["weight_visual"], torch.ones(6))

    # A nonzero scorer still gives orientation-invariant scores and weights.
    torch.nn.init.normal_(crc.relation_scorer[-1].weight, std=0.2)
    torch.nn.init.normal_(crc.relation_scorer[-1].bias, std=0.2)
    relation = crc(h_text, h_visual, edge_index)
    lookup = {
        (int(src), int(dst)): idx
        for idx, (src, dst) in enumerate(edge_index.T.tolist())
    }
    reverse = torch.tensor(
        [lookup[(int(dst), int(src))] for src, dst in edge_index.T.tolist()]
    )
    assert torch.allclose(relation["weight_text"], relation["weight_text"][reverse])
    assert torch.allclose(relation["weight_visual"], relation["weight_visual"][reverse])
    for suffix in ("text", "visual"):
        index = relation[f"P_{suffix}_edge_index"]
        weight = relation[f"P_{suffix}_edge_weight"]
        dense = torch.zeros(4, 4)
        dense[index[1], index[0]] = weight
        assert torch.allclose(dense, dense.T, atol=1e-6, rtol=1e-6)

    assert relation["physical_edge_index"].size(1) == edge_index.size(1)


def test_crc_weight_bounds_and_finiteness():
    crc = CrossModalRelationContext()
    with torch.no_grad():
        crc.relation_scorer[-1].weight.fill_(100.0)
        crc.relation_scorer[-1].bias.copy_(torch.tensor([100.0, -100.0]))
    h_text = torch.randn(3, 8)
    h_visual = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    relation = crc(h_text, h_visual, edge_index)
    for key in ("weight_text", "weight_visual"):
        weight = relation[key]
        assert torch.isfinite(weight).all()
        assert (weight > 0).all()
        assert (weight >= 0.7 - 1e-6).all()
        assert (weight <= 1.3 + 1e-6).all()


def test_relation_scorer_gradient_and_relation_context_path():
    torch.manual_seed(19)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    x, edge_index = small_graph()
    probe = torch.randn(6, 256)

    optimizer.zero_grad(set_to_none=True)
    parts = model._encode_components(x, edge_index)
    q = parts["relation"]["q"]
    q.retain_grad()
    loss = (parts["z"] * probe).sum()
    loss.backward()
    last = model.crc.relation_scorer[-1]
    assert last.weight.grad is not None
    assert torch.isfinite(last.weight.grad).all()
    assert last.weight.grad.abs().sum() > 0
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert q.grad.abs().sum() > 0
    optimizer.step()

    # Zero output-layer initialization delays gradients to earlier scorer layers
    # until after the first update; they must then become trainable.
    optimizer.zero_grad(set_to_none=True)
    z, *_ = model(x, edge_index)
    (z * probe).sum().backward()
    first = model.crc.relation_scorer[0]
    assert first.weight.grad is not None
    assert torch.isfinite(first.weight.grad).all()
    assert first.weight.grad.abs().sum() > 0


def test_sparse_chebyshev_recurrence_matches_dense_reference():
    torch.manual_seed(23)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    edge_weight = torch.tensor([0.5, 0.5, 0.25, 0.25, 0.75, 0.75])
    x = torch.randn(4, 5)
    dense_p = torch.zeros(4, 4)
    dense_p[edge_index[1], edge_index[0]] = edge_weight
    sparse = chebyshev_bank(x, edge_index, edge_weight, max_order=3)
    dense = [x, dense_p @ x]
    for _ in range(2, 4):
        dense.append(2.0 * dense_p @ dense[-1] - dense[-2])
    dense = torch.stack(dense, dim=1)
    assert sparse.shape == (4, 4, 5)
    assert torch.allclose(sparse, dense, atol=1e-6, rtol=1e-6)


def test_isolated_nodes_have_no_structural_order_difference():
    model = make_model().eval()
    x = torch.randn(2, 256)
    identity_edges = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    relation = model.crc(x, x.clone(), identity_edges)
    assert torch.equal(relation["descriptor"], torch.zeros_like(relation["descriptor"]))
    bank = chebyshev_bank(
        x,
        relation["P_text_edge_index"],
        relation["P_text_edge_weight"],
        max_order=3,
    )
    assert torch.allclose(bank, x.unsqueeze(1).expand(-1, 4, -1), atol=1e-6, rtol=1e-6)
    result = model.interaction.forward_modality(x, bank, relation["q"], "text")
    assert torch.allclose(result["differences"], torch.zeros_like(result["differences"]), atol=1e-6)


def test_rcmi_initial_eta_and_gates_match_design():
    model = make_model().eval()
    h0 = torch.randn(3, 256)
    bank = torch.randn(3, 4, 256)
    q = torch.randn(3, 32)
    result = model.interaction.forward_modality(h0, bank, q, "text")
    assert torch.allclose(result["eta"][:, 0], torch.full((3,), 0.2), atol=1e-6)
    assert torch.allclose(result["eta"][:, 1:], torch.zeros(3, 2), atol=1e-7)
    assert result["attention_gate"].item() == pytest.approx(0.1)
    assert torch.allclose(
        result["semantic_gate"], torch.full_like(result["semantic_gate"], 0.1), atol=1e-6
    )


def test_fixed_architecture_config_guards():
    info = {"input_dim": 22, "text_dim": 12, "visual_dim": 10}
    no_loops = make_cfg()
    no_loops.model.diffusion_add_self_loops = False
    with pytest.raises(ValueError, match="self-loop"):
        Model(no_loops, info)
    wrong_fusion = make_cfg()
    wrong_fusion.model.fusion = "mean"
    with pytest.raises(ValueError, match="concat_residual_mlp"):
        Model(wrong_fusion, info)


def test_empty_single_node_duplicate_edges_and_double_dtype():
    torch.manual_seed(29)
    model = make_model()
    for x, edge_index in (
        (torch.randn(5, 22), torch.empty((2, 0), dtype=torch.long)),
        (torch.randn(1, 22), torch.empty((2, 0), dtype=torch.long)),
        (
            torch.randn(3, 22),
            torch.tensor([[0, 1, 0, 1, 1, 2], [1, 0, 1, 0, 2, 1]]),
        ),
    ):
        z, *_ = model(x, edge_index)
        assert torch.isfinite(z).all()
    double_model = make_model(dtype=torch.float64)
    double_x = torch.randn(3, 22, dtype=torch.float64)
    double_edges = torch.tensor([[0, 1], [1, 0]])
    z, *_ = double_model(double_x, double_edges)
    assert z.dtype == torch.float64
    assert torch.isfinite(z).all()
    integer_model = make_model()
    integer_z, *_ = integer_model(torch.ones(2, 22, dtype=torch.long), torch.empty((2, 0), dtype=torch.long))
    assert torch.isfinite(integer_z).all()


def test_eval_forward_inference_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(31)
    model = make_model().eval()
    x, edge_index = small_graph()
    expected, *_ = model(x, edge_index)
    inferred = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=2)
    assert torch.allclose(inferred, expected.cpu(), atol=1e-6, rtol=1e-6)

    path = tmp_path / "vnext.pt"
    torch.save(model.state_dict(), path)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    actual, *_ = restored(x, edge_index)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    model.train()
    before = model.training
    model.relation_diagnostics(x, edge_index)
    assert model.training is before


def test_lp_sampler_depth_and_positive_edge_mask_both_directions():
    cfg = make_cfg()
    cfg.model.num_layers = 3
    assert resolve_num_neighbors(cfg) == [5, 5, 5]

    # The production global-eid path removes both directions and all duplicate
    # instances of the supervised physical edge before model.forward.
    message_edge_index = torch.tensor(
        [[0, 1, 0, 1, 1, 2], [1, 0, 1, 0, 2, 1]], dtype=torch.long
    )
    lookup = _build_message_edge_lookup(message_edge_index, num_nodes=3)
    batch = Data(
        edge_index=message_edge_index.clone(),
        e_id=torch.arange(message_edge_index.size(1)),
        n_id=torch.arange(3),
        edge_label_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_label=torch.tensor([1.0, 0.0]),
    )
    removed = _exclude_positive_label_edges_by_global_eid(batch, lookup)
    assert removed == 4
    remaining = set(map(tuple, batch.edge_index.T.tolist()))
    assert remaining == {(1, 2), (2, 1)}


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_model_mode_switching(mode):
    model = make_model()
    getattr(model, mode)()
    assert model.training is (mode == "train")
    x, edge_index = small_graph()
    z, *_ = model(x, edge_index)
    assert torch.isfinite(z).all()
