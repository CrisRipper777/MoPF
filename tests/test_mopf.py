from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models import mopf
from src.models.factory import build_model
from src.tasks.lp import _resolve_lp_num_neighbors


class _CfgNode(dict):
    def __getattr__(self, name: str):
        return self[name]


def _cfg(**overrides) -> _CfgNode:
    model = _CfgNode(
        name="mopf",
        version="mopf",
        hidden_dim=8,
        dropout=0.0,
        norm="layernorm",
        max_order=3,
        num_layers=3,
        map_prior_restart=0.15,
        map_prior_order=2,
        diffusion_add_self_loops=True,
        edge_weight_mode="separate_cos",
        edge_weight_min=0.1,
        edge_weight_temperature=2.0,
        filter_rank=4,
        global_filter_trainable=True,
        use_modality_residual=True,
        use_node_residual=True,
        fusion_mode="concat_residual_mlp",
        export_aux_stats=False,
        export_node_aux=False,
        export_edge_aux=False,
        lp_pair_operator=None,
    )
    model.update(overrides)
    return _CfgNode(model=model, task=_CfgNode(num_neighbors=[5, 5]))


def _graph() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def _build(cfg: _CfgNode | None = None) -> mopf.MoPF:
    torch.manual_seed(123)
    return mopf.Model(
        cfg or _cfg(),
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )


def test_mopf_factory_and_forward_contract_is_finite() -> None:
    cfg = _cfg()
    model = build_model(
        cfg,
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )
    x, edge_index = _graph()
    z, none_a, none_b, aux_loss, aux_info = model(x, edge_index)

    assert isinstance(model, mopf.MoPF)
    assert model.out_dim == 8
    assert z.shape == (6, 8)
    assert none_a is None and none_b is None
    assert torch.isfinite(z).all()
    assert aux_loss.dim() == 0
    assert aux_loss.item() == 0.0
    assert aux_info == {}


def test_mopf_initial_hierarchical_filter_is_exact_map_prior() -> None:
    x, edge_index = _graph()
    model = _build()
    expected = torch.tensor([0.15, 0.1275, 0.7225, 0.0])

    assert torch.equal(model.gamma_global.detach(), expected)
    assert torch.equal(
        model.delta_gamma_text.detach(), torch.zeros(4, dtype=torch.float32)
    )
    assert torch.equal(
        model.delta_gamma_visual.detach(), torch.zeros(4, dtype=torch.float32)
    )
    assert torch.equal(
        model.node_vector_text.detach(), torch.zeros((4, 4), dtype=torch.float32)
    )
    assert torch.equal(
        model.node_vector_visual.detach(), torch.zeros((4, 4), dtype=torch.float32)
    )

    model.eval()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
    assert torch.equal(components["delta_node_text"], torch.zeros((6, 4)))
    assert torch.equal(components["delta_node_visual"], torch.zeros((6, 4)))
    assert torch.equal(components["eta_text"], expected.expand(6, -1))
    assert torch.equal(components["eta_visual"], expected.expand(6, -1))


def test_mopf_semantic_graph_modes_and_sparse_weight_shapes() -> None:
    x, edge_index = _graph()
    model = _build()
    model.eval()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
    edges = components["edges"]
    assert edges["w_t"].shape == (edge_index.size(1),)
    assert edges["w_v"].shape == (edge_index.size(1),)
    assert edges["cos_t"].shape == (edge_index.size(1),)
    assert edges["cos_v"].shape == (edge_index.size(1),)
    assert torch.isfinite(edges["w_t"]).all()
    assert torch.isfinite(edges["w_v"]).all()

    for mode in ("raw_uniform", "shared_avg_cos"):
        mode_model = _build(_cfg(edge_weight_mode=mode))
        z, _, _, aux_loss, _ = mode_model(x, edge_index)
        assert torch.isfinite(z).all()
        assert aux_loss.item() == 0.0
        with torch.no_grad():
            mode_edges = mode_model._encode_components(x, edge_index)["edges"]
        if mode == "raw_uniform":
            assert torch.equal(mode_edges["w_t"], torch.ones(edge_index.size(1)))
            assert torch.equal(mode_edges["w_v"], torch.ones(edge_index.size(1)))
        else:
            assert torch.equal(mode_edges["w_t"], mode_edges["w_v"])


def test_mopf_filter_parameters_receive_finite_gradients() -> None:
    x, edge_index = _graph()
    model = _build()
    model.train()
    z, _, _, aux_loss, _ = model(x, edge_index)
    target = torch.linspace(-1.0, 1.0, z.numel(), dtype=z.dtype).view_as(z)
    ((z * target).mean() + aux_loss).backward()

    for parameter in (
        model.gamma_global,
        model.delta_gamma_text,
        model.delta_gamma_visual,
        model.node_vector_text,
        model.node_vector_visual,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_mopf_inference_matches_eval_forward_and_uses_max_order_layers() -> None:
    x, edge_index = _graph()
    model = _build()
    model.eval()
    with torch.no_grad():
        forward, _, _, _, _ = model(x, edge_index)
    inferred = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=2)

    assert model.num_layers == 3
    assert torch.allclose(forward.cpu(), inferred, atol=1e-6)


def test_mopf_empty_edge_graph_and_residual_switches_run() -> None:
    x, _ = _graph()
    empty_edge_index = torch.empty((2, 0), dtype=torch.long)
    model = _build(
        _cfg(
            global_filter_trainable=False,
            use_modality_residual=False,
            use_node_residual=False,
        )
    )
    z, _, _, aux_loss, _ = model(x, empty_edge_index)

    assert not model.gamma_global.requires_grad
    assert z.shape == (6, 8)
    assert torch.isfinite(z).all()
    assert aux_loss.item() == 0.0
    with torch.no_grad():
        components = model._encode_components(x, empty_edge_index)
    assert torch.equal(components["delta_node_text"], torch.zeros((6, 4)))
    assert torch.equal(components["delta_node_visual"], torch.zeros((6, 4)))


def test_mopf_lp_sampler_expands_default_two_hops_to_propagation_order() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"name": "mopf", "num_layers": 3},
            "task": {"num_neighbors": [5, 5]},
        }
    )
    assert _resolve_lp_num_neighbors(cfg) == [5, 5, 5]
