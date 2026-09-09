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
        hrc_weight=0.0,
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


def test_mopf_hrc_zero_preserves_embedding_and_has_zero_aux() -> None:
    x, edge_index = _graph()
    model_off = _build(_cfg(hrc_weight=0.0))
    model_on = _build(_cfg(hrc_weight=1.0))
    model_on.load_state_dict(model_off.state_dict())
    with torch.no_grad():
        model_off.node_vector_text.fill_(0.25)
        model_off.node_vector_visual.fill_(-0.15)
        model_on.node_vector_text.fill_(0.25)
        model_on.node_vector_visual.fill_(-0.15)
    model_off.eval()
    model_on.eval()
    with torch.no_grad():
        z_off, _, _, aux_off, _ = model_off(x, edge_index)
        z_on, _, _, _, _ = model_on(x, edge_index)
    assert aux_off.item() == 0.0
    assert torch.equal(z_off, z_on)


def test_mopf_hrc_known_delta_matches_exact_mean_square() -> None:
    delta_text = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    delta_visual = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    expected = torch.tensor(3.75)
    assert torch.equal(mopf.MoPF._hrc_raw_loss(delta_text, delta_visual), expected)


def test_mopf_hrc_zero_delta_is_zero() -> None:
    zeros = torch.zeros(5, 3)
    assert mopf.MoPF._hrc_raw_loss(zeros, zeros).item() == 0.0


def test_mopf_hrc_shared_nonzero_residual_is_penalized() -> None:
    text = torch.full((4, 3), 0.5)
    visual = torch.full((4, 3), -0.25)
    assert mopf.MoPF._hrc_raw_loss(text, visual).item() > 0.0


def test_mopf_hrc_zero_mean_variation_is_not_penalized() -> None:
    text = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    visual = torch.tensor([[0.0, -2.0], [0.0, 2.0]])
    assert torch.allclose(mopf.MoPF._hrc_raw_loss(text, visual), torch.tensor(0.0))


def test_mopf_hrc_uses_current_training_node_index() -> None:
    text = torch.tensor([[2.0], [0.0]])
    visual = torch.tensor([[2.0], [0.0]])
    training_idx = torch.tensor([1])
    assert mopf.MoPF._hrc_raw_loss(text, visual, training_idx).item() == 0.0


def test_mopf_hrc_backward_reaches_node_residual_generator() -> None:
    x, edge_index = _graph()
    model = _build(_cfg(hrc_weight=1.0))
    with torch.no_grad():
        model.node_vector_text.fill_(0.2)
        model.node_vector_visual.fill_(-0.1)
    model.train()
    _, _, _, aux_loss, _ = model(x, edge_index)
    aux_loss.backward()

    for parameter in (
        model.node_vector_text,
        model.node_vector_visual,
        model.node_proj_text[0].weight,
        model.node_proj_visual[0].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_mopf_lp_sampler_expands_default_two_hops_to_propagation_order() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"name": "mopf", "num_layers": 3},
            "task": {"num_neighbors": [5, 5]},
        }
    )
    assert _resolve_lp_num_neighbors(cfg) == [5, 5, 5]


def test_mopf_absolute_mode_is_current_forward_default() -> None:
    x, edge_index = _graph()
    default_model = _build(_cfg(dropout=0.0))
    explicit_model = _build(
        _cfg(dropout=0.0, node_conditioner_mode="absolute", ppc_weight=0.0)
    )
    explicit_model.load_state_dict(default_model.state_dict())
    default_model.eval()
    explicit_model.eval()
    with torch.no_grad():
        default_z, _, _, default_aux, _ = default_model(x, edge_index)
        explicit_z, _, _, explicit_aux, _ = explicit_model(x, edge_index)
    assert torch.equal(default_z, explicit_z)
    assert default_aux.item() == explicit_aux.item() == 0.0


def test_mopf_pdc_zero_theta_matches_absolute_conditioner() -> None:
    x, edge_index = _graph()
    absolute = _build(_cfg(dropout=0.0, node_conditioner_mode="absolute"))
    pdc = _build(_cfg(dropout=0.0, node_conditioner_mode="pdc"))
    pdc.load_state_dict(absolute.state_dict())
    absolute.eval()
    pdc.eval()
    with torch.no_grad():
        absolute_components = absolute._encode_components(x, edge_index)
        pdc_components = pdc._encode_components(x, edge_index)
    assert torch.equal(pdc.pdc_rho("text"), torch.zeros(4))
    assert torch.equal(pdc.pdc_rho("visual"), torch.zeros(4))
    assert torch.allclose(
        absolute_components["delta_node_text"], pdc_components["delta_node_text"]
    )
    assert torch.allclose(
        absolute_components["delta_node_visual"], pdc_components["delta_node_visual"]
    )
    assert torch.allclose(absolute_components["z"], pdc_components["z"])


def test_mopf_pdc_rho_has_finite_gradient() -> None:
    x, edge_index = _graph()
    model = _build(_cfg(dropout=0.0, node_conditioner_mode="pdc"))
    model.train()
    z, _, _, _, _ = model(x, edge_index)
    z.square().mean().backward()
    assert model.pdc_theta_text.grad is not None
    assert model.pdc_theta_visual.grad is not None
    assert torch.isfinite(model.pdc_theta_text.grad).all()
    assert torch.isfinite(model.pdc_theta_visual.grad).all()


def test_mopf_pdc_order_zero_ignores_discrepancy() -> None:
    x, edge_index = _graph()
    baseline = _build(_cfg(dropout=0.0, node_conditioner_mode="pdc"))
    altered = _build(_cfg(dropout=0.0, node_conditioner_mode="pdc"))
    altered.load_state_dict(baseline.state_dict())
    with torch.no_grad():
        altered.pdc_theta_text[0] = 10.0
        altered.pdc_theta_visual[0] = -10.0
    baseline.eval()
    altered.eval()
    with torch.no_grad():
        baseline_components = baseline._encode_components(x, edge_index)
        altered_components = altered._encode_components(x, edge_index)
    assert altered.pdc_rho("text")[0].item() == 0.0
    assert altered.pdc_rho("visual")[0].item() == 0.0
    assert torch.equal(
        baseline_components["delta_node_text"][:, 0],
        altered_components["delta_node_text"][:, 0],
    )
    assert torch.equal(
        baseline_components["delta_node_visual"][:, 0],
        altered_components["delta_node_visual"][:, 0],
    )


def test_mopf_ppc_zero_is_current_forward_and_aux_equivalent() -> None:
    x, edge_index = _graph()
    default_model = _build(_cfg(dropout=0.0, ppc_weight=0.0))
    explicit_model = _build(
        _cfg(dropout=0.0, node_conditioner_mode="absolute", ppc_weight=0.0)
    )
    explicit_model.load_state_dict(default_model.state_dict())
    default_model.train()
    explicit_model.train()
    with torch.no_grad():
        default_z, _, _, default_aux, default_info = default_model(x, edge_index)
        explicit_z, _, _, explicit_aux, explicit_info = explicit_model(x, edge_index)
    assert torch.equal(default_z, explicit_z)
    assert default_aux.item() == explicit_aux.item() == 0.0
    assert default_info == explicit_info == {}


def test_mopf_ppc_identical_profiles_are_zero() -> None:
    profile = torch.randn(5, 4)
    assert mopf.MoPF._ppc_raw_loss(profile, profile, profile, profile).item() == 0.0


def test_mopf_ppc_is_invariant_to_centered_shared_shift() -> None:
    torch.manual_seed(7)
    text_1 = torch.randn(5, 4)
    visual_1 = torch.randn(5, 4)
    text_2 = torch.randn(5, 4)
    visual_2 = torch.randn(5, 4)
    base = mopf.MoPF._ppc_raw_loss(text_1, visual_1, text_2, visual_2)
    shift_text = torch.tensor([2.0, -1.0, 0.5, 3.0])
    shift_visual = torch.tensor([-2.0, 1.0, 0.25, -0.5])
    shifted = mopf.MoPF._ppc_raw_loss(
        text_1 + shift_text,
        visual_1 + shift_visual,
        text_2 + shift_text,
        visual_2 + shift_visual,
    )
    assert torch.allclose(base, shifted, atol=1e-7)


def test_mopf_ppc_different_centered_profiles_are_positive() -> None:
    first = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    second = torch.zeros_like(first)
    assert mopf.MoPF._ppc_raw_loss(first, first, second, second).item() > 0.0


def test_mopf_ppc_backward_is_finite() -> None:
    x, edge_index = _graph()
    model = _build(_cfg(dropout=0.1, ppc_weight=1.0))
    model.train()
    _, _, _, aux_loss, aux_info = model(x, edge_index)
    assert aux_loss.item() >= 0.0
    assert "ppc_raw" in aux_info
    aux_loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_mopf_ppc_eval_mode_uses_one_stochastic_view() -> None:
    x, edge_index = _graph()
    model = _build(_cfg(dropout=0.1, ppc_weight=1.0))
    model.eval()
    original = model._encode_components
    calls = {"count": 0}

    def counted_encode(input_x, input_edge_index):
        calls["count"] += 1
        return original(input_x, input_edge_index)

    model._encode_components = counted_encode
    with torch.no_grad():
        _, _, _, aux_loss, aux_info = model(x, edge_index)
    assert calls["count"] == 1
    assert aux_loss.item() == 0.0
    assert aux_info == {}
