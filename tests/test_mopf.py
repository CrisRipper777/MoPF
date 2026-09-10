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
        node_conditioner_mode="absolute",
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


def _pdc_model() -> mopf.MoPF:
    model = _build(_cfg(node_conditioner_mode="pdc"))
    model.eval()
    with torch.no_grad():
        model.pdc_theta_text.copy_(torch.tensor([0.0, 0.35, -0.55, 0.8]))
        model.pdc_theta_visual.copy_(torch.tensor([0.0, -0.25, 0.65, -0.45]))
        model.node_vector_text.copy_(torch.linspace(-0.7, 0.8, 16).view(4, 4))
        model.node_vector_visual.copy_(torch.linspace(0.9, -0.6, 16).view(4, 4))
    return model


def test_pdc_analysis_all_ones_matches_formal_forward() -> None:
    x, edge_index = _graph()
    model = _pdc_model()
    ones = [1.0] * 4
    with torch.no_grad():
        formal, _, _, _, _ = model(x, edge_index)
        conditioned = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=ones, visual_mask=ones
        )

    assert torch.equal(formal, conditioned["z"])
    assert torch.equal(
        conditioned["delta_node_text"],
        model._encode_components(x, edge_index)["delta_node_text"],
    )
    assert torch.equal(
        conditioned["delta_node_visual"],
        model._encode_components(x, edge_index)["delta_node_visual"],
    )


def test_pdc_analysis_all_zero_is_rho_zero_conditioner() -> None:
    x, edge_index = _graph()
    model = _pdc_model()
    zeros = [0.0] * 4
    with torch.no_grad():
        masked = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=zeros, visual_mask=zeros
        )
        rho_zero = model._encode_components(
            x,
            edge_index,
            pdc_rho_text=torch.zeros(4),
            pdc_rho_visual=torch.zeros(4),
        )

    for modality in ("text", "visual"):
        for left, right in zip(
            masked[f"conditioned_bases_{modality}"],
            rho_zero[f"conditioned_bases_{modality}"],
            strict=True,
        ):
            assert torch.equal(left, right)
    for key in (
        "delta_node_text",
        "delta_node_visual",
        "eta_text",
        "eta_visual",
        "z_text",
        "z_visual",
        "z",
    ):
        assert torch.equal(masked[key], rho_zero[key])


def test_pdc_analysis_modality_masks_do_not_cross_streams() -> None:
    x, edge_index = _graph()
    model = _pdc_model()
    ones = [1.0] * 4
    zeros = [0.0] * 4
    with torch.no_grad():
        on = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=ones, visual_mask=ones
        )
        text_off = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=zeros, visual_mask=ones
        )
        visual_off = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=ones, visual_mask=zeros
        )

    for left, right in zip(
        on["conditioned_bases_visual"],
        text_off["conditioned_bases_visual"],
        strict=True,
    ):
        assert torch.equal(left, right)
    for left, right in zip(
        on["conditioned_bases_text"],
        visual_off["conditioned_bases_text"],
        strict=True,
    ):
        assert torch.equal(left, right)


def test_pdc_analysis_order_mask_only_changes_that_conditioner_order_and_k0() -> None:
    x, edge_index = _graph()
    model = _pdc_model()
    ones = [1.0] * 4
    order = 2
    order_off = [1.0] * 4
    order_off[order] = 0.0
    with torch.no_grad():
        on = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=ones, visual_mask=ones
        )
        off = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=order_off, visual_mask=ones
        )

    assert torch.equal(
        on["conditioned_bases_text"][0], off["conditioned_bases_text"][0]
    )
    for current in range(1, 4):
        if current != order:
            assert torch.equal(
                on["conditioned_bases_text"][current],
                off["conditioned_bases_text"][current],
            )
    assert not torch.equal(
        on["conditioned_bases_text"][order],
        off["conditioned_bases_text"][order],
    )


def test_pdc_analysis_does_not_modify_formal_forward_or_parameters() -> None:
    x, edge_index = _graph()
    model = _pdc_model()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        formal_before, _, _, _, _ = model(x, edge_index)
        model.analysis_encode_with_pdc_mask(
            x,
            edge_index,
            text_mask=[0.0, 0.0, 1.0, 1.0],
            visual_mask=[1.0, 0.0, 1.0, 0.0],
        )
        formal_after, _, _, _, _ = model(x, edge_index)

    assert torch.equal(formal_before, formal_after)
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)


def _v2_model(mode: str) -> mopf.MoPF:
    return _build(_cfg(node_conditioner_mode=mode))


def test_pdc_v2_sep_zero_theta_has_zero_discrepancy_branch_and_current_state_equivalence() -> None:
    x, edge_index = _graph()
    current = _build(_cfg(node_conditioner_mode="absolute"))
    v2 = _v2_model("pdc_v2_sep")
    with torch.no_grad():
        for model in (current, v2):
            model.node_vector_text.fill_(0.2)
            model.node_vector_visual.fill_(-0.15)
        current_components = current._encode_components(x, edge_index)
        v2_components = v2._encode_components(x, edge_index)

    assert torch.equal(v2.pdc_rho("text")[0], torch.tensor(0.0))
    assert torch.equal(v2.pdc_rho("visual")[0], torch.tensor(0.0))
    for modality in ("text", "visual"):
        for branch in v2_components[f"pdc_aux_{modality}"]["branch"]:
            assert torch.equal(branch, torch.zeros_like(branch))
    assert torch.allclose(
        current_components["delta_node_text"],
        v2_components["delta_node_text"],
        atol=1e-7,
    )
    assert torch.allclose(
        current_components["delta_node_visual"],
        v2_components["delta_node_visual"],
        atol=1e-7,
    )
    for modality in ("text", "visual"):
        for current_projected, v2_projected in zip(
            [
                current.node_proj_text[order](current_components["bases_text"][order])
                for order in range(4)
            ]
            if modality == "text"
            else [
                current.node_proj_visual[order](current_components["bases_visual"][order])
                for order in range(4)
            ],
            v2_components[f"pdc_aux_{modality}"]["state_projection"],
            strict=True,
        ):
            assert torch.allclose(current_projected, v2_projected, atol=1e-7)


def test_pdc_v2_discrepancy_and_rho_gradients_are_finite() -> None:
    x, edge_index = _graph()
    model = _v2_model("pdc_v2_sep")
    with torch.no_grad():
        model.pdc_theta_text.fill_(0.4)
        model.pdc_theta_visual.fill_(-0.3)
        model.node_vector_text.fill_(0.2)
        model.node_vector_visual.fill_(-0.15)
    model.train()
    z, _, _, _, _ = model(x, edge_index)
    z.square().mean().backward()

    for module in (model.node_disc_proj_text, model.node_disc_proj_visual):
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    for parameter in (model.pdc_theta_text, model.pdc_theta_visual):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_pdc_v2_full_has_trainable_rho0_and_uses_d0_h1_minus_h0() -> None:
    x, edge_index = _graph()
    model = _v2_model("pdc_v2_full")
    assert model.pdc_theta_text.requires_grad
    assert model.pdc_theta_visual.requires_grad
    assert torch.equal(model.pdc_rho("text")[0], torch.tensor(0.0))
    assert torch.equal(model.pdc_rho("visual")[0], torch.tensor(0.0))
    model.eval()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
    for modality in ("text", "visual"):
        d0 = components[f"pdc_aux_{modality}"]["discrepancy"][0]
        expected = components[f"bases_{modality}"][1] - components[f"bases_{modality}"][0]
        assert torch.equal(d0, expected)


def test_pdc_v2_full_all_off_matches_explicit_zero_rho_frozen_path() -> None:
    x, edge_index = _graph()
    model = _v2_model("pdc_v2_full")
    with torch.no_grad():
        model.pdc_theta_text.copy_(torch.tensor([0.35, -0.2, 0.5, -0.4]))
        model.pdc_theta_visual.copy_(torch.tensor([-0.25, 0.3, -0.45, 0.2]))
        model.node_vector_text.fill_(0.2)
        model.node_vector_visual.fill_(-0.15)
        masked = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=[0.0] * 4, visual_mask=[0.0] * 4
        )
        explicit = model._encode_components(
            x,
            edge_index,
            pdc_rho_text=torch.zeros(4),
            pdc_rho_visual=torch.zeros(4),
        )
    for key in ("delta_node_text", "delta_node_visual", "eta_text", "eta_visual", "z"):
        assert torch.equal(masked[key], explicit[key])


def test_pdc_v2_full_order0_off_only_changes_order0_path() -> None:
    x, edge_index = _graph()
    model = _v2_model("pdc_v2_full")
    with torch.no_grad():
        model.pdc_theta_text.copy_(torch.tensor([0.6, -0.2, 0.5, -0.4]))
        model.pdc_theta_visual.copy_(torch.tensor([-0.5, 0.3, -0.45, 0.2]))
        model.node_vector_text.fill_(0.2)
        model.node_vector_visual.fill_(-0.15)
        on = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=[1.0] * 4, visual_mask=[1.0] * 4
        )
        off = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=[0.0, 1.0, 1.0, 1.0], visual_mask=[1.0] * 4
        )
    assert not torch.equal(on["delta_node_text"][:, 0], off["delta_node_text"][:, 0])
    assert torch.equal(on["delta_node_text"][:, 1:], off["delta_node_text"][:, 1:])
    assert torch.equal(on["delta_node_visual"], off["delta_node_visual"])


def test_mopf_pdc_v2_lp_forward_is_finite_and_uses_three_hop_sampler() -> None:
    x, edge_index = _graph()
    for mode in ("pdc_v2_sep", "pdc_v2_full"):
        model = _v2_model(mode)
        z, _, _, aux_loss, _ = model(x, edge_index)
        assert z.shape == (6, 8)
        assert torch.isfinite(z).all()
        assert torch.isfinite(aux_loss)

    cfg = OmegaConf.create(
        {"model": {"name": "mopf", "num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}}
    )
    assert _resolve_lp_num_neighbors(cfg) == [5, 5, 5]


def test_pdc_v2_frozen_masks_do_not_modify_checkpoint_state() -> None:
    x, edge_index = _graph()
    for mode in ("pdc_v2_sep", "pdc_v2_full"):
        model = _v2_model(mode)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            model.analysis_encode_with_pdc_mask(
                x,
                edge_index,
                text_mask=[0.0, 0.0, 1.0, 1.0],
                visual_mask=[1.0, 0.0, 1.0, 0.0],
            )
        for name, value in model.state_dict().items():
            assert torch.equal(before[name], value)


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
