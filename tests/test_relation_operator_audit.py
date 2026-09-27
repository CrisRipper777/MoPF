from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data.types import MAGData
from src.models.cmrf_probe import Model as CMRFModel
from src.models.relation_operator_audit import Model
from src.tasks.nc import run_nc


def make_model(variant: str = "plain", chunk: int = 3, dropout: float = 0.0) -> Model:
    cfg = OmegaConf.create(
        {
            "model": {
                "hidden_dim": 256,
                "max_order": 3,
                "dropout": dropout,
                "operator_variant": variant,
                "edge_chunk_size": chunk,
            }
        }
    )
    return Model(cfg, {"input_dim": 7, "text_dim": 4, "visual_dim": 3})


def graph_inputs():
    torch.manual_seed(21)
    x = torch.randn(7, 7)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 0, 1],
            [1, 0, 2, 1, 3, 2, 5, 5, 4, 6, 0, 2],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def copy_frozen_backbone(source, target):
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = {key: value for key, value in source_state.items() if key in target_state}
    target.load_state_dict(shared, strict=False)


def test_plain_exact_equivalence_to_cmrf_uniform():
    torch.manual_seed(7)
    x, edge_index = graph_inputs()
    cmrf_cfg = OmegaConf.create(
        {"model": {"hidden_dim": 256, "max_order": 3, "dropout": 0.0,
                   "composition_mode": "uniform", "controller_latent_dim": 32,
                   "controller_hidden_dim": 64, "controller_lambda": 0.25}}
    )
    audit_cfg = OmegaConf.create(
        {"model": {"hidden_dim": 256, "max_order": 3, "dropout": 0.0,
                   "operator_variant": "plain", "edge_chunk_size": 3}}
    )
    cmrf = CMRFModel(cmrf_cfg, {"input_dim": 7, "text_dim": 4, "visual_dim": 3}).eval()
    audit = Model(audit_cfg, {"input_dim": 7, "text_dim": 4, "visual_dim": 3}).eval()
    copy_frozen_backbone(cmrf, audit)
    with torch.no_grad():
        expected = cmrf.analyze(x, edge_index)
        actual = audit.analyze(x, edge_index)
    for modality in ("text", "visual"):
        errors = [
            (expected[f"S_{modality}"][k] - actual[f"C_{modality}"][k]).abs().max().item()
            for k in range(4)
        ]
        assert max(errors) < 1e-6
        assert (expected[f"Z_{modality}"] - actual[f"Z_{modality}"]).abs().max().item() < 1e-6
    assert (expected["fused_z"] - actual["fused_z"]).abs().max().item() < 1e-6


def test_chunked_edge_aggregation_matches_explicit_dense_operator_and_orientation():
    model = make_model("scalar_weight", chunk=2).eval()
    _, edge_index = graph_inputs()
    operator = model._build_propagation_operator(edge_index, 7, torch.float32)
    target, source, weight, self_target, self_source, self_weight = model._operator_edges(operator)
    torch.manual_seed(12)
    state = torch.randn(7, 256)
    got, _ = model._propagate_semantic(
        state, target, source, weight, self_target, self_source, self_weight,
        torch.ones(target.numel()), modality="text",
    )
    expected = operator.to_dense() @ state
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_self_loops_bypass_relation_encoder_and_experts():
    model = make_model("relation_expert").eval()
    x, _ = graph_inputs()
    only_self = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    calls = {"relation": 0, "experts": 0}
    handles = [
        model.relation_text.encoder.register_forward_hook(
            lambda *_: calls.__setitem__("relation", calls["relation"] + 1)
        )
    ]
    handles += [
        expert.register_forward_hook(
            lambda *_: calls.__setitem__("experts", calls["experts"] + 1)
        )
        for expert in model.experts_text
    ]
    with torch.no_grad():
        result = model.analyze(x, only_self)
    for handle in handles:
        handle.remove()
    assert result["num_nonself_edges"] == 0
    assert result["num_self_edges"] == x.size(0)
    assert calls == {"relation": 0, "experts": 0}
    op = model._get_operator(only_self, x.size(0), x.dtype)
    assert torch.equal(result["C_text"][1], op.to_dense() @ result["H0_text"])




def test_self_loop_bypasses_scalar_weight_head():
    model = make_model("scalar_weight").eval()
    x, _ = graph_inputs()
    only_self = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    calls = {"relation_mlp": 0, "scalar_head": 0}
    handles = [
        model.relation_text.encoder.register_forward_hook(
            lambda *_: calls.__setitem__("relation_mlp", calls["relation_mlp"] + 1)
        ),
        model.scalar_head_text.proj.register_forward_hook(
            lambda *_: calls.__setitem__("scalar_head", calls["scalar_head"] + 1)
        ),
    ]
    with torch.no_grad():
        result = model.analyze(x, only_self)
    for handle in handles:
        handle.remove()
    assert result["num_nonself_edges"] == 0
    assert calls == {"relation_mlp": 0, "scalar_head": 0}
    operator = model._get_operator(only_self, x.size(0), x.dtype)
    assert torch.equal(result["C_text"][1], operator.to_dense() @ result["H0_text"])


def test_global_and_relation_expert_parameter_parity():
    global_model = make_model("global_expert")
    relation_model = make_model("relation_expert")
    global_count = sum(parameter.numel() for parameter in global_model.parameters() if parameter.requires_grad)
    relation_count = sum(parameter.numel() for parameter in relation_model.parameters() if parameter.requires_grad)
    assert global_count == relation_count


def test_identical_relation_tokens_make_global_and_relation_coordination_equivalent():
    torch.manual_seed(13)
    x, edge_index = graph_inputs()
    global_model = make_model("global_expert").eval()
    relation_model = make_model("relation_expert").eval()
    global_model.load_state_dict(relation_model.state_dict())
    operator = relation_model._get_operator(edge_index, x.size(0), x.dtype)
    target, *_ = relation_model._operator_edges(operator)
    token = torch.randn(1, 64).expand(target.numel(), -1).clone()
    with torch.no_grad():
        expected = global_model.analyze(
            x, edge_index,
            relation_state_override_text=token,
            relation_state_override_visual=token,
        )
        actual = relation_model.analyze(
            x, edge_index,
            relation_state_override_text=token,
            relation_state_override_visual=token,
        )
    assert (expected["fused_z"] - actual["fused_z"]).abs().max().item() < 1e-6


def test_edge_chunk_size_does_not_change_forward_output():
    torch.manual_seed(15)
    x, edge_index = graph_inputs()
    small = make_model("relation_expert", chunk=2).eval()
    large = make_model("relation_expert", chunk=100).eval()
    large.load_state_dict(small.state_dict())
    with torch.no_grad():
        expected = small(x, edge_index)[0]
        actual = large(x, edge_index)[0]
    assert torch.allclose(expected, actual, atol=2e-6, rtol=2e-6)


def test_relation_router_prototypes_and_each_expert_receive_finite_gradients():
    torch.manual_seed(17)
    model = make_model("relation_expert", chunk=2).train()
    x, edge_index = graph_inputs()
    output = model(x, edge_index)[0]
    output.square().mean().backward()
    required = [
        model.relation_text.reduce.weight,
        model.relation_text.encoder[0].weight,
        model.query_text.weight,
        model.prototypes_text,
        model.relation_visual.reduce.weight,
        model.relation_visual.encoder[0].weight,
        model.query_visual.weight,
        model.prototypes_visual,
    ]
    required.extend(expert.down.weight for expert in model.experts_text)
    required.extend(expert.up.weight for expert in model.experts_text)
    required.extend(expert.down.weight for expert in model.experts_visual)
    required.extend(expert.up.weight for expert in model.experts_visual)
    for parameter in required:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0


def test_initialization_starts_near_plain_physical_propagation():
    torch.manual_seed(19)
    x, edge_index = graph_inputs()
    plain = make_model("plain").eval()
    scalar = make_model("scalar_weight").eval()
    global_model = make_model("global_expert").eval()
    relation_model = make_model("relation_expert").eval()
    for model in (scalar, global_model, relation_model):
        copy_frozen_backbone(plain, model)
    with torch.no_grad():
        plain_result = plain.analyze(x, edge_index)
        scalar_result = scalar.analyze(x, edge_index)
        global_result = global_model.analyze(x, edge_index)
        relation_result = relation_model.analyze(x, edge_index)
    for modality in ("text", "visual"):
        assert scalar_result["routing_stats_" + modality]["scalar_weight_mean"].item() == 1.0
        assert (scalar_result[f"Z_{modality}"] - plain_result[f"Z_{modality}"]).abs().max().item() < 2e-5
        assert (global_result[f"Z_{modality}"] - plain_result[f"Z_{modality}"]).abs().max().item() < 5e-3
        assert (relation_result[f"Z_{modality}"] - plain_result[f"Z_{modality}"]).abs().max().item() < 5e-3


def test_all_variants_have_identical_effective_physical_operator():
    _, edge_index = graph_inputs()
    operators = [
        make_model(variant)._build_propagation_operator(edge_index, 7, torch.float32)
        for variant in ("plain", "scalar_weight", "global_expert", "relation_expert")
    ]
    reference = operators[0]
    for operator in operators[1:]:
        assert torch.equal(reference.indices(), operator.indices())
        assert torch.equal(reference.values(), operator.values())


def test_evaluate_test_false_excludes_test_targets_metrics_and_checkpoint_fields(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "model": {"name": "relation_operator_audit", "hidden_dim": 256,
                      "max_order": 3, "dropout": 0.0, "operator_variant": "plain",
                      "edge_chunk_size": 2},
            "task": {"name": "nc", "protocol_version": "unified_full_graph_nc_v1",
                     "training_mode": "full_graph", "optimizer": "adamw", "lr": 1e-3,
                     "weight_decay": 1e-4, "epochs": 1, "batch_size": 4,
                     "num_neighbors": 3, "inference_mode": "full", "inference_batch_size": 8,
                     "eval_every": 1, "patience": 2, "early_stop_min_epoch": 1,
                     "early_stop_min_delta": 0.0, "grad_clip": 1.0,
                     "loss": {"aux_weight": 0.0}, "evaluate_test": False,
                     "save_ckpt_path": str(tmp_path / "best.pt")},
            "seed": 42,
            "num_runs": 1,
        }
    )
    torch.manual_seed(23)
    x = torch.randn(6, 7)
    y = torch.tensor([0, 1, 1, 0, 999, -100], dtype=torch.long)
    data = MAGData(
        name="test-isolation", source="unit-test", task="nc", x=x,
        x_i=x[:, 4:].contiguous(), x_t=x[:, :4].contiguous(),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        num_nodes=6, y=y, train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2, 3]), test_idx=torch.tensor([4, 5]), num_classes=2,
    )
    metrics = run_nc(cfg, data, torch.device("cpu"), logging.getLogger("test-nc-isolation"))
    assert set(metrics) == {"val_acc", "val_macro_f1"}
    payload = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert payload["selection"] == "best_val_accuracy"
    assert set(payload["metrics"]) == {"val_acc", "val_macro_f1"}
    assert not any("test" in key.lower() for key in payload["metrics"])


def test_formal_launcher_has_exactly_sixty_test_disabled_contexts():
    from scripts.run_relation_operator_audit import formal_contexts, build_command

    contexts = formal_contexts("cuda:1")
    assert len(contexts) == 60
    assert {context.variant for context in contexts} == {
        "plain", "scalar_weight", "global_expert", "relation_expert"
    }
    assert {context.dataset for context in contexts} == {
        "Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"
    }
    assert {context.seed for context in contexts} == {42, 43, 44}
    assert len({context.run_dir for context in contexts}) == 60
    assert len({context.checkpoint for context in contexts}) == 60
    command = " ".join(build_command(contexts[0]))
    assert "task.evaluate_test=false" in command
    assert "task.epochs=300" in command
    assert "model.max_order=3" in command
    assert "num_runs=1" in command


def test_smoke_launcher_uses_separate_nonformal_namespace():
    from scripts.run_relation_operator_audit import formal_contexts, smoke_contexts

    context = smoke_contexts("Movies", "plain", 42, 1, "cuda:1", 1024)[0]
    assert context.run_dir.parts[-4] == "smoke"
    assert context.mode == "smoke"
    assert context.run_dir not in {item.run_dir for item in formal_contexts("cuda:1")}
