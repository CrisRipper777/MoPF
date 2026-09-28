from __future__ import annotations

import csv
import gzip

import torch
from omegaconf import OmegaConf

from scripts import analyze_relation_operator_diagnostics as diag
from src.models.relation_operator_audit import Model


def _make_model(variant: str, seed: int = 4):
    torch.manual_seed(seed)
    cfg = OmegaConf.create({
        "model": {
            "hidden_dim": 256,
            "max_order": 3,
            "dropout": 0.0,
            "edge_chunk_size": 4,
            "operator_variant": variant,
        }
    })
    info = {"input_dim": 6, "text_dim": 3, "visual_dim": 3, "num_nodes": 7, "num_classes": 2}
    model = Model(cfg, info).eval()
    x = torch.randn(7, 6)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 0, 6],
    ])
    return model, x, edge_index


def test_prepared_normal_a2_a3_matches_current_analyzer_and_strict_reload():
    for variant in ("global_expert", "relation_expert"):
        model, x, edge_index = _make_model(variant)
        original_state = {key: value.clone() for key, value in model.state_dict().items()}
        prepared = diag._prepare_frozen(model, x, edge_index)
        actual = model.analyze(x, edge_index)["fused_z"]
        reproduced = diag._forward_routes(
            model, prepared, prepared["route_text"], prepared["route_visual"]
        )
        assert torch.equal(actual, reproduced)

        reloaded, _, _ = _make_model(variant, seed=17)
        reloaded.load_state_dict(original_state, strict=True)
        reloaded.eval()
        reloaded_out = reloaded(x, edge_index)[0]
        assert torch.equal(actual, reloaded_out)
        for key, before in original_state.items():
            assert torch.equal(model.state_dict()[key], before), key


def test_a2_route_interventions_are_mathematically_correct():
    route = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert torch.equal(diag._route_for_d1(route, "zero"), torch.zeros_like(route))
    assert torch.allclose(diag._route_for_d1(route, "uniform"), torch.full_like(route, 0.25))
    assert torch.equal(diag._route_for_d1(route, "top1"), torch.tensor([0.0, 0.0, 0.0, 1.0]))
    for expert in range(4):
        one_hot = diag._route_for_d1(route, f"single_e{expert}")
        assert one_hot[expert] == 1 and one_hot.sum() == 1
        dropped = diag._route_for_d1(route, f"drop_e{expert}")
        assert dropped[expert] == 0
        assert torch.allclose(dropped.sum(), torch.tensor(1.0))
        expected = route.clone()
        expected[expert] = 0
        expected /= expected.sum()
        assert torch.allclose(dropped, expected)


def test_a3_shrinkage_endpoints_and_selected_route_replacement():
    route = torch.tensor([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.6, 0.2, 0.1],
        [0.1, 0.1, 0.2, 0.6],
        [0.4, 0.2, 0.2, 0.2],
    ])
    global_route = route.mean(dim=0)
    full, _ = diag._shrink_route(route, 1.0)
    collapsed, got_global = diag._shrink_route(route, 0.0)
    assert torch.equal(full, route)
    assert torch.equal(got_global, global_route)
    assert torch.allclose(collapsed, global_route.expand_as(route))
    selected = torch.tensor([1, 3])
    changed = diag._route_to_global(route, global_route, selected)
    assert torch.equal(changed[selected], global_route.expand(selected.numel(), -1))
    untouched = torch.tensor([0, 2])
    assert torch.equal(changed[untouched], route[untouched])


def test_matched_random_group_is_disjoint_equal_size_unique_and_reports_matching():
    target = torch.tensor([0, 1, 2, 3])
    candidates = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
    # The target has two exact 3D strata with three candidates apiece.
    strata = torch.tensor([
        [0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1],
        [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1],
        [1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3],
    ])
    gen = __import__("numpy").random.default_rng(1234)
    matched, meta = diag._sample_stratified_random(target, candidates, strata, gen)
    assert meta["ok"]
    assert matched.numel() == target.numel()
    assert matched.unique().numel() == matched.numel()
    assert not torch.isin(matched, target).any()
    assert set(matched.tolist()).issubset(set(candidates.tolist()))
    assert meta["fallback_counts"]["0d"] == target.numel()
    assert meta["fallback_rate"] == 0.0



def test_a2_modality_scope_changes_only_requested_route():
    text = torch.tensor([0.2, 0.3, 0.1, 0.4])
    visual = torch.tensor([0.4, 0.1, 0.4, 0.1])
    changed_text, changed_visual = diag._replace_modality_routes(text, visual, "text_only", "zero")
    assert torch.equal(changed_text, torch.zeros_like(text))
    assert torch.equal(changed_visual, visual)
    changed_text, changed_visual = diag._replace_modality_routes(text, visual, "visual_only", "top1")
    assert torch.equal(changed_text, text)
    assert torch.equal(changed_visual, torch.tensor([1.0, 0.0, 0.0, 0.0]))



def test_random_candidate_pool_uses_all_edges_but_excludes_target_group():
    target = torch.tensor([1, 4, 7])
    candidates = diag._random_candidate_pool(9, target)
    assert candidates.numel() == 6
    assert not torch.isin(candidates, target).any()
    assert set(candidates.tolist()) == {0, 2, 3, 5, 6, 8}

def test_stratified_matching_relaxation_is_counted_and_balance_diagnostics_are_valid():
    target = torch.tensor([0, 1])
    candidates = torch.tensor([2, 3])
    # No candidates share all three bins, but source/target degree bins match exactly.
    strata = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0]])
    matched, meta = diag._sample_stratified_random(
        target, candidates, strata, __import__("numpy").random.default_rng(9)
    )
    assert meta["ok"] and matched.numel() == target.numel()
    assert meta["fallback_counts"]["0d"] == 0
    assert meta["fallback_counts"]["1d"] == 2
    assert meta["fallback_rate"] == 1.0
    deg = torch.tensor([2.0, 3.0, 2.0, 3.0])
    deviation = torch.tensor([0.1, 0.9, 0.1, 0.9])
    quality = diag._matching_quality(target, matched, strata, deg, deg, deviation, meta)
    assert quality["selected_edges"] == 2
    assert quality["no_overlap"] == 1 and quality["duplicate_edges"] == 0
    assert abs(quality["source_degree_mean_diff"]) < 1e-12
    assert abs(quality["target_degree_mean_diff"]) < 1e-12
    assert quality["source_degree_bin_tvd"] == 0.0
    assert quality["target_degree_bin_tvd"] == 0.0


def test_target_vs_random_summary_keeps_empty_touched_controls_explicit():
    base_row = {
        "is_target": 1, "mean_delta_ce_all_val": 0.2, "mean_delta_ce_touched_val": 0.3,
        "mean_message_change_abs": 0.5,
    }
    controls = [
        {
            "is_target": 0, "mean_delta_ce_all_val": 0.1 + i * 0.001,
            "mean_delta_ce_touched_val": None, "mean_message_change_abs": 0.4,
        }
        for i in range(20)
    ]
    rows = diag._d2_random_summaries("Movies", 42, "text", "personalization_deviation", "D4", [base_row, *controls])
    all_row = next(row for row in rows if row["population"] == "all_validation")
    touched_row = next(row for row in rows if row["population"] == "touched_validation")
    assert all_row["n_repeats"] == 20 and all_row["requested_repeats"] == 20
    assert touched_row["n_repeats"] == 0 and touched_row["n_empty_touched_controls"] == 20
    assert touched_row["requested_repeats"] == 20
    assert touched_row["random_mean_utility"] is None

def test_node_delta_ce_margin_and_fractions_use_registered_signs():
    normal = {
        "ce": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        "margin": torch.tensor([2.0, 2.0, 2.0, 2.0]),
        "predictions": torch.tensor([0, 0, 1, 1]),
    }
    changed = {
        "ce": torch.tensor([1.5, 0.5, 1.0, 1.0]),
        "margin": torch.tensor([1.0, 3.0, 2.0, 2.0]),
        "predictions": torch.tensor([1, 0, 1, 0]),
    }
    utility = diag._node_deltas(normal, changed)
    assert torch.equal(utility["delta_ce"], torch.tensor([0.5, -0.5, 0.0, 0.0]))
    assert torch.equal(utility["margin_utility"], torch.tensor([1.0, -1.0, 0.0, 0.0]))
    summary = diag._summary_values(normal, changed)
    assert summary["harmed_fraction"] == 0.25
    assert summary["improved_fraction"] == 0.25
    assert summary["near_zero_fraction"] == 0.5
    assert summary["prediction_flip_rate"] == 0.5
    assert summary["mean_margin_utility"] == 0.0
    assert summary["mean_absolute_delta_ce"] == 0.25
    assert summary["mean_absolute_margin_utility"] == 0.5


def test_validation_metric_path_never_reads_test_index_or_test_labels():
    class Data:
        num_nodes = 4
        val_idx = torch.tensor([1, 3])
        y = torch.tensor([0, 1, 0, 1])

        @property
        def test_idx(self):
            raise AssertionError("test split must not be accessed")

    head = torch.nn.Identity()
    embedding = torch.tensor([[4.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 3.0]])
    result = diag._validation_arrays(head, embedding, Data(), [0, 1])
    assert result["labels"].tolist() == [1, 1]
    assert result["predictions"].tolist() == [1, 1]
    assert result["acc"] == 1.0


def test_analyzer_helpers_do_not_mutate_checkpoint_parameters_and_are_deterministic():
    model, x, edge_index = _make_model("relation_expert", seed=33)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    prepared1 = diag._prepare_frozen(model, x, edge_index)
    out1 = diag._forward_routes(model, prepared1, prepared1["route_text"], prepared1["route_visual"])
    prepared2 = diag._prepare_frozen(model, x, edge_index)
    out2 = diag._forward_routes(model, prepared2, prepared2["route_text"], prepared2["route_visual"])
    assert torch.equal(out1, out2)
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key

    target = torch.tensor([0, 1, 2])
    candidates = torch.tensor([3, 4, 5, 6])
    strata = torch.zeros(7, 3, dtype=torch.long)
    first, _ = diag._sample_stratified_random(
        target, candidates, strata, __import__("numpy").random.default_rng(77)
    )
    second, _ = diag._sample_stratified_random(
        target, candidates, strata, __import__("numpy").random.default_rng(77)
    )
    assert torch.equal(first, second)


def test_target_node_concentration_stream_summary_and_matching_aggregate(tmp_path):
    path = tmp_path / "target_nodes.csv.gz"
    fields = ["dataset", "seed", "modality", "group_type", "group", "is_target",
              "touched_validation_node", "delta_ce", "margin_utility"]
    rows = [
        {"dataset": "Movies", "seed": 42, "modality": "text", "group_type": "semantic_similarity",
         "group": "Q1", "is_target": 1, "touched_validation_node": 1, "delta_ce": 0.5, "margin_utility": 1.0},
        {"dataset": "Movies", "seed": 42, "modality": "text", "group_type": "semantic_similarity",
         "group": "Q1", "is_target": 1, "touched_validation_node": 0, "delta_ce": -0.25, "margin_utility": -0.5},
        {"dataset": "Movies", "seed": 42, "modality": "text", "group_type": "semantic_similarity",
         "group": "Q1", "is_target": 0, "touched_validation_node": 1, "delta_ce": 99.0, "margin_utility": 99.0},
    ]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = diag._target_concentration_from_node_file(path)
    values = next(iter(result.values()))
    assert values["n_validation_val"] == 2
    assert values["n_touched_val"] == 1
    assert values["mean_absolute_delta_ce_all_val"] == 0.375
    assert values["mean_absolute_delta_ce_touched_val"] == 0.5
    assert values["harmed_fraction_all_val"] == 0.5
    assert values["improved_fraction_all_val"] == 0.5

    match = diag._matching_quality_summary([{
        "selected_edges": 4, "fallback_rate": 0.25, "fallback_2d_count": 1,
        "fallback_1d_count": 0, "fallback_unmatched_strata_count": 0,
        "fallback_3d_count": 3, "no_overlap": 1, "duplicate_edges": 0,
        "source_degree_smd": -0.1, "target_degree_smd": 0.2,
        "personalization_magnitude_smd": 0.3, "source_degree_bin_tvd": 0.0,
        "target_degree_bin_tvd": 0.0, "personalization_bin_tvd": 0.25,
    }])
    assert match["relaxed_edges"] == 1
    assert match["relaxed_edge_fraction"] == 0.25
    assert match["n_relaxed_repeats"] == 1
    assert match["no_overlap_failures"] == 0
