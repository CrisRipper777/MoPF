from __future__ import annotations

import csv
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate_robustness import clean_checkpoint_check
from scripts.prepare_robustness_perturbations import (
    balanced_category_counts,
    build_balanced_nested_order,
    budget_for_rho,
    conflict_category_masks,
    edge_index_from_canonical,
    maximum_balanced_capacity,
    sample_true_nonedges,
    sha256_edges,
    sha256_file,
    validate_shared_plan_artifacts,
)


def test_random_noise_edges_are_true_canonical_nonedges_without_duplicates() -> None:
    clean = np.asarray([[0, 1], [1, 3], [2, 4]], dtype=np.int64)
    sampled = sample_true_nonedges(6, clean, 12, np.random.default_rng(101))
    assert sampled.shape == (12, 2)
    assert np.all(sampled[:, 0] < sampled[:, 1])
    assert np.all((sampled >= 0) & (sampled < 6))
    assert len(np.unique(sampled, axis=0)) == len(sampled)
    assert not set(map(tuple, sampled.tolist())).intersection(map(tuple, clean.tolist()))


def test_noise_prefixes_are_nested_at_frozen_budgets() -> None:
    num_edges = 101
    ordered = sample_true_nonedges(30, np.asarray([[0, 1]], dtype=np.int64), 41, np.random.default_rng(102))
    counts = [budget_for_rho(rho, num_edges) for rho in (0.05, 0.10, 0.20, 0.30, 0.40)]
    prefixes = [set(map(tuple, ordered[:count].tolist())) for count in counts]
    assert all(left.issubset(right) for left, right in zip(prefixes, prefixes[1:]))
    assert counts == [5, 10, 20, 30, 40]


def test_same_perturbation_artifact_is_reused_across_models() -> None:
    common = {
        "dataset": "Movies",
        "perturbation_type": "random_structural_noise",
        "perturbation_seed": 1001,
        "rho": 0.1,
        "edge_artifact": "edges/movies_010.npy",
        "edge_sha256": "abc",
    }
    rows = [
        {**common, "model": model, "model_seed": model_seed}
        for model in ("mopf", "wo_relation_calibration", "dip")
        for model_seed in (42, 43, 44)
    ]
    validate_shared_plan_artifacts(rows)
    rows[-1]["edge_sha256"] = "different"
    with pytest.raises(ValueError, match="same perturbation artifact"):
        validate_shared_plan_artifacts(rows)


def test_conflict_categories_obey_frozen_average_rank_thresholds() -> None:
    rank_text = np.asarray([0.80, 0.10, 0.90, 0.74, 0.25])
    rank_visual = np.asarray([0.20, 0.90, 0.80, 0.10, 0.80])
    c_text, c_visual = conflict_category_masks(rank_text, rank_visual)
    assert np.array_equal(np.flatnonzero(c_text), np.asarray([0]))
    assert np.array_equal(np.flatnonzero(c_visual), np.asarray([1, 4]))
    assert not np.any(c_text & c_visual)
    assert np.all(rank_text[c_text] >= 0.75)
    assert np.all(rank_visual[c_text] <= 0.25)
    assert np.all(rank_visual[c_visual] >= 0.75)
    assert np.all(rank_text[c_visual] <= 0.25)


def test_conflict_order_and_all_prefixes_remain_balanced_and_nested() -> None:
    text_edges = np.asarray([[0, 2], [1, 5], [2, 7], [4, 9]], dtype=np.int64)
    visual_edges = np.asarray([[0, 3], [1, 6], [2, 8]], dtype=np.int64)
    ordered, categories = build_balanced_nested_order(text_edges, visual_edges, seed=88)
    assert len(ordered) == maximum_balanced_capacity(4, 3) == 7
    assert np.all(ordered[:, 0] < ordered[:, 1])
    assert len(np.unique(ordered, axis=0)) == len(ordered)
    assert set(categories.tolist()).issubset({1, 2})
    for prefix_size in range(len(ordered) + 1):
        prefix = categories[:prefix_size]
        n_text = int(np.sum(prefix == 1))
        n_visual = int(np.sum(prefix == 2))
        assert abs(n_text - n_visual) <= 1
        assert n_text + n_visual == prefix_size
        assert n_text <= 4 and n_visual <= 3
    assert set(map(tuple, ordered[:4].tolist())).issubset(set(map(tuple, ordered[:7].tolist())))


def test_edge_generation_apis_do_not_accept_or_read_labels_or_model_outputs() -> None:
    generators = (sample_true_nonedges, conflict_category_masks)
    forbidden_parameters = {"labels", "y", "predictions", "logits", "embeddings", "conductance", "model"}
    for function in generators:
        signature = inspect.signature(function)
        assert not forbidden_parameters.intersection(signature.parameters)
        source = inspect.getsource(function).lower()
        assert "data.y" not in source
        assert "logit" not in source
        assert "conductance" not in source


def test_canonical_pairs_restore_bidirected_loader_convention_and_keep_clean_prefix() -> None:
    clean = torch.tensor([[0, 2, 1, 3, 4, 1], [2, 0, 3, 1, 1, 4]], dtype=torch.long)
    injected = np.asarray([[0, 3], [2, 4]], dtype=np.int64)
    result = edge_index_from_canonical(clean, injected)
    assert torch.equal(result[:, : clean.size(1)], clean)
    assert result.size(1) == clean.size(1) + 2 * len(injected)
    oriented = set(map(tuple, result.t().tolist()))
    assert {(0, 3), (3, 0), (2, 4), (4, 2)}.issubset(oriented)


def test_rho_zero_edge_index_is_bitwise_equal_to_clean() -> None:
    clean = torch.tensor([[3, 0, 2, 1], [0, 3, 1, 2]], dtype=torch.long)
    rho_zero = edge_index_from_canonical(clean, np.empty((0, 2), dtype=np.int64))
    assert torch.equal(rho_zero, clean)


def test_balanced_capacity_and_odd_budget_class_allocation() -> None:
    assert maximum_balanced_capacity(20, 20) == 40
    assert maximum_balanced_capacity(20, 19) == 39
    assert maximum_balanced_capacity(20, 0) == 1
    assert balanced_category_counts(5, 20, 19) == (3, 2)
    assert balanced_category_counts(3, 1, 20) == (1, 2)


def test_clean_full_movies_seed42_checkpoint_matches_saved_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    audit_path = root / "outputs/robustness_analysis/checkpoint_audit.csv"
    if not audit_path.is_file():
        pytest.skip("offline checkpoint audit has not been generated")
    with audit_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(
        r for r in rows
        if r["dataset"] == "Movies" and r["model"] == "mopf" and r["model_seed"] == "42"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    measured = clean_checkpoint_check(
        Path(row["checkpoint_path"]),
        Path(row["config_path"]),
        float(row["clean_test_accuracy"]),
        float(row["clean_test_macro_f1"]),
        device,
    )
    assert np.isclose(measured["accuracy"], float(row["clean_test_accuracy"]), rtol=0.0, atol=1e-6)
    assert np.isclose(measured["macro_f1"], float(row["clean_test_macro_f1"]), rtol=0.0, atol=1e-6)


def test_prepared_perturbation_artifacts_are_hashed_nested_and_reused() -> None:
    root = Path(__file__).resolve().parents[1] / "outputs/robustness_analysis"
    random_path = root / "random_noise/dry_run_plan.csv"
    conflict_path = root / "semantic_conflict/dry_run_plan.csv"
    if not random_path.is_file() or not conflict_path.is_file():
        pytest.skip("robustness perturbations have not been prepared")
    with random_path.open("r", newline="", encoding="utf-8") as handle:
        random_rows = list(csv.DictReader(handle))
    with conflict_path.open("r", newline="", encoding="utf-8") as handle:
        conflict_rows = list(csv.DictReader(handle))
    validate_shared_plan_artifacts(random_rows)
    validate_shared_plan_artifacts(conflict_rows)
    assert len(random_rows) == 432
    assert len(conflict_rows) == 270

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in random_rows:
        groups.setdefault((row["dataset"], row["perturbation_seed"]), []).append(row)
    for (_, _), rows in groups.items():
        representative = rows[0]
        output = np.load(root / representative["edge_artifact"].replace("_rho0.00", "_maxrho040"), allow_pickle=False)
        by_rho: dict[float, np.ndarray] = {}
        for row in rows:
            path = root / row["edge_artifact"]
            assert sha256_file(path) == row["edge_artifact_sha256"]
            edges = np.load(path, allow_pickle=False)
            assert sha256_edges(edges) == row["edge_sha256"]
            assert len(edges) == int(row["injected_edge_count"])
            assert np.all(edges[:, 0] < edges[:, 1])
            by_rho[float(row["rho"])] = edges
        ordered = output
        for rho, edges in by_rho.items():
            assert np.array_equal(edges, ordered[: len(edges)])

    conflict_unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in conflict_rows:
        conflict_unique[(row["dataset"], row["perturbation_seed"], row["rho"])] = row
    conflict_groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for (dataset, perturb_seed, _), row in conflict_unique.items():
        conflict_groups.setdefault((dataset, perturb_seed), []).append(row)
    for (dataset, _), rows in conflict_groups.items():
        first = rows[0]
        candidate_path = root / first["edge_artifact"].replace("perturbation_edges", "candidate_pools").replace(
            f"_rho{float(first['rho']):.2f}.npz", "_pool1000000.npz"
        )
        with np.load(candidate_path, allow_pickle=False) as archive:
            candidates = archive["candidate_edges"]
            rank_text = archive["rank_text"]
            rank_visual = archive["rank_visual"]
            candidate_category = archive["category"]
        num_nodes = 16672 if dataset == "Movies" else 17074
        candidate_keys = candidates[:, 0] * num_nodes + candidates[:, 1]
        order = np.argsort(candidate_keys)
        sorted_keys = candidate_keys[order]
        for row in rows:
            path = root / row["edge_artifact"]
            assert sha256_file(path) == row["edge_artifact_sha256"]
            with np.load(path, allow_pickle=False) as archive:
                edges = archive["edges"]
                category = archive["category"]
            assert sha256_edges(edges) == row["edge_sha256"]
            assert len(edges) == int(row["injected_edge_count"])
            assert abs(int(np.sum(category == 1)) - int(np.sum(category == 2))) <= 1
            injected_keys = edges[:, 0] * num_nodes + edges[:, 1]
            positions = np.searchsorted(sorted_keys, injected_keys)
            assert np.all(positions < len(candidate_keys))
            assert np.array_equal(sorted_keys[positions], injected_keys)
            indices = order[positions]
            assert np.array_equal(candidate_category[indices], category)
            text_mask = category == 1
            visual_mask = category == 2
            assert np.all(rank_text[indices[text_mask]] >= 0.75)
            assert np.all(rank_visual[indices[text_mask]] <= 0.25)
            assert np.all(rank_visual[indices[visual_mask]] >= 0.75)
            assert np.all(rank_text[indices[visual_mask]] <= 0.25)
