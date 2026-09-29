from __future__ import annotations

import math

from scripts import analyze_rcfa_relation_condition_ablation as analyzer
from scripts import preflight_rcfa_relation_condition_ablation as preflight
from scripts import run_rcfa_relation_condition_ablation as runner


def test_formal_launcher_is_exact_fixed_validation_only_matrix():
    contexts = runner.formal_contexts("cuda:1")
    assert len(contexts) == 15
    assert len({context.run_key for context in contexts}) == 15
    assert {(context.dataset, context.seed) for context in contexts} == {
        (dataset, seed) for dataset in runner.DATASETS for seed in runner.SEEDS
    }
    command = runner.build_command(contexts[0])
    assert "task=nc" in command
    assert "model=crsa_rcfa" in command
    assert "model.use_rcfa=true" in command
    assert "model.use_relation_condition=false" in command
    assert "model.max_order=3" in command
    assert "task.evaluate_test=false" in command
    assert "task.training_mode=full_graph" in command
    assert "task.optimizer=adamw" in command
    assert "task.scheduler=null" in command
    assert not any(arg.startswith("task=test") or arg.startswith("task=lp") for arg in command)


def test_preflight_proves_only_gate_relation_input_changes():
    result = preflight.audit()
    assert result["status"] == "passed"
    assert result["state_dict_keys_identical"]
    assert result["initialization_tensors_identical"]
    assert result["parameter_count_full"] == result["parameter_count_noRSE"]
    assert all(result["stage_i_state_and_effect_checks"].values())


def test_three_way_analyzer_pairs_each_dataset_seed_and_adds_means():
    keys = {(dataset, seed) for dataset in runner.DATASETS for seed in runner.SEEDS}
    crsa = {key: {"val_acc": 0.50, "val_macro_f1": 0.40} for key in keys}
    no_rse = {key: {"val_acc": 0.52, "val_macro_f1": 0.42} for key in keys}
    full = {key: {"val_acc": 0.51, "val_macro_f1": 0.45} for key in keys}
    rows = analyzer.three_way_rows(crsa, no_rse, full)
    assert len(rows) == (15 + 5) * 2
    row = next(item for item in rows if item["dataset"] == "Movies" and item["seed"] == 42
               and item["metric"] == "val_acc")
    assert math.isclose(row["no_rse_minus_crsa"], 0.02)
    assert math.isclose(row["full_minus_no_rse"], -0.01)
    assert math.isclose(row["full_minus_crsa"], 0.01)
    mean = next(item for item in rows if item["dataset"] == "Movies" and item["seed"] == "MEAN"
                and item["metric"] == "val_macro_f1")
    assert math.isclose(mean["no_rse_minus_crsa"], 0.02)
    assert math.isclose(mean["full_minus_no_rse"], 0.03)


def test_analyzer_rejects_test_metrics_and_unpaired_inputs():
    assert analyzer._test_metric_columns(["dataset", "val_acc", "test_acc"]) == ["test_acc"]
    keys = {(dataset, seed) for dataset in runner.DATASETS for seed in runner.SEEDS}
    good = {key: {"val_acc": 0.5, "val_macro_f1": 0.4} for key in keys}
    bad = dict(good)
    bad.pop(next(iter(bad)))
    try:
        analyzer.three_way_rows(good, bad, good)
    except ValueError as exc:
        assert "matching 15" in str(exc)
    else:
        raise AssertionError("unpaired comparison was accepted")
