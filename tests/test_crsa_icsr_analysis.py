from __future__ import annotations

from scripts import analyze_crsa_icsr_nc as analyzer
from scripts import run_crsa_icsr_nc as runner


def test_fixed_formal_and_smoke_matrices_and_protocol_command():
    formal = runner.formal_contexts("cuda:1")
    smoke = runner.smoke_contexts("cuda:1")
    assert len(formal) == 15
    assert len({row.run_key for row in formal}) == 15
    assert {(row.dataset, row.seed) for row in formal} == analyzer.EXPECTED_PAIRS
    assert [(row.dataset, row.seed, row.epochs) for row in smoke] == [
        ("Movies", 42, 2), ("ele-fashion", 42, 1), ("Reddit-S", 42, 1)
    ]
    command = runner.build_command(formal[0])
    assert "model=crsa_icsr" in command
    assert "model.use_icsr=true" in command
    assert "task.evaluate_test=false" in command
    assert "task.loss.aux_weight=0.0" in command
    assert "task.epochs=300" in command


def test_history_provenance_requires_exact_complete_paired_matrix():
    rows = [
        {"dataset": dataset, "seed": str(seed), "status": "complete",
         "test_evaluated": "false", "source_commit": "expected",
         "val_acc": "0.5", "val_macro_f1": "0.4"}
        for dataset in runner.DATASETS for seed in runner.SEEDS
    ]
    audit, keyed = analyzer.validate_source_rows("test", rows, "expected")
    assert audit["status"] == "valid"
    assert len(keyed) == 15
    rows[0]["source_commit"] = "wrong"
    bad_audit, _ = analyzer.validate_source_rows("test", rows, "expected")
    assert bad_audit["status"] == "invalid"


def test_four_way_paired_values_and_dataset_mean_rows():
    histories = {name: {} for name in ("crsa", "rcfa_no_rse", "rcfa_full")}
    v3 = {}
    for dataset in runner.DATASETS:
        for seed in runner.SEEDS:
            key = (dataset, seed)
            for name, value in (("crsa", 0.50), ("rcfa_no_rse", 0.51), ("rcfa_full", 0.49)):
                histories[name][key] = {"val_acc": value, "val_macro_f1": value - 0.1}
            v3[key] = {"val_acc": 0.52, "val_macro_f1": 0.42}
    rows = analyzer.four_way_rows(histories, v3)
    assert len(rows) == 5 * 3 * 2 + 5 * 2
    first = next(r for r in rows if r["dataset"] == "Movies" and r["seed"] == 42 and r["metric"] == "Accuracy")
    assert abs(first["icsr_minus_crsa"] - 0.02) < 1e-12
    assert abs(first["icsr_minus_rcfa_full"] - 0.03) < 1e-12
    mean = next(r for r in rows if r["dataset"] == "Movies" and r["row_type"] == "dataset_mean" and r["metric"] == "Accuracy")
    assert abs(mean["icsr_minus_crsa"] - 0.02) < 1e-12
