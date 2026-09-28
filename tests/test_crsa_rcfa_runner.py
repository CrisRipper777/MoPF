from __future__ import annotations

import csv

import scripts.analyze_crsa_rcfa_nc as analyzer
import scripts.run_crsa_rcfa_nc as runner


def test_formal_matrix_is_exactly_fifteen_full_v2_validation_runs():
    contexts = runner.formal_contexts("cuda:1")
    assert len(contexts) == 15
    assert len({context.run_key for context in contexts}) == 15
    assert {context.dataset for context in contexts} == set(runner.DATASETS)
    assert {context.seed for context in contexts} == set(runner.SEEDS)
    assert {context.variant for context in contexts} == {"full"}
    command = runner.build_command(contexts[0])
    assert "model=crsa_rcfa" in command
    assert "model.max_order=3" in command
    assert "task.evaluate_test=false" in command
    assert "task.eval_modality_masks.enabled=false" in command
    assert "task.scheduler=null" in command
    assert "model.use_rcfa=true" in command
    assert "model.use_relation_condition=true" in command
    assert contexts[0].run_dir != contexts[1].run_dir
    assert contexts[0].checkpoint != contexts[1].checkpoint


def test_smoke_suite_contains_reference_full_ablation_and_two_large_datasets():
    contexts = runner.smoke_suite("cuda:1")
    assert len(contexts) == 5
    assert [(c.dataset, c.variant, c.seed) for c in contexts] == [
        ("Movies", "crsa_ref", 42),
        ("Movies", "full", 42),
        ("Movies", "wo_relation_condition", 42),
        ("ele-fashion", "full", 42),
        ("Reddit-S", "full", 42),
    ]
    assert len({c.run_dir for c in contexts}) == 5
    assert len({c.checkpoint for c in contexts}) == 5


def test_historical_crsa_reference_has_complete_validation_only_pairs():
    reference, reason = analyzer._historical_reference()
    assert len(reference) == 15, reason
    assert set(reference) == {
        (dataset, seed) for dataset in runner.DATASETS for seed in runner.SEEDS
    }


def test_paired_analyzer_is_descriptive_and_emits_dataset_means(monkeypatch, tmp_path):
    monkeypatch.setattr(analyzer, "RESULTS", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "preflight_report.md").write_text(
        "legacy_crsa_parity_max_abs_error: 1.0e-7\n", encoding="utf-8"
    )
    rows = [
        {
            "dataset": dataset, "seed": seed, "status": "complete",
            "val_acc": 0.6 + seed / 1000, "val_macro_f1": 0.5 + seed / 1000,
            "source_commit": "formal-source-sha",
        }
        for dataset in runner.DATASETS for seed in runner.SEEDS
    ]
    reference = {
        (dataset, seed): {"val_acc": 0.5, "val_macro_f1": 0.4}
        for dataset in runner.DATASETS for seed in runner.SEEDS
    }
    monkeypatch.setattr(analyzer, "_historical_reference", lambda: (reference, "verified"))
    paired, status = analyzer._paired_rows(rows, reference)
    assert len(paired) == 40
    assert "passed" in status
    assert sum(row["seed"] == "MEAN" for row in paired) == 10
    assert all("status" not in row and "significance" not in row for row in paired)
