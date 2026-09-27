from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import analyze_relation_operator_audit as analysis
from scripts import run_relation_operator_audit as launcher


class TinyDiagnosticModel:
    edge_chunk_size = 2
    NUM_EXPERTS = 2

    def __init__(self, variant: str):
        self.operator_variant = variant
        self.experts_text = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.experts_visual = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def _edge_controls(self, modality, h0, target, source):
        if self.operator_variant == "scalar_weight":
            return torch.linspace(0.8, 1.2, target.numel(), device=target.device), {}
        route = torch.tensor([0.35, 0.65], device=target.device).expand(target.numel(), -1)
        return route, {}


def _semantic_fixture():
    data = SimpleNamespace(
        num_nodes=6,
        train_idx=torch.tensor([0, 1, 2, 3]),
        val_idx=torch.tensor([4, 5]),
    )
    target = torch.tensor([0, 1, 2, 3, 0, 2, 4, 4, 4, 4, 5, 5])
    source = torch.tensor([1, 0, 3, 2, 2, 0, 0, 1, 2, 3, 0, 1])
    angle = torch.tensor([0.0, 0.45, 1.3, 2.6, -0.4, 2.2])
    h0 = torch.stack([angle.cos(), angle.sin()], dim=-1)
    ref_model = SimpleNamespace(edge_chunk_size=3)
    semantic = analysis._semantic_bins(ref_model, h0, data, target, source, "text")
    assert semantic is not None
    return data, target, source, h0, semantic


def test_relation_expert_cosine_collapse_is_flagged_and_seen_by_decision_rows():
    model = TinyDiagnosticModel("relation_expert")
    data = SimpleNamespace(train_idx=torch.tensor([0, 1, 2]))
    stage = torch.randn(4, 5)
    report_rows = analysis._pairwise_expert_cosine_rows(
        "relation_expert", "Movies", 42, model,
        {"H0_text": stage, "C_text": [stage, stage + 1, stage + 2]},
        data, "text",
    )
    assert len(report_rows) == 3
    assert all(row["value"] > 0.98 for row in report_rows)
    assert all(row["collapse_flag"] == "EXPERT_COLLAPSE_OBSERVED" for row in report_rows)

    comparisons = {}
    for contrast, _, _ in analysis.COMPARISONS:
        for metric in ("val_acc", "val_macro_f1"):
            comparisons[(contrast, metric)] = {
                "complete_seed_pairs": 0,
                "positive_dataset_means": 0,
                "positive_seed_pairs": 0,
                "nonnegative_dataset_means": 0,
                "aggregate_mean_delta": None,
                **{f"mean_delta_{dataset}": None for dataset in analysis.DATASETS},
            }
    decisions = analysis._decision_rows([], comparisons, [], report_rows)
    collapse = next(row for row in decisions if row["hypothesis"] == "expert_collapse_diagnostic")
    assert collapse["status"] == "EXPERT_COLLAPSE_OBSERVED"
    assert "relation_expert/Movies/seed42/text/hop1" in collapse["collapse_cosine_hops"]
    assert "hop2" in collapse["collapse_cosine_hops"]
    assert "hop3" in collapse["collapse_cosine_hops"]


def test_a1_and_a3_diagnostics_reuse_the_same_plain_a0_edge_groups():
    data, target, source, a0_h0, semantic = _semantic_fixture()
    bins = semantic["bins"].clone()
    counts = torch.bincount(bins, minlength=4).tolist()
    rows_by_variant = {}
    for variant in ("scalar_weight", "relation_expert"):
        model = TinyDiagnosticModel(variant)
        # Deliberately different trained representations must not redefine the groups.
        trained_h0 = -a0_h0 if variant == "scalar_weight" else torch.roll(a0_h0, shifts=1, dims=0)
        c_states = [trained_h0, trained_h0 * 2, trained_h0 * 3]
        diagnostics, used_semantic = analysis._quartile_relation_rows(
            variant, "Movies", 42, model,
            {"H0_text": trained_h0, "C_text": c_states}, data,
            target, source, "text", semantic,
        )
        assert used_semantic is semantic
        assert all(row["similarity_reference"] == "plain_A0_H0" for row in diagnostics)
        rows_by_variant[variant] = {
            int(row["similarity_quartile"]) - 1: int(row["n_edges"])
            for row in diagnostics
        }
    assert rows_by_variant["scalar_weight"] == rows_by_variant["relation_expert"]
    assert rows_by_variant["scalar_weight"] == {q: count for q, count in enumerate(counts) if count}
    a3_rows, _ = analysis._quartile_relation_rows(
        "relation_expert", "Movies", 42, TinyDiagnosticModel("relation_expert"),
        {"H0_text": -a0_h0, "C_text": [-a0_h0, -a0_h0 * 2, -a0_h0 * 3]},
        data, target, source, "text", semantic,
    )
    assert {row["hop"] for row in a3_rows} == {1, 2, 3}
    assert all(
        0.0 <= row["correction_fraction_gt_1"] <= 1.0
        and 0.0 <= row["correction_fraction_gt_2"] <= 1.0
        and 0.0 <= row["message_direction_negative_fraction"] <= 1.0
        for row in a3_rows
    )


def test_pairwise_expert_cosines_cover_hops_one_two_and_three():
    model = TinyDiagnosticModel("global_expert")
    data = SimpleNamespace(train_idx=torch.tensor([0, 1]))
    h0 = torch.randn(3, 4)
    rows = analysis._pairwise_expert_cosine_rows(
        "global_expert", "Movies", 42, model,
        {"H0_visual": h0, "C_visual": [h0, h0 + 1, h0 + 2]},
        data, "visual",
    )
    assert {row["hop"] for row in rows} == {1, 2, 3}
    assert all(row["diagnostic"] == "pairwise_expert_output_cosine" for row in rows)
    assert "hop" in analysis.SPECIALIZATION_FIELDS


def test_matched_random_group_is_disjoint_and_samples_without_replacement():
    positions = torch.arange(12)
    bins = torch.tensor([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3)
    target_group, matched, candidate_count = analysis._sample_matched_random_edge_group(
        positions, bins, 0, torch.Generator(device="cpu").manual_seed(7)
    )
    assert matched is not None
    assert candidate_count == 9
    assert matched.numel() == target_group.numel()
    assert matched.unique().numel() == matched.numel()
    assert not torch.isin(matched, target_group).any()

    target_group, matched, candidate_count = analysis._sample_matched_random_edge_group(
        torch.arange(4), torch.tensor([0, 0, 0, 1]), 0,
        torch.Generator(device="cpu").manual_seed(7),
    )
    assert target_group.numel() == 3
    assert matched is None
    assert candidate_count == 1


def _make_git_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-b", "audit-test-branch"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "audit-test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Audit Test"], cwd=path, check=True)
    (path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True)


def test_formal_launcher_dirty_worktree_fails_before_any_job(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    (repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "ROOT", repo)
    monkeypatch.setattr(launcher.sys, "argv", ["run_relation_operator_audit.py", "--mode", "formal", "--device", "cpu"])
    monkeypatch.setattr(
        launcher, "run_context",
        lambda *args, **kwargs: pytest.fail("formal job started before clean-worktree rejection"),
    )
    with pytest.raises(RuntimeError, match="formal mode requires a clean Git worktree"):
        launcher.main()


def test_formal_context_manifest_has_branch_field_and_sixty_contexts(tmp_path, monkeypatch):
    contexts = launcher.formal_contexts("cpu")
    assert len(contexts) == 60
    assert "git_branch" in launcher.FORMAL_FIELDS
    monkeypatch.setattr(launcher, "OUTPUT", tmp_path / "outputs")
    context = launcher.formal_contexts("cpu")[0]
    row = launcher.run_context(context, "relation_operator_audit", "abc123", dry_run=True)
    assert row["git_branch"] == "relation_operator_audit"
    assert row["git_commit"] == "abc123"
    launcher._append_manifest(context, {**row, "status": "dry_run"})
    with context.manifest.open(newline="", encoding="utf-8") as handle:
        manifest_row = next(csv.DictReader(handle))
    assert manifest_row["git_branch"] == "relation_operator_audit"


def test_smoke_dry_run_does_not_require_clean_worktree(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_git_repo(repo)
    (repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "ROOT", repo)
    monkeypatch.setattr(
        launcher.sys, "argv",
        ["run_relation_operator_audit.py", "--mode", "smoke", "--device", "cpu", "--dry-run"],
    )
    monkeypatch.setattr(
        launcher, "_require_formal_clean_worktree",
        lambda *_: pytest.fail("smoke mode unexpectedly required a clean worktree"),
    )
    called = []
    monkeypatch.setattr(
        launcher, "run_context",
        lambda context, branch, commit, dry_run=False: called.append((context, branch, commit, dry_run))
        or {"status": "dry_run"},
    )
    launcher.main()
    assert len(called) == 1
    assert called[0][0].mode == "smoke"
    assert called[0][3] is True
