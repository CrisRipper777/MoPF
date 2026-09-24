from __future__ import annotations

import json
import sys

import pytest

from scripts.run_ssi_mag_final_nc_ablation import (
    ABLATIONS,
    DATASETS,
    SEEDS,
    _complete,
    _jobs,
    _prepare_formal_lock,
    _prepare_dry_run,
    _provenance,
)


def test_dry_run_plan_is_120_and_does_not_create_formal_lock(tmp_path):
    class Args:
        dry_run = True
        datasets = list(DATASETS)
        seeds = list(SEEDS)
        ablations = list(ABLATIONS)
        device = "cuda:0"
    args = Args()
    jobs = _jobs(tmp_path, args.datasets, args.seeds, args.ablations, args.device)
    provenance = _provenance("abc", args.datasets, args.seeds, args.ablations, dry_run=True)
    plan = _prepare_dry_run(tmp_path, args, jobs, provenance)
    assert len(jobs) == 120
    assert plan.name == "provenance.plan.json"
    assert plan.is_file()
    assert not (tmp_path / "provenance.lock.json").exists()


def test_formal_lock_same_payload_resumes_and_changed_payload_rejects(tmp_path, monkeypatch):
    class Args:
        dry_run = False
        datasets = list(DATASETS)
        seeds = list(SEEDS)
        ablations = list(ABLATIONS)
        device = "cuda:0"
    args = Args()
    jobs = _jobs(tmp_path, args.datasets, args.seeds, args.ablations, args.device)
    provenance = _provenance("abc", args.datasets, args.seeds, args.ablations, dry_run=False)
    monkeypatch.setattr("scripts.run_ssi_mag_final_nc_ablation._assert_clean_synced", lambda: None)
    lock = _prepare_formal_lock(tmp_path, args, jobs, provenance)
    assert json.loads(lock.read_text()) == provenance
    assert _prepare_formal_lock(tmp_path, args, jobs, provenance) == lock
    changed = dict(provenance, git_commit="different")
    with pytest.raises(RuntimeError, match="provenance lock mismatch"):
        _prepare_formal_lock(tmp_path, args, jobs, changed)


def test_completed_formal_root_cannot_be_overwritten_by_new_provenance(tmp_path, monkeypatch):
    class Args:
        dry_run = False
        datasets = [DATASETS[0]]
        seeds = [SEEDS[0]]
        ablations = ["full"]
        device = "cuda:0"
    args = Args()
    jobs = _jobs(tmp_path, args.datasets, args.seeds, args.ablations, args.device)
    provenance = _provenance("abc", args.datasets, args.seeds, args.ablations, dry_run=False)
    monkeypatch.setattr("scripts.run_ssi_mag_final_nc_ablation._assert_clean_synced", lambda: None)
    lock = _prepare_formal_lock(tmp_path, args, jobs, provenance)
    run = tmp_path / "Movies" / "full" / "seed42"
    run.mkdir(parents=True)
    for name in ("best.pt", "metrics.json", "results.json", "resolved_config.json", "complete.marker"):
        (run / name).write_text("{}")
    with pytest.raises(RuntimeError, match="provenance lock mismatch"):
        _prepare_formal_lock(tmp_path, args, jobs, dict(provenance, git_commit="new"))
