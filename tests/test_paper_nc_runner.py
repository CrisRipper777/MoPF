from __future__ import annotations

from pathlib import Path

from scripts.run_paper_nc_final import (
    DATASETS,
    MODELS,
    SEEDS,
    _base_overrides,
    _build_jobs,
)


def test_final_runner_is_nc_only_and_has_the_registered_matrix() -> None:
    jobs = _build_jobs(Path("/tmp/paper-nc-test"), ["cuda:0", "cuda:1"])
    assert len(jobs) == 5 * 10 * 3
    assert {job.dataset for job in jobs} == set(DATASETS)
    assert {job.model for job in jobs} == set(MODELS)
    assert {job.seed for job in jobs} == set(SEEDS)
    assert all("task=nc" in " ".join(_base_overrides(job.dataset, job.model, job.seed, job.device, job.run_dir)) for job in jobs)
    assert all("task=lp" not in " ".join(_base_overrides(job.dataset, job.model, job.seed, job.device, job.run_dir)) for job in jobs)


def test_final_runner_uses_one_frozen_split_across_training_seeds() -> None:
    movies = [
        _base_overrides("Movies", "mlp", seed, "cuda:0", Path("/tmp/run"))
        for seed in SEEDS
    ]
    split_values = [next(value for value in overrides if value.startswith("dataset.nc_split_path=")) for overrides in movies]
    assert len(set(split_values)) == 1
    assert "Movies_nc_seed42_train0.6_val0.2.pt" in split_values[0]


def test_mopf_uses_preregistered_dataset_order() -> None:
    grocery = _base_overrides("Grocery", "mopf", 42, "cuda:0", Path("/tmp/run"))
    movies = _base_overrides("Movies", "mopf", 42, "cuda:0", Path("/tmp/run"))
    assert "model.max_order=2" in grocery
    assert "model.num_layers=2" in grocery
    assert "model.max_order=3" in movies
