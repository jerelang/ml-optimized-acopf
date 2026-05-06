from __future__ import annotations

from pathlib import Path

import polars as pl

from ml_acopf.cases.generate import generate_cases
from ml_acopf.cli import ensure_julia_ready
from ml_acopf.config import Config, DataConfig, PerturbConfig, SolverConfig


def test_generate_cases(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = Config(
        data=DataConfig(
            name="test_dataset",
            network_name="case14",
            n_cases=1,
            seed=123,
            max_attempts_multiplier=100,
        ),
        perturb=PerturbConfig(
            load_scale_min=0.95,
            load_scale_max=1.05,
        ),
        solver=SolverConfig(pm_tol=1e-7),
    )
    ensure_julia_ready()
    summary = generate_cases(cfg)

    dataset_dir = Path("data/test_dataset/baseline")
    assert dataset_dir.is_dir()

    cases_path = dataset_dir / "cases.parquet"
    load_inputs_path = dataset_dir / "load_inputs.parquet"
    bus_targets_path = dataset_dir / "bus_targets.parquet"
    dispatch_targets_path = dataset_dir / "dispatch_targets.parquet"
    buses_static_path = dataset_dir / "buses_static.parquet"
    edges_static_path = dataset_dir / "edges_static.parquet"

    assert cases_path.exists()
    assert load_inputs_path.exists()
    assert bus_targets_path.exists()
    assert dispatch_targets_path.exists()
    assert buses_static_path.exists()
    assert edges_static_path.exists()

    cases = pl.read_parquet(cases_path)
    load_inputs = pl.read_parquet(load_inputs_path)
    bus_targets = pl.read_parquet(bus_targets_path)
    dispatch_targets = pl.read_parquet(dispatch_targets_path)

    assert summary["generated_cases"] == 1
    assert cases.height == 1

    case_id = cases["case_id"][0]
    assert case_id in load_inputs["case_id"].to_list()
    assert case_id in bus_targets["case_id"].to_list()
    assert case_id in dispatch_targets["case_id"].to_list()
