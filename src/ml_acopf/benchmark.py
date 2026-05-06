from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandapower as pp
import polars as pl
from tqdm import tqdm

from .cases.generate import (
    apply_bus_load_profile,
    export_load_inputs,
    sample_bus_load_profile,
)
from .cases.networks import export_static_tables, network_template
from .config import SolverConfig
from .solver.io import (
    SolveStats,
    WarmStartPayload,
    export_bus_warmstart,
    export_device_metadata,
    export_device_warmstart,
)
from .solver.solver import solve_ac_opf
from .utils import make_case_id, write_parquet

if TYPE_CHECKING:
    from torch_geometric.data import Data

BASELINE_BENCHMARK_FIELDS: dict[str, pl.DataType] = {
    "case_id": pl.String(),
    "network_name": pl.String(),
    "sample_index": pl.Int64(),
    "seed": pl.Int64(),
    "total_p_mw": pl.Float64(),
    "total_q_mvar": pl.Float64(),
    "success": pl.Boolean(),
    "wall_time_s": pl.Float64(),
    "solver_time_s": pl.Float64(),
    "objective": pl.Float64(),
    "termination_status": pl.String(),
    "iterations": pl.Int64(),
    "error": pl.String(),
}

BASELINE_BENCHMARK_SCHEMA = pl.Schema(BASELINE_BENCHMARK_FIELDS)
WARMSTART_BENCHMARK_SCHEMA = pl.Schema(
    {
        **BASELINE_BENCHMARK_FIELDS,
        "method": pl.String(),
        "prep_time_s": pl.Float64(),
        "total_time_s": pl.Float64(),
    }
)


class WarmStartPredictor(Protocol):
    name: str

    def predict(self, graph: Data, case_id: str) -> WarmStartPayload: ...


def run_benchmark(
    *,
    network_names: Sequence[str],
    n_cases: int,
    seed: int,
    load_scale_min: float,
    load_scale_max: float,
    local_load_noise_scale: float = 0.05,
    reactive_noise_scale: float = 0.05,
    solver: SolverConfig,
    include_flat: bool = True,
    include_pf: bool = True,
    predictor: WarmStartPredictor | None = None,
    max_voltage_angle_deg: float = 60.0,
    out: Path | None = None,
) -> pl.DataFrame:
    unique_network_names = tuple(dict.fromkeys(network_names))
    if not unique_network_names:
        raise ValueError("network_names must not be empty.")
    if not include_flat and not include_pf and predictor is None:
        raise ValueError("At least one warm-start method must be enabled.")

    rows: list[dict[str, object]] = []

    for network_name in unique_network_names:
        rng = np.random.default_rng(seed)
        base_net = network_template(network_name)
        buses_static, edges_static = export_static_tables(base_net)
        device_metadata = export_device_metadata(base_net)

        for sample_index in tqdm(range(n_cases), desc="Number of fresh cases"):
            profile = sample_bus_load_profile(
                base_net,
                rng,
                total_scale_min=load_scale_min,
                total_scale_max=load_scale_max,
                local_load_noise_scale=local_load_noise_scale,
                reactive_noise_scale=reactive_noise_scale,
            )
            perturbed_net = apply_bus_load_profile(base_net, profile)
            case_id = make_case_id(network_name, seed, sample_index)

            if include_flat:
                stats = solve_ac_opf(copy.deepcopy(perturbed_net), solver)
                rows.append(
                    _warmstart_row(
                        case_id=case_id,
                        network_name=network_name,
                        sample_index=sample_index,
                        seed=seed,
                        total_p_mw=profile.total_p_mw,
                        total_q_mvar=profile.total_q_mvar,
                        method="flat",
                        prep_time_s=0.0,
                        stats=stats,
                    )
                )

            if include_pf:
                net_pf = copy.deepcopy(perturbed_net)
                started_at = time.perf_counter()
                try:
                    pp.runpp(
                        net_pf,
                        calculate_voltage_angles=solver.calculate_voltage_angles,
                        check_connectivity=solver.check_connectivity,
                        init="auto",
                    )
                    warmstart = WarmStartPayload(
                        bus=export_bus_warmstart(net_pf, case_id),
                        device=export_device_warmstart(net_pf, case_id),
                    )
                    prep_time_s = time.perf_counter() - started_at
                    stats = solve_ac_opf(net_pf, solver, warmstart=warmstart)
                except Exception as error:
                    prep_time_s = time.perf_counter() - started_at
                    stats = _failed_stats(
                        prefix="PF warm-start failed",
                        error=error,
                    )

                rows.append(
                    _warmstart_row(
                        case_id=case_id,
                        network_name=network_name,
                        sample_index=sample_index,
                        seed=seed,
                        total_p_mw=profile.total_p_mw,
                        total_q_mvar=profile.total_q_mvar,
                        method="pf",
                        prep_time_s=prep_time_s,
                        stats=stats,
                    )
                )

            if predictor is not None:
                from .learning.dataset import build_graph_data_from_case

                started_at = time.perf_counter()
                try:
                    load_inputs = export_load_inputs(perturbed_net, case_id, profile)
                    graph = build_graph_data_from_case(
                        buses_static=buses_static,
                        edges_static=edges_static,
                        load_inputs=load_inputs,
                        device_metadata=device_metadata,
                        max_voltage_angle_deg=max_voltage_angle_deg,
                    )
                    warmstart = predictor.predict(graph, case_id)
                    prep_time_s = time.perf_counter() - started_at
                    stats = solve_ac_opf(copy.deepcopy(perturbed_net), solver, warmstart=warmstart)
                except Exception as error:
                    prep_time_s = time.perf_counter() - started_at
                    stats = _failed_stats(
                        prefix=f"{predictor.name} warm-start failed",
                        error=error,
                    )

                rows.append(
                    _warmstart_row(
                        case_id=case_id,
                        network_name=network_name,
                        sample_index=sample_index,
                        seed=seed,
                        total_p_mw=profile.total_p_mw,
                        total_q_mvar=profile.total_q_mvar,
                        method=predictor.name,
                        prep_time_s=prep_time_s,
                        stats=stats,
                    )
                )

    frame = pl.DataFrame(rows, schema=WARMSTART_BENCHMARK_SCHEMA)
    if out is not None:
        write_parquet(frame, out)
    return frame


def summarize_benchmark(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame

    group_by = ["network_name"]
    if "method" in frame.columns:
        group_by.append("method")

    aggregations = [
        pl.len().alias("n_cases"),
        pl.col("success").cast(pl.Float64).mean().alias("success_rate"),
        pl.col("wall_time_s").mean().alias("wall_time_s_mean"),
        pl.col("wall_time_s").median().alias("wall_time_s_median"),
        pl.col("solver_time_s").mean().alias("solver_time_s_mean"),
        pl.col("objective").mean().alias("objective_mean"),
    ]

    if "prep_time_s" in frame.columns:
        aggregations.extend(
            [
                pl.col("prep_time_s").mean().alias("prep_time_s_mean"),
                pl.col("total_time_s").mean().alias("total_time_s_mean"),
            ]
        )

    return frame.group_by(group_by).agg(*aggregations).sort(group_by)


def _warmstart_row(
    *,
    case_id: str,
    network_name: str,
    sample_index: int,
    seed: int,
    total_p_mw: float,
    total_q_mvar: float,
    method: str,
    prep_time_s: float,
    stats: SolveStats,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "network_name": network_name,
        "sample_index": sample_index,
        "seed": seed,
        "total_p_mw": total_p_mw,
        "total_q_mvar": total_q_mvar,
        "method": method,
        "prep_time_s": prep_time_s,
        "success": stats.success,
        "wall_time_s": stats.wall_time_s,
        "solver_time_s": stats.solver_time_s,
        "total_time_s": prep_time_s + stats.wall_time_s,
        "objective": stats.objective,
        "termination_status": stats.termination_status,
        "iterations": stats.iterations,
        "error": stats.error,
    }


def _failed_stats(*, prefix: str, error: Exception) -> SolveStats:
    return SolveStats(
        success=False,
        wall_time_s=0.0,
        error=f"{prefix}: {type(error).__name__}: {error}",
    )
