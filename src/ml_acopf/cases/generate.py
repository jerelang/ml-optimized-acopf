from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import polars as pl
from pandapower.auxiliary import pandapowerNet

from ..config import Config
from ..solver.io import export_bus_results, export_dispatch_results
from ..solver.solver import solve_ac_opf
from ..utils import make_case_id, write_parquet
from .networks import build_network, export_static_tables

# Load Perturbation

LOAD_SCHEMA = pl.Schema(
    {
        "case_id": pl.String(),
        "load_index": pl.Int64(),
        "bus": pl.Int64(),
        "p_mw": pl.Float64(),
        "q_mvar": pl.Float64(),
        "total_p_mw": pl.Float64(),
        "total_q_mvar": pl.Float64(),
    }
)


@dataclass(frozen=True, slots=True)
class BusLoadProfile:
    bus_index: np.ndarray
    p_mw: np.ndarray
    q_mvar: np.ndarray
    total_p_mw: float
    total_q_mvar: float


def sample_bus_load_profile(
    net: pandapowerNet,
    rng: np.random.Generator,
    *,
    total_scale_min: float,
    total_scale_max: float,
    dirichlet_concentration: float = 25.0,
    reactive_noise_scale: float = 0.05,
) -> BusLoadProfile:
    if len(net.load) == 0:
        raise ValueError("No loads to sample from.")

    grouped = (
        pl.from_pandas(net.load.reset_index(names="load_index"))
        .group_by("bus")
        .agg(
            pl.col("p_mw").sum().alias("base_p_mw"),
            pl.col("q_mvar").sum().alias("base_q_mvar"),
        )
        .sort("bus")
    )

    bus_index = grouped["bus"].to_numpy().astype(np.int64)
    base_p = grouped["base_p_mw"].to_numpy().astype(np.float64)
    base_q = grouped["base_q_mvar"].to_numpy().astype(np.float64)

    total_base_p = float(base_p.sum())
    if total_base_p <= 0.0:
        raise ValueError("Total base active load must be positive.")

    total_p_mw = float(rng.uniform(total_scale_min, total_scale_max) * total_base_p)

    weights = np.clip(base_p / total_base_p, 1e-6, None)
    concentration = dirichlet_concentration * weights
    shares = rng.dirichlet(concentration)
    p_mw = total_p_mw * shares

    q_over_p = np.divide(
        base_q,
        np.maximum(base_p, 1e-9),
        out=np.zeros_like(base_q),
        where=base_p > 0.0,
    )
    reactive_noise = rng.normal(loc=1.0, scale=reactive_noise_scale, size=q_over_p.shape)
    reactive_noise = np.clip(reactive_noise, 0.5, 1.5)
    q_mvar = p_mw * q_over_p * reactive_noise

    return BusLoadProfile(
        bus_index=bus_index,
        p_mw=p_mw,
        q_mvar=q_mvar,
        total_p_mw=float(p_mw.sum()),
        total_q_mvar=float(q_mvar.sum()),
    )


def apply_bus_load_profile(
    net: pandapowerNet,
    profile: BusLoadProfile,
) -> pandapowerNet:
    net_copy = copy.deepcopy(net)

    if len(net_copy.load) == 0 or profile.bus_index.size == 0:
        return net_copy

    profile_p = {int(bus): float(p) for bus, p in zip(profile.bus_index, profile.p_mw, strict=True)}
    profile_q = {
        int(bus): float(q) for bus, q in zip(profile.bus_index, profile.q_mvar, strict=True)
    }

    for bus in profile.bus_index.tolist():
        mask = net_copy.load["bus"] == int(bus)
        load_indices = net_copy.load.index[mask]

        if len(load_indices) == 0:
            continue

        base_p = net_copy.load.loc[load_indices, "p_mw"].to_numpy(dtype=float)
        base_q = (
            net_copy.load.loc[load_indices, "q_mvar"].to_numpy(dtype=float)
            if "q_mvar" in net_copy.load.columns
            else np.zeros(len(load_indices), dtype=float)
        )

        if base_p.sum() > 0.0:
            p_share = base_p / base_p.sum()
        else:
            p_share = np.full(len(load_indices), 1.0 / len(load_indices))

        if np.abs(base_q).sum() > 0.0:
            q_share = np.abs(base_q) / np.abs(base_q).sum()
        else:
            q_share = p_share

        net_copy.load.loc[load_indices, "p_mw"] = profile_p[int(bus)] * p_share
        if "q_mvar" in net_copy.load.columns:
            net_copy.load.loc[load_indices, "q_mvar"] = profile_q[int(bus)] * q_share

    return net_copy


def apply_load_inputs(net: pandapowerNet, load_inputs: pl.DataFrame) -> pandapowerNet:
    net_copy = copy.deepcopy(net)

    if len(net_copy.load) == 0 or load_inputs.is_empty():
        return net_copy

    for row in load_inputs.sort("load_index").iter_rows(named=True):
        load_index = int(row["load_index"])
        if load_index not in net_copy.load.index:
            continue

        net_copy.load.at[load_index, "p_mw"] = float(row["p_mw"])
        if "q_mvar" in net_copy.load.columns and row.get("q_mvar") is not None:
            net_copy.load.at[load_index, "q_mvar"] = float(row["q_mvar"])

    return net_copy


def export_load_inputs(net: pandapowerNet, case_id: str, profile: BusLoadProfile) -> pl.DataFrame:
    if len(net.load) == 0:
        return pl.DataFrame(schema=LOAD_SCHEMA)
    frame = pl.from_pandas(net.load.reset_index(names="load_index")).select(
        pl.lit(case_id).alias("case_id"),
        pl.col("load_index").cast(pl.Int64),
        pl.col("bus").cast(pl.Int64),
        pl.col("p_mw").cast(pl.Float64),
        pl.col("q_mvar").cast(pl.Float64),
        pl.lit(profile.total_p_mw).cast(pl.Float64).alias("total_p_mw"),
        pl.lit(profile.total_q_mvar).cast(pl.Float64).alias("total_q_mvar"),
    )

    if frame.schema != LOAD_SCHEMA:
        raise pl.exceptions.SchemaError(
            f"Unexpected load input schema: {frame.schema!r} != {LOAD_SCHEMA!r}"
        )

    return frame


# Case generation

CASES_SCHEMA = pl.Schema(
    {
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
)


def _concat_frames(frames: list[pl.DataFrame], *, schema: pl.Schema | None = None) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame(schema=schema) if schema is not None else pl.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical")


def generate_cases(cfg: Config) -> dict[str, object]:
    dataset_dir = cfg.dataset_root / "baseline"

    base_net = build_network(cfg.data.network_name)
    buses_static, edges_static = export_static_tables(base_net)

    write_parquet(buses_static, dataset_dir / "buses_static.parquet")
    write_parquet(edges_static, dataset_dir / "edges_static.parquet")

    rng = np.random.default_rng(cfg.data.seed)

    rows: list[dict[str, object]] = []
    load_frames: list[pl.DataFrame] = []
    bus_target_frames: list[pl.DataFrame] = []
    dispatch_target_frames: list[pl.DataFrame] = []

    max_attempts = cfg.data.n_cases * cfg.data.max_attempts_multiplier
    successful_cases = 0
    attempted_cases = 0

    for sample_index in range(max_attempts):
        if successful_cases >= cfg.data.n_cases:
            break

        attempted_cases += 1

        profile = sample_bus_load_profile(
            base_net,
            rng,
            total_scale_min=cfg.perturb.load_scale_min,
            total_scale_max=cfg.perturb.load_scale_max,
        )
        net = apply_bus_load_profile(base_net, profile)
        case_id = make_case_id(cfg.data.network_name, cfg.data.seed, sample_index)
        stats = solve_ac_opf(net, cfg.solver)

        if not stats.success:
            continue

        load_inputs = export_load_inputs(net, case_id, profile)
        rows.append(
            {
                "case_id": case_id,
                "network_name": cfg.data.network_name,
                "sample_index": sample_index,
                "seed": cfg.data.seed,
                "total_p_mw": profile.total_p_mw,
                "total_q_mvar": profile.total_q_mvar,
                "success": True,
                "wall_time_s": stats.wall_time_s,
                "solver_time_s": stats.solver_time_s,
                "objective": stats.objective,
                "termination_status": stats.termination_status,
                "iterations": stats.iterations,
                "error": stats.error,
            }
        )
        load_frames.append(load_inputs)
        bus_target_frames.append(export_bus_results(net, case_id))
        dispatch_target_frames.append(export_dispatch_results(net, case_id))
        successful_cases += 1

    if not rows:
        raise RuntimeError("No converged cases were generated.")

    write_parquet(pl.DataFrame(data=rows, schema=CASES_SCHEMA), dataset_dir / "cases.parquet")
    write_parquet(
        _concat_frames(load_frames, schema=LOAD_SCHEMA), dataset_dir / "load_inputs.parquet"
    )
    write_parquet(_concat_frames(bus_target_frames), dataset_dir / "bus_targets.parquet")
    write_parquet(_concat_frames(dispatch_target_frames), dataset_dir / "dispatch_targets.parquet")

    return {
        "dataset_dir": str(dataset_dir),
        "network_name": cfg.data.network_name,
        "requested_cases": cfg.data.n_cases,
        "generated_cases": successful_cases,
        "attempted_cases": attempted_cases,
    }
