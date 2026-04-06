from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pandapower.networks as pn
import polars as pl
from pandapower.auxiliary import pandapowerNet
from pandapower.create import create_poly_cost

from ..utils import optional_string

DEFAULT_BUS_MIN_VM_PU = 0.95
DEFAULT_BUS_MAX_VM_PU = 1.05
DEFAULT_REACTIVE_LIMIT_MVAR = 1e4
DEFAULT_SLACK_ACTIVE_LIMIT_MW = 1e4
DEFAULT_MAX_LOADING_PERCENT = 100.0

NetworkBuilder = Callable[[], pandapowerNet]

NETWORK_BUILDERS: dict[str, NetworkBuilder] = {
    "case14": cast(NetworkBuilder, pn.case14),
    "case30": cast(NetworkBuilder, pn.case30),
    "case57": cast(NetworkBuilder, pn.case57),
    "case118": cast(NetworkBuilder, pn.case118),
    "case300": cast(NetworkBuilder, pn.case300),
}


def _fill_missing_scalar(table, column: str, value: object) -> None:
    if column not in table.columns:
        table[column] = value
        return

    missing = table[column].isna()
    if bool(missing.any()):
        table.loc[missing, column] = value


def _fill_missing_series(table, column: str, values) -> None:
    if column not in table.columns:
        table[column] = values
        return

    missing = table[column].isna()
    if bool(missing.any()):
        table.loc[missing, column] = values.loc[missing]


def _ensure_bus_constraints(net: pandapowerNet) -> None:
    _fill_missing_scalar(net.bus, "min_vm_pu", DEFAULT_BUS_MIN_VM_PU)
    _fill_missing_scalar(net.bus, "max_vm_pu", DEFAULT_BUS_MAX_VM_PU)


def _ensure_gen_constraints(net: pandapowerNet) -> None:
    if len(net.gen) == 0:
        return

    _fill_missing_scalar(net.gen, "controllable", True)
    _fill_missing_scalar(net.gen, "min_p_mw", 0.0)

    default_max_p = net.gen["p_mw"].abs().clip(lower=1.0) * 1.5
    _fill_missing_series(net.gen, "max_p_mw", default_max_p)

    _fill_missing_scalar(net.gen, "min_q_mvar", -DEFAULT_REACTIVE_LIMIT_MVAR)
    _fill_missing_scalar(net.gen, "max_q_mvar", +DEFAULT_REACTIVE_LIMIT_MVAR)


def _ensure_ext_grid_constraints(net: pandapowerNet) -> None:
    if len(net.ext_grid) == 0:
        return

    _fill_missing_scalar(net.ext_grid, "controllable", True)
    _fill_missing_scalar(net.ext_grid, "min_p_mw", -DEFAULT_SLACK_ACTIVE_LIMIT_MW)
    _fill_missing_scalar(net.ext_grid, "max_p_mw", +DEFAULT_SLACK_ACTIVE_LIMIT_MW)
    _fill_missing_scalar(net.ext_grid, "min_q_mvar", -DEFAULT_REACTIVE_LIMIT_MVAR)
    _fill_missing_scalar(net.ext_grid, "max_q_mvar", +DEFAULT_REACTIVE_LIMIT_MVAR)


def _ensure_sgen_constraints(net: pandapowerNet) -> None:
    if len(net.sgen) == 0:
        return

    _fill_missing_scalar(net.sgen, "controllable", True)
    _fill_missing_scalar(net.sgen, "min_p_mw", 0.0)

    default_max_p = net.sgen["p_mw"].abs().clip(lower=1.0) * 1.5
    _fill_missing_series(net.sgen, "max_p_mw", default_max_p)

    _fill_missing_scalar(net.sgen, "min_q_mvar", -DEFAULT_REACTIVE_LIMIT_MVAR)
    _fill_missing_scalar(net.sgen, "max_q_mvar", +DEFAULT_REACTIVE_LIMIT_MVAR)


def _ensure_branch_limits(net: pandapowerNet) -> None:
    if len(net.line) > 0:
        _fill_missing_scalar(net.line, "max_loading_percent", DEFAULT_MAX_LOADING_PERCENT)
    if len(net.trafo) > 0:
        _fill_missing_scalar(net.trafo, "max_loading_percent", DEFAULT_MAX_LOADING_PERCENT)


def _align_bus_limits_with_voltage_setpoints(net: pandapowerNet) -> None:
    for table_name in ("gen", "ext_grid"):
        table = getattr(net, table_name, None)
        if table is None or len(table) == 0 or "vm_pu" not in table.columns:
            continue

        grouped = table.groupby("bus")["vm_pu"].agg(["min", "max"])

        for bus_index, row in grouped.iterrows():
            vm_min = float(row["min"])
            vm_max = float(row["max"])

            current_min = float(net.bus.at[bus_index, "min_vm_pu"])
            current_max = float(net.bus.at[bus_index, "max_vm_pu"])

            net.bus.at[bus_index, "min_vm_pu"] = min(current_min, vm_min)
            net.bus.at[bus_index, "max_vm_pu"] = max(current_max, vm_max)


def _ensure_costs(net: pandapowerNet) -> None:
    poly_cost = getattr(net, "poly_cost", None)
    if poly_cost is not None and len(poly_cost) > 0:
        return

    pwl_cost = getattr(net, "pwl_cost", None)
    if pwl_cost is not None and len(pwl_cost) > 0:
        return

    if len(net.gen) > 0:
        for position, element_index in enumerate(net.gen.index.to_list()):
            create_poly_cost(
                net,
                int(element_index),
                "gen",
                cp1_eur_per_mw=1.0 + 0.1 * position,
            )

    if len(net.ext_grid) > 0:
        for position, element_index in enumerate(net.ext_grid.index.to_list()):
            create_poly_cost(
                net,
                int(element_index),
                "ext_grid",
                cp1_eur_per_mw=10.0 + 0.1 * position,
            )


def _ensure_opf_ready(net: pandapowerNet) -> pandapowerNet:
    _ensure_bus_constraints(net)
    _ensure_branch_limits(net)
    _ensure_gen_constraints(net)
    _ensure_sgen_constraints(net)
    _ensure_ext_grid_constraints(net)
    _ensure_costs(net)
    _align_bus_limits_with_voltage_setpoints(net)
    return net


def _empty_bus_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "bus_index": pl.Int64,
            "name": pl.Utf8,
            "vn_kv": pl.Float64,
            "type": pl.Utf8,
            "zone": pl.Utf8,
            "min_vm_pu": pl.Float64,
            "max_vm_pu": pl.Float64,
            "in_service": pl.Boolean,
        }
    )


def _empty_edge_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "edge_id": pl.Utf8,
            "edge_type": pl.Utf8,
            "element_index": pl.Int64,
            "name": pl.Utf8,
            "from_bus": pl.Int64,
            "to_bus": pl.Int64,
            "length_km": pl.Float64,
            "r_ohm_per_km": pl.Float64,
            "x_ohm_per_km": pl.Float64,
            "c_nf_per_km": pl.Float64,
            "max_i_ka": pl.Float64,
            "sn_mva": pl.Float64,
            "vn_hv_kv": pl.Float64,
            "vn_lv_kv": pl.Float64,
            "vk_percent": pl.Float64,
            "vkr_percent": pl.Float64,
            "tap_pos": pl.Float64,
            "max_loading_percent": pl.Float64,
            "in_service": pl.Boolean,
        }
    )


def list_supported_networks() -> tuple[str, ...]:
    return tuple(NETWORK_BUILDERS)


def build_network(name: str) -> pandapowerNet:
    try:
        builder = NETWORK_BUILDERS[name]
    except KeyError as error:
        available = ", ".join(list_supported_networks())
        raise ValueError(f"Unknown network {name!r}. Available networks: {available}.") from error

    return _ensure_opf_ready(cast(pandapowerNet, builder()))


def export_static_tables(net: pandapowerNet) -> tuple[pl.DataFrame, pl.DataFrame]:
    # --- Buses ---
    buses = (
        pl.from_pandas(net.bus.reset_index().rename(columns={"index": "bus_index"})).select(
            pl.col("bus_index").cast(pl.Int64),
            pl.col("name").map_elements(optional_string, return_dtype=pl.Utf8),
            pl.col("vn_kv").cast(pl.Float64),
            pl.col("type").map_elements(optional_string, return_dtype=pl.Utf8),
            pl.col("zone").map_elements(optional_string, return_dtype=pl.Utf8),
            pl.col("min_vm_pu").cast(pl.Float64),
            pl.col("max_vm_pu").cast(pl.Float64),
            pl.col("in_service").cast(pl.Boolean),
        )
        if len(net.bus) > 0
        else _empty_bus_frame()
    )

    # --- Lines ---
    if len(net.line) > 0:
        lines = pl.from_pandas(
            net.line.reset_index().rename(columns={"index": "element_index"})
        ).select(
            (pl.lit("line:") + pl.col("element_index").cast(pl.Utf8)).alias("edge_id"),
            pl.lit("line").alias("edge_type"),
            pl.col("element_index").cast(pl.Int64),
            pl.col("name").map_elements(optional_string, return_dtype=pl.Utf8),
            pl.col("from_bus").cast(pl.Int64),
            pl.col("to_bus").cast(pl.Int64),
            pl.col("length_km").cast(pl.Float64),
            pl.col("r_ohm_per_km").cast(pl.Float64),
            pl.col("x_ohm_per_km").cast(pl.Float64),
            pl.col("c_nf_per_km").cast(pl.Float64),
            pl.col("max_i_ka").cast(pl.Float64),
            pl.lit(None).cast(pl.Float64).alias("sn_mva"),
            pl.lit(None).cast(pl.Float64).alias("vn_hv_kv"),
            pl.lit(None).cast(pl.Float64).alias("vn_lv_kv"),
            pl.lit(None).cast(pl.Float64).alias("vk_percent"),
            pl.lit(None).cast(pl.Float64).alias("vkr_percent"),
            pl.lit(None).cast(pl.Float64).alias("tap_pos"),
            pl.col("max_loading_percent").cast(pl.Float64),
            pl.col("in_service").cast(pl.Boolean),
        )
    else:
        lines = _empty_edge_frame()

    # --- Trafos ---
    if len(net.trafo) > 0:
        trafos = pl.from_pandas(
            net.trafo.reset_index().rename(columns={"index": "element_index"})
        ).select(
            (pl.lit("trafo:") + pl.col("element_index").cast(pl.Utf8)).alias("edge_id"),
            pl.lit("trafo").alias("edge_type"),
            pl.col("element_index").cast(pl.Int64),
            pl.col("name").map_elements(optional_string, return_dtype=pl.Utf8),
            pl.col("hv_bus").cast(pl.Int64).alias("from_bus"),
            pl.col("lv_bus").cast(pl.Int64).alias("to_bus"),
            pl.lit(None).cast(pl.Float64).alias("length_km"),
            pl.lit(None).cast(pl.Float64).alias("r_ohm_per_km"),
            pl.lit(None).cast(pl.Float64).alias("x_ohm_per_km"),
            pl.lit(None).cast(pl.Float64).alias("c_nf_per_km"),
            pl.lit(None).cast(pl.Float64).alias("max_i_ka"),
            pl.col("sn_mva").cast(pl.Float64),
            pl.col("vn_hv_kv").cast(pl.Float64),
            pl.col("vn_lv_kv").cast(pl.Float64),
            pl.col("vk_percent").cast(pl.Float64),
            pl.col("vkr_percent").cast(pl.Float64),
            pl.col("tap_pos").cast(pl.Float64),
            pl.col("max_loading_percent").cast(pl.Float64),
            pl.col("in_service").cast(pl.Boolean),
        )
    else:
        trafos = _empty_edge_frame()

    edges = (
        pl.concat([lines, trafos])
        if (len(net.line) > 0 or len(net.trafo) > 0)
        else _empty_edge_frame()
    )
    return buses, edges
