from __future__ import annotations

from functools import cache
from importlib.resources import as_file, files

import polars as pl
from pandapower.auxiliary import pandapowerNet
from pandapower.converter.matpower import from_mpc

from ..utils import optional_string

CASE_FILES: dict[str, str] = {
    "case14": "pglib_opf_case14_ieee.m",
    "case30": "pglib_opf_case30_ieee.m",
    "case118": "pglib_opf_case118_ieee.m",
    "case300": "pglib_opf_case300_ieee.m",
    "case118_sad": "pglib_opf_case118_ieee_sad.m",
    "case118_api": "pglib_opf_case118_ieee_api.m",
}

RESOURCE_PACKAGE = "ml_acopf.resources"


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
    return tuple(CASE_FILES)


def build_network(name: str) -> pandapowerNet:
    """Load a supported MATPOWER PGLib-OPF case from resources as a pandapower network."""
    try:
        filename = CASE_FILES[name]
    except KeyError as error:
        available = ", ".join(list_supported_networks())
        raise ValueError(f"Unknown network {name!r}. Available networks: {available}.") from error

    resource = files(RESOURCE_PACKAGE).joinpath("pglib_opf", filename)
    with as_file(resource) as path:
        net = from_mpc(str(path), validate_conversion=False)

    # Because of a current bug in importing MATPOWER cases, the external grids must be set to
    # controllable manually.
    # This preserves the intended opf limits, otherwise an unrelated setpoint is forced.
    # Without this, even small cases like case30 become unsolvable.
    if len(net.ext_grid) > 0:
        net.ext_grid["controllable"] = True
    return net


# Cached to avoid building more than once
@cache
def network_template(name: str) -> pandapowerNet:
    return build_network(name)


def export_static_tables(net: pandapowerNet) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Export static bus and edge tables used to build graph inputs.
    The returned tables contain fixed network attributes, but
    no case-specific loads, dispatch values, or solver outputs.
    """
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
