from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl
from pandapower.auxiliary import pandapowerNet

from ..utils import as_optional_float

DeviceElementType = Literal["gen", "ext_grid", "sgen"]

BUS_WARMSTART_SCHEMA = pl.Schema(
    {
        "case_id": pl.String(),
        "bus_index": pl.Int64(),
        "vm_pu": pl.Float64(),
        "va_degree": pl.Float64(),
    }
)

DEVICE_WARMSTART_SCHEMA = pl.Schema(
    {
        "case_id": pl.String(),
        "element_type": pl.String(),
        "element_index": pl.Int64(),
        "p_mw": pl.Float64(),
        "q_mvar": pl.Float64(),
    }
)

DEVICE_METADATA_SCHEMA = pl.Schema(
    {
        "element_type": pl.String(),
        "element_index": pl.Int64(),
        "bus": pl.Int64(),
        "min_p_mw": pl.Float64(),
        "max_p_mw": pl.Float64(),
        "min_q_mvar": pl.Float64(),
        "max_q_mvar": pl.Float64(),
    }
)

BUS_RESULTS_SCHEMA = pl.Schema(
    {
        "case_id": pl.String(),
        "bus_index": pl.Int64(),
        "vm_pu": pl.Float64(),
        "va_degree": pl.Float64(),
        "p_mw": pl.Float64(),
        "q_mvar": pl.Float64(),
    }
)

DISPATCH_RESULTS_SCHEMA = pl.Schema(
    {
        "case_id": pl.String(),
        "element_type": pl.String(),
        "element_index": pl.Int64(),
        "bus": pl.Int64(),
        "p_mw": pl.Float64(),
        "q_mvar": pl.Float64(),
        "min_p_mw": pl.Float64(),
        "max_p_mw": pl.Float64(),
        "min_q_mvar": pl.Float64(),
        "max_q_mvar": pl.Float64(),
    }
)


@dataclass(frozen=True, slots=True)
class WarmStartPayload:
    bus: pl.DataFrame | None = None
    device: pl.DataFrame | None = None

    @property
    def is_empty(self) -> bool:
        return (self.bus is None or self.bus.is_empty()) and (
            self.device is None or self.device.is_empty()
        )


@dataclass(frozen=True, slots=True)
class SolveStats:
    success: bool
    wall_time_s: float
    solver_time_s: float | None = None
    objective: float | None = None
    termination_status: str | None = None
    iterations: int | None = None
    error: str | None = None


def export_bus_warmstart(net: pandapowerNet, case_id: str) -> pl.DataFrame:
    result_table = getattr(net, "res_bus", None)
    if result_table is None or len(result_table) == 0:
        return pl.DataFrame(schema=BUS_WARMSTART_SCHEMA)

    rows: list[dict[str, object]] = []
    for bus_index, row in result_table.iterrows():
        rows.append(
            {
                "case_id": case_id,
                "bus_index": int(bus_index),
                "vm_pu": as_optional_float(row.get("vm_pu")),
                "va_degree": as_optional_float(row.get("va_degree")),
            }
        )

    return pl.DataFrame(rows, schema=BUS_WARMSTART_SCHEMA)


def export_device_warmstart(net: pandapowerNet, case_id: str) -> pl.DataFrame:
    rows: list[dict[str, object]] = []

    _append_device_start_rows(rows, net, case_id, "gen", "gen")
    _append_device_start_rows(rows, net, case_id, "ext_grid", "ext_grid")
    _append_device_start_rows(rows, net, case_id, "sgen", "sgen")

    if not rows:
        return pl.DataFrame(schema=DEVICE_WARMSTART_SCHEMA)

    return pl.DataFrame(rows, schema=DEVICE_WARMSTART_SCHEMA)


def export_device_metadata(net: pandapowerNet) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    _append_device_metadata_rows(rows, net, "gen", "gen")
    _append_device_metadata_rows(rows, net, "ext_grid", "ext_grid")
    _append_device_metadata_rows(rows, net, "sgen", "sgen")

    if not rows:
        return pl.DataFrame(schema=DEVICE_METADATA_SCHEMA)

    return pl.DataFrame(rows, schema=DEVICE_METADATA_SCHEMA)


def export_bus_results(net: pandapowerNet, case_id: str) -> pl.DataFrame:
    result_table = getattr(net, "res_bus", None)
    if result_table is None or len(result_table) == 0:
        return pl.DataFrame(schema=BUS_RESULTS_SCHEMA)

    rows: list[dict[str, object]] = []
    for bus_index, row in result_table.iterrows():
        rows.append(
            {
                "case_id": case_id,
                "bus_index": int(bus_index),
                "vm_pu": as_optional_float(row.get("vm_pu")),
                "va_degree": as_optional_float(row.get("va_degree")),
                "p_mw": as_optional_float(row.get("p_mw")),
                "q_mvar": as_optional_float(row.get("q_mvar")),
            }
        )

    return pl.DataFrame(rows, schema=BUS_RESULTS_SCHEMA)


def export_dispatch_results(net: pandapowerNet, case_id: str) -> pl.DataFrame:
    rows: list[dict[str, object]] = []

    _append_dispatch_rows(rows, net, case_id, "gen", "gen")
    _append_dispatch_rows(rows, net, case_id, "ext_grid", "ext_grid")
    _append_dispatch_rows(rows, net, case_id, "sgen", "sgen")

    if not rows:
        return pl.DataFrame(schema=DISPATCH_RESULTS_SCHEMA)

    return pl.DataFrame(rows, schema=DISPATCH_RESULTS_SCHEMA)


def _append_device_metadata_rows(
    rows: list[dict[str, object]],
    net: pandapowerNet,
    table_name: str,
    element_type: str,
) -> None:
    element_table = getattr(net, table_name, None)
    if element_table is None or len(element_table) == 0:
        return

    for element_index, element_row in element_table.iterrows():
        rows.append(
            {
                "element_type": element_type,
                "element_index": int(element_index),
                "bus": int(element_row["bus"]),
                "min_p_mw": as_optional_float(element_row.get("min_p_mw")),
                "max_p_mw": as_optional_float(element_row.get("max_p_mw")),
                "min_q_mvar": as_optional_float(element_row.get("min_q_mvar")),
                "max_q_mvar": as_optional_float(element_row.get("max_q_mvar")),
            }
        )


def _append_device_start_rows(
    rows: list[dict[str, object]],
    net: pandapowerNet,
    case_id: str,
    table_name: str,
    element_type: str,
) -> None:
    result_table = getattr(net, f"res_{table_name}", None)
    if result_table is None or len(result_table) == 0:
        return

    for element_index, result_row in result_table.iterrows():
        rows.append(
            {
                "case_id": case_id,
                "element_type": element_type,
                "element_index": int(element_index),
                "p_mw": as_optional_float(result_row.get("p_mw")),
                "q_mvar": as_optional_float(result_row.get("q_mvar")),
            }
        )


def _append_dispatch_rows(
    rows: list[dict[str, object]],
    net: pandapowerNet,
    case_id: str,
    table_name: str,
    element_type: str,
) -> None:
    element_table = getattr(net, table_name, None)
    result_table = getattr(net, f"res_{table_name}", None)

    if element_table is None or result_table is None:
        return
    if len(element_table) == 0 or len(result_table) == 0:
        return

    for element_index, result_row in result_table.iterrows():
        if element_index not in element_table.index:
            continue

        element_row = element_table.loc[element_index]
        rows.append(
            {
                "case_id": case_id,
                "element_type": element_type,
                "element_index": int(element_index),
                "bus": int(element_row["bus"]),
                "p_mw": as_optional_float(result_row.get("p_mw")),
                "q_mvar": as_optional_float(result_row.get("q_mvar")),
                "min_p_mw": as_optional_float(element_row.get("min_p_mw")),
                "max_p_mw": as_optional_float(element_row.get("max_p_mw")),
                "min_q_mvar": as_optional_float(element_row.get("min_q_mvar")),
                "max_q_mvar": as_optional_float(element_row.get("max_q_mvar")),
            }
        )
