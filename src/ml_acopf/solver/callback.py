from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import polars as pl

from ..utils import as_optional_float
from .io import WarmStartPayload


def make_pp_to_pm_callback(
    payload: WarmStartPayload,
) -> Callable[[object, object, dict[str, Any]], None]:
    def callback(
        net: object,
        ppci: object,
        pm: dict[str, Any],
    ) -> None:
        del net, ppci  # required by pandapower callback signature

        if payload.bus is not None and not payload.bus.is_empty():
            _inject_bus_starts(pm, payload.bus)

        if payload.device is not None and not payload.device.is_empty():
            _inject_device_starts(pm, payload.device)

    return callback


def _inject_bus_starts(pm: dict[str, Any], frame: pl.DataFrame) -> None:
    rows = (
        frame.select("bus_index", "vm_pu", "va_degree")
        .with_columns(
            pl.col("bus_index").cast(pl.Int64),
            pl.col("vm_pu").cast(pl.Float64),
            pl.col("va_degree").cast(pl.Float64),
        )
        .sort("bus_index")
        .iter_rows(named=True)
    )

    lookup: dict[int, tuple[float, float]] = {}
    for row in rows:
        vm_pu = row["vm_pu"]
        va_degree = row["va_degree"]
        if vm_pu is None or va_degree is None:
            continue

        lookup[int(row["bus_index"])] = (
            float(vm_pu),
            math.radians(float(va_degree)),
        )

    for bus in _pm_component_values(pm, "bus"):
        bus_key = _source_key(bus, default_type="bus")
        if bus_key is None:
            continue
        bus_index = bus_key[1]

        start = lookup.get(bus_index)
        if start is None:
            continue

        vm_start, va_start = start
        bus["vm_start"] = vm_start
        bus["va_start"] = va_start


def _inject_device_starts(pm: dict[str, Any], frame: pl.DataFrame) -> None:
    rows = (
        frame.select("element_type", "element_index", "p_mw", "q_mvar")
        .with_columns(
            pl.col("element_type").cast(pl.String),
            pl.col("element_index").cast(pl.Int64),
            pl.col("p_mw").cast(pl.Float64),
            pl.col("q_mvar").cast(pl.Float64),
        )
        .sort("element_type", "element_index")
        .iter_rows(named=True)
    )

    lookup: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        lookup[(str(row["element_type"]), int(row["element_index"]))] = row

    base_mva = float(pm.get("baseMVA", 1.0))
    if base_mva == 0.0:
        base_mva = 1.0

    # In PowerModels, active/reactive start values live on generator-like components.
    for gen in _pm_component_values(pm, "gen"):
        source_key = _source_key(gen, default_type="gen")
        if source_key is None:
            continue

        row = lookup.get(source_key)
        if row is None:
            continue

        p_mw = as_optional_float(row.get("p_mw"))
        q_mvar = as_optional_float(row.get("q_mvar"))

        if p_mw is not None:
            gen["pg_start"] = p_mw / base_mva
        if q_mvar is not None:
            gen["qg_start"] = q_mvar / base_mva


def _pm_component_values(pm: dict[str, Any], component_type: str) -> list[dict[str, Any]]:
    component_dict = pm.get(component_type)
    if not isinstance(component_dict, dict):
        return []

    values: list[dict[str, Any]] = []
    for value in component_dict.values():
        if isinstance(value, dict):
            values.append(value)
    return values


def _source_key(
    component: dict[str, Any],
    *,
    default_type: str,
) -> tuple[str, int] | None:
    source_id = component.get("source_id")

    if isinstance(source_id, (list, tuple)) and len(source_id) >= 2:
        try:
            return str(source_id[0]), int(source_id[1])
        except (TypeError, ValueError):
            pass

    try:
        return default_type, int(component["index"])
    except (KeyError, TypeError, ValueError):
        return None
