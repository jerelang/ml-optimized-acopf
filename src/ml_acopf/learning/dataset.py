from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..solver.io import BUS_WARMSTART_SCHEMA, DEVICE_WARMSTART_SCHEMA

DEFAULT_MAX_VOLTAGE_ANGLE_DEG = 60.0

DEVICE_TYPE_TO_ID = {"gen": 0, "ext_grid": 1, "sgen": 2}
ID_TO_DEVICE_TYPE = {value: key for key, value in DEVICE_TYPE_TO_ID.items()}


@dataclass(frozen=True, slots=True)
class CaseMetadata:
    case_id: str
    network_name: str
    sample_index: int
    seed: int
    total_p_mw: float
    total_q_mvar: float


@dataclass(frozen=True, slots=True)
class CaseTables:
    cases: pl.DataFrame
    cases_all: pl.DataFrame
    buses_static: pl.DataFrame
    edges_static: pl.DataFrame
    device_metadata: pl.DataFrame
    load_inputs: pl.DataFrame
    load_inputs_all: pl.DataFrame
    bus_targets: pl.DataFrame
    dispatch_targets: pl.DataFrame


def load_case_tables(dataset_root: Path | str) -> CaseTables:
    root = Path(dataset_root)
    baseline_dir = root if root.name == "baseline" else root / "baseline"

    cases = pl.read_parquet(baseline_dir / "cases.parquet")
    load_inputs = pl.read_parquet(baseline_dir / "load_inputs.parquet")

    attempted_cases_path = baseline_dir / "attempted_cases.parquet"
    attempted_load_inputs_path = baseline_dir / "attempted_load_inputs.parquet"
    device_metadata_path = baseline_dir / "device_metadata.parquet"

    cases_all = pl.read_parquet(attempted_cases_path) if attempted_cases_path.exists() else cases
    load_inputs_all = (
        pl.read_parquet(attempted_load_inputs_path)
        if attempted_load_inputs_path.exists()
        else load_inputs
    )
    device_metadata = (
        pl.read_parquet(device_metadata_path) if device_metadata_path.exists() else pl.DataFrame()
    )

    return CaseTables(
        cases=cases,
        cases_all=cases_all,
        buses_static=pl.read_parquet(baseline_dir / "buses_static.parquet"),
        edges_static=pl.read_parquet(baseline_dir / "edges_static.parquet"),
        device_metadata=device_metadata,
        load_inputs=load_inputs,
        load_inputs_all=load_inputs_all,
        bus_targets=pl.read_parquet(baseline_dir / "bus_targets.parquet"),
        dispatch_targets=pl.read_parquet(baseline_dir / "dispatch_targets.parquet"),
    )


class WarmStartDataset(Dataset[Data]):
    def __init__(
        self,
        dataset_root: Path | str,
        *,
        case_ids: list[str] | None = None,
        max_voltage_angle_deg: float = DEFAULT_MAX_VOLTAGE_ANGLE_DEG,
        case_source: Literal["converged", "all"] = "converged",
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.case_source = case_source
        self.tables = load_case_tables(dataset_root)

        if case_source == "converged":
            cases_source = self.tables.cases
            load_inputs_source = self.tables.load_inputs
        elif case_source == "all":
            cases_source = self.tables.cases_all
            load_inputs_source = self.tables.load_inputs_all

        cases_sorted = cases_source.sort("sample_index").select(
            "case_id",
            "network_name",
            "sample_index",
            "seed",
            "total_p_mw",
            "total_q_mvar",
        )

        if case_ids is not None:
            cases_sorted = cases_sorted.filter(pl.col("case_id").is_in(case_ids))

        self._metadata: tuple[CaseMetadata, ...] = tuple(
            CaseMetadata(*row) for row in cases_sorted.iter_rows()
        )

        self._index_by_case = {item.case_id: i for i, item in enumerate(self._metadata)}
        self._load_inputs_by_case = _partition_by_case(load_inputs_source)
        self._bus_targets_by_case = _partition_by_case(self.tables.bus_targets)
        self._dispatch_targets_by_case = _partition_by_case(self.tables.dispatch_targets)

        buses_static = self.tables.buses_static.sort("bus_index")
        edges_static = self.tables.edges_static
        self._angle_bounds = _symmetric_angle_bounds(max_voltage_angle_deg)

        shared_device_metadata = (
            self.tables.device_metadata if not self.tables.device_metadata.is_empty() else None
        )

        self._graphs: tuple[Data, ...] = tuple(
            build_graph_data_from_case(
                buses_static=buses_static,
                edges_static=edges_static,
                load_inputs=self._load_inputs_by_case.get(item.case_id, pl.DataFrame()),
                bus_targets=self._bus_targets_by_case.get(item.case_id),
                dispatch_targets=self._dispatch_targets_by_case.get(item.case_id),
                device_metadata=shared_device_metadata,
                max_voltage_angle_deg=max_voltage_angle_deg,
            )
            for item in self._metadata
        )

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, index: int) -> Data:
        return self._graphs[index].clone()

    def case_metadata(self, index: int) -> CaseMetadata:
        return self._metadata[index]

    def graph_for_case_id(self, case_id: str) -> Data:
        return self._graphs[self._index_by_case[case_id]].clone()

    def load_inputs_for_case(self, case_id: str) -> pl.DataFrame:
        return self._load_inputs_by_case.get(case_id, pl.DataFrame())

    @property
    def input_channels(self) -> int:
        if not self._graphs or self._graphs[0].x is None:
            raise ValueError("Dataset is empty.")
        return int(self._graphs[0].x.size(-1))

    @property
    def angle_bounds(self) -> tuple[float, float]:
        return self._angle_bounds


def build_graph_data_from_case(
    *,
    buses_static: pl.DataFrame,
    edges_static: pl.DataFrame,
    load_inputs: pl.DataFrame,
    bus_targets: pl.DataFrame | None = None,
    dispatch_targets: pl.DataFrame | None = None,
    device_metadata: pl.DataFrame | None = None,
    max_voltage_angle_deg: float = DEFAULT_MAX_VOLTAGE_ANGLE_DEG,
) -> Data:
    buses = buses_static.sort("bus_index")
    bus_ids = buses["bus_index"].to_list()
    bus_id = torch.tensor(bus_ids, dtype=torch.long)

    bus_positions = pl.DataFrame(
        {
            "bus": bus_ids,
            "bus_pos": list(range(len(bus_ids))),
        }
    )
    edge_index = _edge_index_from_static(edges_static, bus_positions)
    if load_inputs.is_empty():
        load_agg = pl.DataFrame(
            schema={
                "bus": pl.Int64,
                "p_mw": pl.Float64,
                "q_mvar": pl.Float64,
            }
        )
    else:
        load_agg = (
            load_inputs.group_by("bus")
            .agg(
                pl.col("p_mw").sum().alias("p_mw"),
                pl.col("q_mvar").sum().alias("q_mvar"),
            )
            .with_columns(pl.col("bus"))
        )
    buses_joined = (
        buses.rename({"bus_index": "bus"})
        .join(load_agg, on="bus", how="left")
        .with_columns(
            pl.col("p_mw").fill_null(0.0),
            pl.col("q_mvar").fill_null(0.0),
            pl.col("vn_kv").fill_null(0.0),
            pl.col("min_vm_pu").fill_null(0.0),
            pl.col("max_vm_pu").fill_null(0.0),
            pl.col("in_service").cast(pl.Float32).fill_null(1.0),
        )
        .sort("bus")
    )
    x = torch.tensor(
        buses_joined.select(
            pl.col("p_mw").cast(pl.Float32),
            pl.col("q_mvar").cast(pl.Float32),
            pl.col("vn_kv").cast(pl.Float32),
            pl.col("min_vm_pu").cast(pl.Float32),
            pl.col("max_vm_pu").cast(pl.Float32),
            pl.col("in_service").cast(pl.Float32),
        ).to_numpy(),
        dtype=torch.float32,
    )

    vm_lower = torch.tensor(
        buses["min_vm_pu"].to_numpy(),
        dtype=torch.float32,
    )
    vm_upper = torch.tensor(
        buses["max_vm_pu"].to_numpy(),
        dtype=torch.float32,
    )

    va_min, va_max = _symmetric_angle_bounds(max_voltage_angle_deg)
    va_lower = torch.full((len(bus_ids),), float(va_min), dtype=torch.float32)
    va_upper = torch.full((len(bus_ids),), float(va_max), dtype=torch.float32)

    data_kwargs: dict[str, Tensor] = {
        "x": x,
        "edge_index": edge_index,
        "bus_id": bus_id,
        "vm_lower": vm_lower,
        "vm_upper": vm_upper,
        "va_lower": va_lower,
        "va_upper": va_upper,
    }

    if bus_targets is not None and not bus_targets.is_empty():
        bus_targets_joined = (
            buses_joined.select("bus")
            .join(
                bus_targets.select(
                    pl.col("bus_index").alias("bus"),
                    pl.col("vm_pu"),
                    pl.col("va_degree"),
                ),
                on="bus",
                how="left",
            )
            .sort("bus")
        )

        missing_vm = bus_targets_joined["vm_pu"].null_count()
        missing_va = bus_targets_joined["va_degree"].null_count()

        if missing_vm > 0 or missing_va > 0:
            raise ValueError(
                "Incomplete bus targets: "
                f"missing vm_pu={missing_vm}, missing va_degree={missing_va}"
            )

        raw_bus_targets = torch.tensor(
            bus_targets_joined.select("vm_pu", "va_degree").to_numpy(),
            dtype=torch.float32,
        )
        data_kwargs["y_bus"] = normalize_bus_values(
            raw_bus_targets,
            vm_lower,
            vm_upper,
            va_lower,
            va_upper,
        )
    else:
        data_kwargs["y_bus"] = torch.zeros((len(bus_ids), 2), dtype=torch.float32)
    prepared_device_table = _prepare_device_table(dispatch_targets, device_metadata)

    if not prepared_device_table.is_empty():
        devices_joined = (
            prepared_device_table.with_columns(
                pl.col("bus"),
                pl.col("element_index"),
            )
            .join(bus_positions, on="bus", how="inner")
            .sort("element_type", "element_index")
        )
        if devices_joined.height != prepared_device_table.height:
            raise ValueError("Devices and bus dataset length deviate.")
        if devices_joined.height > 0:
            device_bus_index = torch.tensor(
                devices_joined["bus_pos"].to_numpy(),
                dtype=torch.long,
            )
            device_type_id = torch.tensor(
                [
                    DEVICE_TYPE_TO_ID[str(value)]
                    for value in devices_joined["element_type"].to_list()
                ],
                dtype=torch.long,
            )
            device_element_id = torch.tensor(
                devices_joined["element_index"].to_numpy(),
                dtype=torch.long,
            )

            if devices_joined["min_p_mw"].null_count():
                raise ValueError("Incomplete device p lower bound")
            device_p_lower = torch.tensor(
                devices_joined["min_p_mw"].to_numpy(),
                dtype=torch.float32,
            )

            if devices_joined["max_p_mw"].null_count():
                raise ValueError("Incomplete device p higher bound")
            device_p_upper = torch.tensor(
                devices_joined["max_p_mw"].to_numpy(),
                dtype=torch.float32,
            )

            if devices_joined["min_q_mvar"].null_count():
                raise ValueError("Incomplete device q lower bound")
            device_q_lower = torch.tensor(
                devices_joined["min_q_mvar"].to_numpy(),
                dtype=torch.float32,
            )

            if devices_joined["max_q_mvar"].null_count():
                raise ValueError("Incomplete device q higher bound")
            device_q_upper = torch.tensor(
                devices_joined["max_q_mvar"].to_numpy(),
                dtype=torch.float32,
            )

            data_kwargs.update(
                {
                    "device_bus_index": device_bus_index,
                    "device_type_id": device_type_id,
                    "device_element_id": device_element_id,
                    "device_p_lower": device_p_lower,
                    "device_p_upper": device_p_upper,
                    "device_q_lower": device_q_lower,
                    "device_q_upper": device_q_upper,
                }
            )

            if dispatch_targets is not None and not dispatch_targets.is_empty():
                if devices_joined["p_mw"].null_count() or devices_joined["q_mvar"].null_count():
                    raise ValueError("Incomplete device targets")

                raw_device_targets = torch.tensor(
                    devices_joined.select("p_mw", "q_mvar").to_numpy(),
                    dtype=torch.float32,
                )
                data_kwargs["y_device"] = normalize_device_values(
                    raw_device_targets,
                    device_p_lower,
                    device_p_upper,
                    device_q_lower,
                    device_q_upper,
                )
            else:
                data_kwargs["y_device"] = torch.zeros(
                    (devices_joined.height, 2),
                    dtype=torch.float32,
                )
        else:
            data_kwargs.update(_empty_device_tensors())
    else:
        data_kwargs.update(_empty_device_tensors())

    return Data(**data_kwargs)


def _empty_device_tensors() -> dict[str, Tensor]:
    return {
        "device_bus_index": torch.empty((0,), dtype=torch.long),
        "device_type_id": torch.empty((0,), dtype=torch.long),
        "device_element_id": torch.empty((0,), dtype=torch.long),
        "device_p_lower": torch.empty((0,), dtype=torch.float32),
        "device_p_upper": torch.empty((0,), dtype=torch.float32),
        "device_q_lower": torch.empty((0,), dtype=torch.float32),
        "device_q_upper": torch.empty((0,), dtype=torch.float32),
        "y_device": torch.empty((0, 2), dtype=torch.float32),
    }


def normalize_bus_values(
    values: Tensor,
    vm_lower: Tensor,
    vm_upper: Tensor,
    va_lower: Tensor,
    va_upper: Tensor,
) -> Tensor:
    vm_span = (vm_upper - vm_lower).clamp_min(1e-6)
    va_span = (va_upper - va_lower).clamp_min(1e-6)

    out = torch.empty_like(values)
    out[:, 0] = ((values[:, 0] - vm_lower) / vm_span).clamp(0.0, 1.0)
    out[:, 1] = ((values[:, 1] - va_lower) / va_span).clamp(0.0, 1.0)
    return out


def denormalize_bus_values(
    values: Tensor,
    vm_lower: Tensor,
    vm_upper: Tensor,
    va_lower: Tensor,
    va_upper: Tensor,
) -> Tensor:
    clamped = values.clamp(0.0, 1.0)
    vm_span = (vm_upper - vm_lower).clamp_min(1e-6)
    va_span = (va_upper - va_lower).clamp_min(1e-6)

    out = torch.empty_like(clamped)
    out[:, 0] = vm_lower + clamped[:, 0] * vm_span
    out[:, 1] = va_lower + clamped[:, 1] * va_span
    return out


def normalize_device_values(
    values: Tensor,
    p_lower: Tensor,
    p_upper: Tensor,
    q_lower: Tensor,
    q_upper: Tensor,
) -> Tensor:
    p_span = (p_upper - p_lower).clamp_min(1e-6)
    q_span = (q_upper - q_lower).clamp_min(1e-6)

    out = torch.empty_like(values)
    out[:, 0] = ((values[:, 0] - p_lower) / p_span).clamp(0.0, 1.0)
    out[:, 1] = ((values[:, 1] - q_lower) / q_span).clamp(0.0, 1.0)
    return out


def denormalize_device_values(
    values: Tensor,
    p_lower: Tensor,
    p_upper: Tensor,
    q_lower: Tensor,
    q_upper: Tensor,
) -> Tensor:
    clamped = values.clamp(0.0, 1.0)
    p_span = (p_upper - p_lower).clamp_min(1e-6)
    q_span = (q_upper - q_lower).clamp_min(1e-6)

    out = torch.empty_like(clamped)
    out[:, 0] = p_lower + clamped[:, 0] * p_span
    out[:, 1] = q_lower + clamped[:, 1] * q_span
    return out


def build_bus_warmstart_frame(
    case_id: str,
    bus_id: Tensor,
    bus_values: Tensor,
) -> pl.DataFrame:
    cpu_id = bus_id.detach().cpu().to(dtype=torch.int64)
    cpu_values = bus_values.detach().cpu().to(dtype=torch.float32)

    rows = [
        {
            "case_id": case_id,
            "bus_index": int(cpu_id[i].item()),
            "vm_pu": float(cpu_values[i, 0].item()),
            "va_degree": float(cpu_values[i, 1].item()),
        }
        for i in range(cpu_id.numel())
    ]
    return pl.DataFrame(rows, schema=BUS_WARMSTART_SCHEMA)


def build_device_warmstart_frame(
    case_id: str,
    device_type_id: Tensor,
    device_element_id: Tensor,
    device_values: Tensor,
) -> pl.DataFrame:
    cpu_type = device_type_id.detach().cpu().to(dtype=torch.int64)
    cpu_element = device_element_id.detach().cpu().to(dtype=torch.int64)
    cpu_values = device_values.detach().cpu().to(dtype=torch.float32)

    rows = [
        {
            "case_id": case_id,
            "element_type": ID_TO_DEVICE_TYPE[int(cpu_type[i].item())],
            "element_index": int(cpu_element[i].item()),
            "p_mw": float(cpu_values[i, 0].item()),
            "q_mvar": float(cpu_values[i, 1].item()),
        }
        for i in range(cpu_type.numel())
    ]
    return pl.DataFrame(rows, schema=DEVICE_WARMSTART_SCHEMA)


def _partition_by_case(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    if frame.is_empty():
        return {}

    out: dict[str, pl.DataFrame] = {}
    for key, value in frame.partition_by("case_id", as_dict=True).items():
        case_id = str(key[0]) if isinstance(key, tuple) else str(key)
        out[case_id] = value
    return out


def _edge_index_from_static(
    edges_static: pl.DataFrame,
    bus_positions: pl.DataFrame,
) -> Tensor:
    df = (
        edges_static.select("from_bus", "to_bus")
        .join(
            bus_positions.rename({"bus": "from_bus", "bus_pos": "src"}), on="from_bus", how="inner"
        )
        .join(bus_positions.rename({"bus": "to_bus", "bus_pos": "dst"}), on="to_bus", how="inner")
    )

    src = pl.concat([df["src"], df["dst"]])
    dst = pl.concat([df["dst"], df["src"]])

    return torch.tensor([src.to_list(), dst.to_list()], dtype=torch.long)


def _prepare_device_table(
    dispatch_targets: pl.DataFrame | None,
    device_metadata: pl.DataFrame | None,
) -> pl.DataFrame:
    if dispatch_targets is not None and not dispatch_targets.is_empty():
        return dispatch_targets.select(
            "element_type",
            "element_index",
            "bus",
            "min_p_mw",
            "max_p_mw",
            "min_q_mvar",
            "max_q_mvar",
            "p_mw",
            "q_mvar",
        ).sort("element_type", "element_index")

    if device_metadata is not None and not device_metadata.is_empty():
        return device_metadata.select(
            "element_type",
            "element_index",
            "bus",
            "min_p_mw",
            "max_p_mw",
            "min_q_mvar",
            "max_q_mvar",
        ).sort("element_type", "element_index")

    return pl.DataFrame()


def _symmetric_angle_bounds(max_voltage_angle_deg: float) -> tuple[float, float]:
    max_angle = float(max_voltage_angle_deg)
    return -max_angle, max_angle
