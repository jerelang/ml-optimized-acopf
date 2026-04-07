from __future__ import annotations

import math
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

from .config import Config


def make_case_id(network_name: str, seed: int, sample_index: int) -> str:
    return f"{network_name}_seed{seed}_sample{sample_index:06d}"


def write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def optional_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()
    return text or None


def make_run_name(cfg: Config) -> str:
    return f"{cfg.data.name}_n{cfg.data.network_name}_seed{cfg.data.seed}"


def print_rich(df: pl.DataFrame):
    table = Table(show_header=True, header_style="bold cyan")
    for col in df.columns:
        table.add_column(col, justify="right")
    for row in df.iter_rows():
        table.add_row(*[str(v) for v in row])
    Console().print(table)
