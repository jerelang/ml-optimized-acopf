from __future__ import annotations

import math

import polars as pl

from ml_acopf.solver.callback import make_pp_to_pm_callback
from ml_acopf.solver.io import BUS_WARMSTART_SCHEMA, DEVICE_WARMSTART_SCHEMA, WarmStartPayload


def test_callback_injects_warmstart_values() -> None:
    payload = WarmStartPayload(
        bus=pl.DataFrame(
            [
                {"case_id": "c1", "bus_index": 0, "vm_pu": 1.02, "va_degree": 10.0},
                {"case_id": "c1", "bus_index": 1, "vm_pu": 0.98, "va_degree": -5.0},
            ],
            schema=BUS_WARMSTART_SCHEMA,
        ),
        device=pl.DataFrame(
            [
                {
                    "case_id": "c1",
                    "element_type": "gen",
                    "element_index": 0,
                    "p_mw": 50.0,
                    "q_mvar": 10.0,
                },
                {
                    "case_id": "c1",
                    "element_type": "ext_grid",
                    "element_index": 0,
                    "p_mw": 20.0,
                    "q_mvar": -4.0,
                },
            ],
            schema=DEVICE_WARMSTART_SCHEMA,
        ),
    )

    pm = {
        "baseMVA": 100.0,
        "bus": {
            "1": {"source_id": ["bus", 0]},
            "2": {"source_id": ["bus", 1]},
        },
        "gen": {
            "1": {"source_id": ["gen", 0]},
            "2": {"source_id": ["ext_grid", 0]},
        },
    }

    callback = make_pp_to_pm_callback(payload)
    callback(None, None, pm)

    assert pm["bus"]["1"]["vm_start"] == 1.02
    assert pm["bus"]["2"]["vm_start"] == 0.98
    assert pm["bus"]["1"]["va_start"] == math.radians(10.0)
    assert pm["bus"]["2"]["va_start"] == math.radians(-5.0)

    assert pm["gen"]["1"]["pg_start"] == 0.5
    assert pm["gen"]["1"]["qg_start"] == 0.1
    assert pm["gen"]["2"]["pg_start"] == 0.2
    assert pm["gen"]["2"]["qg_start"] == -0.04
