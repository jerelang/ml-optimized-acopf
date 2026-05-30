from __future__ import annotations

import contextlib
import io
import time
from typing import Any, TypedDict

from pandapower.auxiliary import OPFNotConverged, pandapowerNet
from pandapower.runpm import runpm_ac_opf

from ..config import SolverConfig
from ..utils import as_optional_float, as_optional_int
from .callback import make_pp_to_pm_callback
from .io import SolveStats, WarmStartPayload


@contextlib.contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        yield


def solve_ac_opf(
    net: pandapowerNet,
    settings: SolverConfig,
    warmstart: WarmStartPayload | None = None,
) -> SolveStats:
    """Solve AC-OPF with PandaModels/PowerModels and optional warm-start data.
    Returns solve statistics for benchmarking and training.
    """
    pp_to_pm_callback = None
    if warmstart is not None and not warmstart.is_empty:
        pp_to_pm_callback = make_pp_to_pm_callback(warmstart)

    started_at = time.perf_counter()
    try:
        with _suppress_output(settings.silence):
            runpm_ac_opf(
                net,
                pp_to_pm_callback=pp_to_pm_callback,
                calculate_voltage_angles=settings.calculate_voltage_angles,
                delta=settings.delta,
                check_connectivity=settings.check_connectivity,
                pm_solver=settings.pm_solver,
                correct_pm_network_data=settings.correct_pm_network_data,
                silence=settings.silence,
                pm_log_level=settings.pm_log_level,
                delete_buffer_file=settings.delete_buffer_file,
                opf_flow_lim=settings.opf_flow_lim,
                pm_tol=settings.pm_tol,
            )

        wall_time_s = time.perf_counter() - started_at
        return SolveStats(success=True, wall_time_s=wall_time_s, **_extract_pm_stats(net))

    except OPFNotConverged as error:
        wall_time_s = time.perf_counter() - started_at
        return SolveStats(
            success=False,
            wall_time_s=wall_time_s,
            error=str(error),
            **_extract_pm_stats(net),
        )

    except Exception as error:
        wall_time_s = time.perf_counter() - started_at
        return SolveStats(
            success=False,
            wall_time_s=wall_time_s,
            error=f"{type(error).__name__}: {error}",
        )


class PmStats(TypedDict):
    solver_time_s: float | None
    objective: float | None
    termination_status: str | None
    iterations: int | None


def _extract_pm_stats(net: pandapowerNet) -> PmStats:
    objective = as_optional_float(net.get("res_cost", None))
    solver_time_s: float | None = None
    termination_status: str | None = None
    iterations: int | None = None

    pm_result = net.get("_pm_result", None)
    if isinstance(pm_result, dict):
        if objective is None:
            objective = as_optional_float(_first_present(pm_result, ("objective", "f")))
        solver_time_s = as_optional_float(_first_present(pm_result, ("solve_time", "et")))

    pm_org_result = net.get("_pm_org_result", None)
    if isinstance(pm_org_result, dict):
        raw_status = _first_present(
            pm_org_result,
            ("termination_status", "status", "terminationStatus"),
        )
        if raw_status is not None:
            termination_status = str(raw_status)

        iterations = as_optional_int(
            _first_present(
                pm_org_result,
                ("iterations", "iteration_count", "iter", "solve_iterations"),
            )
        )

    return {
        "solver_time_s": solver_time_s,
        "objective": objective,
        "termination_status": termination_status,
        "iterations": iterations,
    }


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None
