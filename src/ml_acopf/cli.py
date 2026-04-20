from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from .utils import detect_os

opsys = detect_os()


def ensure_julia_ready() -> None:
    if opsys == "linux":
        import juliacall  # noqa F401
        from juliacall import Main as jl

        jl.seval("using Ipopt, PowerModels, JSON, JuMP, PandaModels")
    elif opsys == "macos":
        import juliacall  # noqa F401


app = typer.Typer(add_completion=False, no_args_is_help=True)

DEFAULT_CONFIG_PATH = Path("configs/default.toml")
DEFAULT_BENCHMARK_OUT = Path("outputs/results")
DEFAULT_ACTOR_OUT = Path("outputs/models/")
DEFAULT_SEARCH_VALIDATE_OUT = Path("outputs/search/")


@app.command("list-networks", help="List the bundled benchmark networks.")
def list_networks_command() -> None:
    from .cases.networks import list_supported_networks

    typer.echo("\n".join(list_supported_networks()))


@app.command(
    name="generate-cases",
    help="Generate the networks and their baselines and store them as parquet files.",
)
def cases_command(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    ensure_julia_ready()
    from .cases.generate import generate_cases
    from .config import load_config

    cfg = load_config(config)
    info = generate_cases(cfg)
    typer.echo(json.dumps(info, indent=2))


@app.command(
    name="search",
    help="Do a small hyperparameter search using a train/validation split.",
)
def search_command(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = DEFAULT_CONFIG_PATH,
    out: Annotated[
        Path,
        typer.Option("--out", "-o"),
    ] = DEFAULT_SEARCH_VALIDATE_OUT,
) -> None:
    ensure_julia_ready()
    import torch

    from .learning.search import search

    device = torch.device("cpu")
    if opsys == "linux" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif opsys == "macos" and torch.backends.mps.is_available():
        device = torch.device("mps")
    search(config, out, device)


@app.command(
    name="train",
    help="Pretrain the actor using the offline dataset and then train with PPO.",
)
def train_command(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ] = DEFAULT_CONFIG_PATH,
    config_from_actor: Annotated[
        Path | None,
        typer.Option("--config-actor", "-ca", exists=True, dir_okay=False, readable=True),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", "-o"),
    ] = DEFAULT_ACTOR_OUT,
    plot: bool = True,
) -> None:
    ensure_julia_ready()
    import polars as pl
    import torch

    from .config import Config, load_config
    from .learning.dataset import WarmStartDataset
    from .learning.models import (
        VoltageWarmStartActor,
        VoltageWarmStartCritic,
        save_actor,
    )
    from .learning.ppo import PPOConfig
    from .learning.train import pretrain_actor_supervised, train_ppo
    from .plotting import plot_ppo_history, plot_pretrain_history
    from .utils import make_run_name, write_parquet

    device = torch.device("cpu")
    if opsys == "linux" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif opsys == "macos" and torch.backends.mps.is_available():
        device = torch.device("mps")

    if config_from_actor is not None:
        checkpoint = torch.load(config_from_actor, map_location="cpu")
        cfg = Config.model_validate(checkpoint["full_cfg"])
    else:
        cfg = load_config(config)

    pretrain_dataset = WarmStartDataset(
        cfg.dataset_root,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
        case_source="converged",
    )

    ppo_dataset = WarmStartDataset(
        cfg.dataset_root,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
        case_source="all",
    )
    typer.echo(f"pretrain/PPO cases: {len(pretrain_dataset)} / {len(ppo_dataset)}")
    actor = VoltageWarmStartActor(
        in_channels=pretrain_dataset.input_channels,
        hidden_channels=cfg.model.hidden_channels,
        dropout=cfg.model.dropout,
        action_std_init=cfg.model.action_std_init,
        device_type_embedding_dim=cfg.model.device_type_embedding_dim,
    )
    pretrain_history = pretrain_actor_supervised(
        actor,
        pretrain_dataset,
        device=device,
        epochs=cfg.pretrain.epochs,
        batch_size=cfg.pretrain.batch_size,
        learning_rate=cfg.pretrain.learning_rate,
        weight_decay=cfg.pretrain.weight_decay,
    )
    run_name = make_run_name(cfg)
    out = out / run_name
    out.mkdir(parents=True, exist_ok=True)
    save_actor(
        actor=actor,
        input_channels=pretrain_dataset.input_channels,
        cfg=cfg,
        history=pretrain_history,
        out=out,
    )
    write_parquet(pl.DataFrame(pretrain_history), out / "pretrain_history.parquet")

    critic = VoltageWarmStartCritic(
        in_channels=ppo_dataset.input_channels,
        hidden_channels=cfg.model.hidden_channels,
        dropout=cfg.model.dropout,
    )
    ppo_model_config = PPOConfig(
        learning_rate=cfg.ppo.learning_rate,
        clip_ratio=cfg.ppo.clip_ratio,
        value_loss_weight=cfg.ppo.value_loss_weight,
        entropy_weight=cfg.ppo.entropy_weight,
        ppo_epochs=cfg.ppo.ppo_epochs,
        max_grad_norm=cfg.ppo.max_grad_norm,
    )
    agent, ppo_history = train_ppo(
        actor,
        critic,
        ppo_dataset,
        device=device,
        solver=cfg.solver,
        ppo_config=ppo_model_config,
        updates=cfg.ppo.updates,
        rollout_size=cfg.ppo.rollout_size,
        seed=cfg.ppo.seed,
        nonconvergence_penalty=cfg.ppo.nonconvergence_penalty,
    )
    save_actor(
        actor=agent.actor,
        input_channels=ppo_dataset.input_channels,
        cfg=cfg,
        history=ppo_history,
        out=out,
    )
    write_parquet(pl.DataFrame(ppo_history), out / "ppo_history.parquet")

    if plot:
        created: list[Path] = []

        if (out / "pretrain_history.parquet").exists():
            created.append(plot_pretrain_history(out))
        if (out / "ppo_history.parquet").exists():
            created.extend(plot_ppo_history(out))

        for path in created:
            typer.echo(f"wrote: {path}")


@app.command(
    "benchmark",
    help="""Run the warm-start benchmark on fresh sampled AC-OPF cases.\n
    Uses the benchmark and normalization settings stored in the actor checkpoint config.""",
)
def benchmark_command(
    actor_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
        ),
    ] = DEFAULT_BENCHMARK_OUT,
    plot: bool = True,
) -> None:
    ensure_julia_ready()

    import torch

    from .benchmark import run_benchmark, summarize_benchmark
    from .config import Config
    from .learning.models import WarmStartPredictor, load_actor
    from .plotting import plot_benchmark_results
    from .utils import make_run_name, print_rich, write_parquet

    checkpoint = torch.load(actor_path, map_location="cpu")
    cfg = Config.model_validate(checkpoint["full_cfg"])

    actor = load_actor(actor_path)
    device = torch.device("cpu")
    if opsys == "linux" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif opsys == "macos" and torch.backends.mps.is_available():
        device = torch.device("mps")
    actor.to(device)
    predictor = WarmStartPredictor(model=actor, device=device)
    network_names = tuple(cfg.benchmark.network_names) or (cfg.data.network_name,)
    run_name = make_run_name(cfg)
    result_path = out / run_name / f"{run_name}_benchmark.parquet"

    frame = run_benchmark(
        network_names=network_names,
        n_cases=cfg.benchmark.n_cases,
        seed=cfg.benchmark.seed,
        load_scale_min=cfg.perturb.load_scale_min,
        load_scale_max=cfg.perturb.load_scale_max,
        local_load_noise_scale=cfg.perturb.local_load_noise_scale,
        reactive_noise_scale=cfg.perturb.reactive_noise_scale,
        solver=cfg.solver,
        include_flat=cfg.benchmark.include_flat,
        include_pf=cfg.benchmark.include_pf,
        predictor=predictor,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
        out=result_path,
    )

    summary = summarize_benchmark(frame)
    summary_out = result_path.with_name(f"{result_path.stem}_summary.parquet")
    write_parquet(summary, summary_out)

    typer.echo(f"wrote: {result_path}")
    typer.echo(f"wrote: {summary_out}")
    print_rich(summary)

    if plot:
        for path in plot_benchmark_results(result_path):
            typer.echo(f"wrote: {path}")


@app.command("plot-benchmark", help="Create plots for a saved benchmark parquet file.")
def plot_benchmark_command(
    benchmark_file: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True),
    ],
) -> None:
    from .plotting import plot_benchmark_results

    created = plot_benchmark_results(benchmark_file)
    for path in created:
        typer.echo(f"wrote: {path}")


@app.command("plot-training", help="Create training plots for a saved training run.")
def plot_training_command(
    run_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, dir_okay=True, readable=True)
    ],
) -> None:
    from .plotting import plot_ppo_history, plot_pretrain_history

    created: list[Path] = []

    if (run_dir / "pretrain_history.parquet").exists():
        created.append(plot_pretrain_history(run_dir))
    if (run_dir / "ppo_history.parquet").exists():
        created.extend(plot_ppo_history(run_dir))

    for path in created:
        typer.echo(f"wrote: {path}")


if __name__ == "__main__":
    app()
