from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from .benchmark import run_benchmark, summarize_benchmark
from .cases.generate import generate_cases
from .cases.networks import list_supported_networks
from .config import load_config
from .learning.dataset import WarmStartDataset
from .learning.models import VoltageWarmStartActor, VoltageWarmStartCritic, load_actor
from .learning.ppo import PPOConfig
from .learning.train import pretrain_actor_supervised, train_ppo
from .plotting import plot_ppo_history, plot_pretrain_history
from .utils import make_run_name, print_rich, write_parquet

app = typer.Typer(add_completion=False, no_args_is_help=True)

DEFAULT_CONFIG_PATH = Path("configs/default.toml")
DEFAULT_BENCHMARK_OUT = Path("outputs/results")
DEFAULT_ACTOR_OUT = Path("outputs/models/")
DEFAULT_SEARCH_VALIDATE_OUT = Path("outputs/search/")


@app.command("list-networks", help="List the bundled benchmark networks.")
def list_networks_command() -> None:
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
    cfg = load_config(config)
    info = generate_cases(cfg)
    typer.echo(json.dumps(info, indent=2))


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
    pretrain: bool = True,
) -> None:
    import polars as pl
    import torch

    from .config import Config
    from .learning.models import save_actor

    if config_from_actor is not None:
        checkpoint = torch.load(config_from_actor, map_location="cpu")
        cfg = Config.model_validate(checkpoint["full_cfg"])
    else:
        cfg = load_config(config)

    dataset = WarmStartDataset(
        cfg.dataset_root,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
    )

    actor = VoltageWarmStartActor(
        in_channels=dataset.input_channels,
        hidden_channels=cfg.model.hidden_channels,
        dropout=cfg.model.dropout,
        action_std_init=cfg.model.action_std_init,
        device_type_embedding_dim=cfg.model.device_type_embedding_dim,
    )
    pretrain_history = None
    if pretrain:
        pretrain_history = pretrain_actor_supervised(
            actor,
            dataset,
            epochs=cfg.pretrain.epochs,
            batch_size=cfg.pretrain.batch_size,
            learning_rate=cfg.pretrain.learning_rate,
            weight_decay=cfg.pretrain.weight_decay,
        )
    critic = VoltageWarmStartCritic(
        in_channels=dataset.input_channels,
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
        dataset,
        solver=cfg.solver,
        ppo_config=ppo_model_config,
        updates=cfg.ppo.updates,
        rollout_size=cfg.ppo.rollout_size,
        seed=cfg.ppo.seed,
    )
    run_name = make_run_name(cfg)
    out = out / run_name
    out.mkdir(parents=True, exist_ok=True)
    save_actor(
        actor=agent.actor,
        input_channels=dataset.input_channels,
        cfg=cfg,
        history=ppo_history,
        out=out,
    )
    if pretrain_history:
        write_parquet(pl.DataFrame(pretrain_history), out / "pretrain_history.parquet")
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
    import random
    from itertools import product

    import torch
    from tqdm import tqdm

    from .learning.models import save_actor
    from .learning.train import evaluate_policy, evaluate_supervised_actor

    @dataclass(slots=True)
    class BestPretrain:
        hidden_channels: int
        score: float
        actor_state_dict: dict[str, torch.Tensor]

    @dataclass(slots=True)
    class BestPPO:
        success_rate: float
        mean_solve_time: float
        mean_reward: float
        search_config: dict[str, float | int]

    def to_cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}

    def is_better(
        success_rate: float,
        mean_solve_time: float,
        current_best: BestPPO | None,
    ) -> bool:
        if current_best is None:
            return True
        if success_rate != current_best.success_rate:
            return success_rate > current_best.success_rate
        return mean_solve_time < current_best.mean_solve_time

    cfg = load_config(config)
    dataset_root = cfg.dataset_root
    full_dataset = WarmStartDataset(
        dataset_root=dataset_root,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
    )

    case_ids = [full_dataset.case_metadata(i).case_id for i in range(len(full_dataset))]

    rng = random.Random(cfg.ppo.seed)
    rng.shuffle(case_ids)

    split_index = max(1, int(0.8 * len(case_ids)))
    if split_index >= len(case_ids):
        split_index = len(case_ids) - 1

    train_case_ids = case_ids[:split_index]
    val_case_ids = case_ids[split_index:]

    train_dataset = WarmStartDataset(
        dataset_root,
        case_ids=train_case_ids,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
    )
    val_dataset = WarmStartDataset(
        dataset_root,
        case_ids=val_case_ids,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = make_run_name(cfg)
    output_dir = out / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_pretrain: BestPretrain | None = None

    for hidden_channels in tqdm(
        cfg.search.hidden_channels, desc="Pretraining search", unit="config", position=2
    ):
        actor = VoltageWarmStartActor(
            in_channels=full_dataset.input_channels,
            hidden_channels=hidden_channels,
            dropout=cfg.model.dropout,
            action_std_init=cfg.model.action_std_init,
            device_type_embedding_dim=cfg.model.device_type_embedding_dim,
        )

        pretrain_actor_supervised(
            actor,
            dataset=train_dataset,
            epochs=cfg.pretrain.epochs,
            batch_size=cfg.pretrain.batch_size,
            learning_rate=cfg.pretrain.learning_rate,
            weight_decay=cfg.pretrain.weight_decay,
        )

        val_loss = evaluate_supervised_actor(
            val_dataset,
            actor,
            cfg.pretrain.batch_size,
            device,
        )

        if best_pretrain is None or val_loss < best_pretrain.score:
            best_pretrain = BestPretrain(
                hidden_channels=hidden_channels,
                score=val_loss,
                actor_state_dict=to_cpu_state_dict(actor),
            )

    if best_pretrain is None:
        raise ValueError("No pretrained actor.")

    typer.echo(
        f"Best hidden channel dimension during pretraining: "
        f"{best_pretrain.hidden_channels} (val_loss={best_pretrain.score:.6f})"
    )

    best_final: BestPPO | None = None

    ppo_combinations = list(
        product(
            cfg.search.ppo_learning_rate,
            cfg.search.ppo_entropy_weight,
            cfg.search.nonconvergence_penalty,
        )
    )

    for learning_rate, entropy_weight, nonconvergence_penalty in tqdm(
        ppo_combinations, desc="PPO search", unit="config", position=2
    ):
        actor = VoltageWarmStartActor(
            in_channels=full_dataset.input_channels,
            hidden_channels=best_pretrain.hidden_channels,
            dropout=cfg.model.dropout,
            action_std_init=cfg.model.action_std_init,
            device_type_embedding_dim=cfg.model.device_type_embedding_dim,
        )
        actor.load_state_dict(best_pretrain.actor_state_dict)

        critic = VoltageWarmStartCritic(
            in_channels=full_dataset.input_channels,
            hidden_channels=best_pretrain.hidden_channels,
            dropout=cfg.model.dropout,
        )

        ppo_model_config = PPOConfig(
            learning_rate=learning_rate,
            clip_ratio=cfg.ppo.clip_ratio,
            value_loss_weight=cfg.ppo.value_loss_weight,
            entropy_weight=entropy_weight,
            ppo_epochs=cfg.ppo.ppo_epochs,
            max_grad_norm=cfg.ppo.max_grad_norm,
        )

        agent, history = train_ppo(
            actor,
            critic,
            train_dataset,
            solver=cfg.solver,
            ppo_config=ppo_model_config,
            updates=cfg.ppo.updates,
            rollout_size=cfg.ppo.rollout_size,
            nonconvergence_penalty=nonconvergence_penalty,
            seed=cfg.ppo.seed,
        )

        metrics = evaluate_policy(
            val_dataset,
            agent.actor,
            cfg.solver,
            nonconvergence_penalty=nonconvergence_penalty,
            device=device,
        )

        if is_better(metrics.success_rate, metrics.mean_solve_time, best_final):
            best_final = BestPPO(
                success_rate=metrics.success_rate,
                mean_solve_time=metrics.mean_solve_time,
                mean_reward=metrics.mean_reward,
                search_config={
                    "hidden_channels": best_pretrain.hidden_channels,
                    "ppo_learning_rate": learning_rate,
                    "ppo_entropy_weight": entropy_weight,
                    "nonconvergence_penalty": nonconvergence_penalty,
                },
            )

            best_cfg = cfg.model_copy(
                update={
                    "model": cfg.model.model_copy(
                        update={
                            "hidden_channels": int(best_pretrain.hidden_channels),
                        }
                    ),
                    "ppo": cfg.ppo.model_copy(
                        update={
                            "learning_rate": float(learning_rate),
                            "entropy_weight": float(entropy_weight),
                            "nonconvergence_penalty": float(nonconvergence_penalty),
                        }
                    ),
                }
            )

            save_actor(
                actor=agent.actor,
                input_channels=full_dataset.input_channels,
                cfg=best_cfg,
                history=history,
                out=output_dir,
            )

            typer.echo(
                f"New best config saved:\n"
                f"  hidden_channels:         {best_final.search_config['hidden_channels']}\n"
                f"  ppo_learning_rate:       {best_final.search_config['ppo_learning_rate']}\n"
                f"  ppo_entropy_weight:      {best_final.search_config['ppo_entropy_weight']}\n"
                f"  nonconvergence_penalty:  {best_final.search_config['nonconvergence_penalty']}\n"
                f"  success_rate:            {best_final.success_rate:.2%}\n"
                f"  mean_solve_time:         {best_final.mean_solve_time:.6f}\n"
                f"  mean_reward:             {best_final.mean_reward:.6f}\n"
                f"saved to: {output_dir}"
            )

    if best_final is None:
        raise ValueError("No PPO config was selected.")

    typer.echo(
        f"Best final config:\n"
        f"  hidden_channels:         {best_final.search_config['hidden_channels']}\n"
        f"  ppo_learning_rate:       {best_final.search_config['ppo_learning_rate']}\n"
        f"  ppo_entropy_weight:      {best_final.search_config['ppo_entropy_weight']}\n"
        f"  nonconvergence_penalty:  {best_final.search_config['nonconvergence_penalty']}\n"
        f"  success_rate:            {best_final.success_rate:.2%}\n"
        f"  mean_solve_time:         {best_final.mean_solve_time:.6f}\n"
        f"  mean_reward:             {best_final.mean_reward:.6f}\n"
        f"saved to: {output_dir}"
    )


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
) -> None:
    import torch

    from .config import Config
    from .learning.models import WarmStartPredictor
    from .plotting import plot_benchmark_results

    checkpoint = torch.load(actor_path, map_location="cpu")
    cfg = Config.model_validate(checkpoint["full_cfg"])

    actor = load_actor(actor_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = WarmStartPredictor(model=actor, device=device)

    network_names = tuple(cfg.benchmark.network_names) or (cfg.data.network_name,)
    run_name = make_run_name(cfg)
    result_path = out / f"{run_name}_benchmark.parquet"

    frame = run_benchmark(
        network_names=network_names,
        n_cases=cfg.benchmark.n_cases,
        seed=cfg.benchmark.seed,
        load_scale_min=cfg.perturb.load_scale_min,
        load_scale_max=cfg.perturb.load_scale_max,
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
    created: list[Path] = []

    if (run_dir / "pretrain_history.parquet").exists():
        created.append(plot_pretrain_history(run_dir))
    if (run_dir / "ppo_history.parquet").exists():
        created.extend(plot_ppo_history(run_dir))

    for path in created:
        typer.echo(f"wrote: {path}")


if __name__ == "__main__":
    app()
