from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from .benchmark import summarize_benchmark

sns.set_theme(style="darkgrid", palette="tab10")


def plot_pretrain_history(run_dir: Path | str) -> Path:
    run_dir = Path(run_dir)
    history = pl.read_parquet(run_dir / "pretrain_history.parquet").sort("epoch")

    output_path = run_dir / "pretrain_loss.png"

    fig, ax = plt.subplots()
    sns.lineplot(
        x=history["epoch"].to_list(),
        y=history["loss"].to_list(),
        ax=ax,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Supervised pretraining loss")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_ppo_history(run_dir: Path | str) -> list[Path]:
    run_dir = Path(run_dir)
    history = pl.read_parquet(run_dir / "ppo_history.parquet").sort("update")

    outputs: list[Path] = []

    for column, ylabel, filename in [
        ("mean_reward", "Mean reward", "ppo_mean_reward.png"),
        ("success_rate", "Success rate", "ppo_success_rate.png"),
        ("mean_solve_time", "Mean solve time [s]", "ppo_mean_solve_time.png"),
    ]:
        if column not in history.columns:
            continue

        output_path = run_dir / filename
        fig, ax = plt.subplots()
        sns.lineplot(
            x=history["update"].to_list(),
            y=history[column].to_list(),
            ax=ax,
        )
        ax.set_xlabel("Update")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        outputs.append(output_path)

    return outputs


def plot_benchmark_results(benchmark_file: Path | str) -> list[Path]:
    benchmark_path = Path(benchmark_file)
    frame = pl.read_parquet(benchmark_path)
    if frame.is_empty():
        return []

    summary = summarize_benchmark(frame)
    outputs: list[Path] = []

    success_rate_path = benchmark_path.with_name(f"{benchmark_path.stem}_success_rate.png")
    _plot_grouped_bar(
        summary,
        value_column="success_rate",
        ylabel="Success rate",
        title="Benchmark success rate by method",
        output_path=success_rate_path,
    )
    outputs.append(success_rate_path)

    total_mean_time_path = benchmark_path.with_name(f"{benchmark_path.stem}_total_time_mean.png")
    _plot_grouped_bar(
        summary,
        value_column="total_time_s_mean",
        ylabel="Mean total time [s]",
        title="Benchmark mean total time by method",
        output_path=total_mean_time_path,
    )
    outputs.append(total_mean_time_path)

    total_p90_time_path = benchmark_path.with_name(f"{benchmark_path.stem}_total_time_p90.png")
    _plot_grouped_bar(
        summary,
        value_column="total_time_s_p90",
        ylabel="P90 total time [s]",
        title="Benchmark P90 total time by method",
        output_path=total_p90_time_path,
    )
    outputs.append(total_p90_time_path)

    total_median_time_path = benchmark_path.with_name(f"{benchmark_path.stem}total_time_median.png")
    _plot_grouped_bar(
        summary,
        value_column="total_time_s_median",
        ylabel="Median total time [s]",
        title="Benchmark median total time by method",
        output_path=total_median_time_path,
    )
    outputs.append(total_median_time_path)

    boxplot_path = benchmark_path.with_name(f"{benchmark_path.stem}_total_time_boxplot.png")
    _plot_boxplot_by_group(
        frame,
        value_column="total_time_s",
        ylabel="total time [s]",
        title="Benchmark total time distribution",
        output_path=boxplot_path,
    )
    outputs.append(boxplot_path)

    violin_path = benchmark_path.with_name(f"{benchmark_path.stem}_total_time_violinplot.png")
    _plot_boxplot_by_group(
        frame,
        value_column="total_time_s",
        ylabel="total time [s]",
        title="Benchmark total time distribution",
        plot_type="violin",
        output_path=violin_path,
    )
    outputs.append(violin_path)

    dist_path = benchmark_path.with_name(f"{benchmark_path.stem}_total_time_dist.png")
    _plot_boxplot_by_group(
        frame,
        value_column="total_time_s",
        ylabel="total time [s]",
        title="Benchmark total time histogram",
        output_path=dist_path,
    )
    outputs.append(dist_path)
    return outputs


def _plot_grouped_bar(
    summary: pl.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    df = summary.to_pandas()
    fig, ax = plt.subplots()
    sns.barplot(
        data=df,
        x="method",
        y=value_column,
        ax=ax,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_boxplot_by_group(
    frame: pl.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    plot_type: Literal["box", "violin"] = "box",
    output_path: Path,
) -> bool:
    df = frame.select(["network_name", "method", value_column]).drop_nulls().to_pandas()
    if df.empty:
        return False

    df["group"] = df["network_name"] + "\n" + df["method"]

    fig, ax = plt.subplots()
    if plot_type == "box":
        sns.boxplot(data=df, x="group", y=value_column, ax=ax, showmeans=True)
    else:
        assert plot_type == "violin"
        sns.violinplot(data=df, x="group", y=value_column, ax=ax, cut=0)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _plot_dist(
    summary: pl.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    df = summary.to_pandas()
    fig, ax = plt.subplots()
    sns.barplot(
        data=df,
        x="method",
        y=value_column,
        ax=ax,
    )
    sns.displot(df, x=value_column, hue="method", ax=ax, fill=True, bins=100, log_scale=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
