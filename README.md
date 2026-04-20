# ml-assisted-acopf

Generate solved AC-OPF cases, train a GNN warm-start actor with supervised pretraining and PPO, and benchmark flat, PF, and actor warm starts.

## Supported PGLIB-OPF networks

- `case14`
- `case30`
- `case57`
- `case118`
- `case300`
- `case118.api`
- `case118.sad`

## Install

Main dependencies:

```bash
uv sync
```

With development dependencies:

```bash
uv sync --group dev
```

## Workflow

Generate a solved dataset:

```bash
uv run ml_acopf generate-cases --config configs/default.toml
```

This writes the dataset to:

```bash
data/<data.name>/baseline/
```

Run a basic 2-phase grid search for finding good hyperparameters:

```bash
uv run ml_acopf search \
  --config configs/default.toml \
  --out outputs/search
```

This writes the best searched actor to:

```bash
outputs/search/<run_name>/agent_ppo.pt
```

Pretrain the actor GNN and then train with PPO on the generated dataset:

```bash
uv run ml_acopf train \
  --config configs/default.toml \
  --out outputs/models
```

This writes the actor to:

```bash
outputs/models/<run_name>/agent_ppo.pt
```
Benchmark flat, PF, and actor predicted warm starts:

```bash
uv run ml_acopf benchmark \
  outputs/models/<run_name>/agent_ppo.pt \
  --out outputs/results
```

This writes the benchmark outputs to:

```bash
outputs/results/<run_name>_benchmark.parquet
outputs/results/<run_name>_benchmark_summary.parquet
outputs/results/<run_name>_benchmark_success_rate.png
outputs/results/<run_name>_benchmark_total_time_mean.png
outputs/results/<run_name>_benchmark_total_time_boxplot.png
```

Create training plots for a saved run:

```bash
uv run ml_acopf plot-training outputs/models/<run_name>
```

Create benchmark plots for a saved benchmark file:

```bash
uv run ml_acopf plot-benchmark outputs/results/<run_name>_benchmark.parquet
```

## Config notes

- `[normalization].max_voltage_angle_deg` sets one fixed symmetric angle range `[-x, x]` for voltage-angle normalization.
- `[benchmark]` controls fresh-case benchmarking and optional plot generation.

## Data layout

Case generation writes to `data/<data.name>/baseline/`:

- `cases.parquet`: solved case metadata
- `load_inputs.parquet`: per-load perturbed inputs
- `bus_targets.parquet`: per-bus solved OPF targets
- `dispatch_targets.parquet`: per-device solved OPF targets
- `buses_static.parquet`: static bus graph features
- `edges_static.parquet`: static edge graph features
