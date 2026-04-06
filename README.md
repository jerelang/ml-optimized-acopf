# ml-assisted-acopf

Generate solved AC-OPF cases, train a GNN warm-start actor with supervised pretraining and PPO, and run a small validation search.

## Supported networks

- `case14`
- `case30`
- `case57`
- `case118`
- `case300`

## Install

Main dependencies:

```bash
uv sync
````

With development and learning dependencies:

```bash
uv sync --group dev
```

## Workflow

Generate a solved dataset:

```bash
uv run ml_acopf generate-cases --config configs/default.toml```
````

This writes the dataset to:

```bash
data/<data.name>/baseline/
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

Run a basic 2-phase grid search for finding good hyperparameters:

```bash
uv run ml_acopf search \
  --config configs/default.toml \
  --out outputs/search
```

This writes the best searched actor to:

```bash
outputs/search/<run_name>/agent_ppo.pt
````

Benchmark flat, PF, and actor predicted warm starts:

```bash
uv run ml_acopf benchmark \
  outputs/models/<run_name>/agent_ppo.pt \
  --config configs/default.toml \
  --out outputs/results
```

This writes the benchmark result to:

```bash
outputs/results/<run_name>_benchmark.parquet
```

## Data layout

Case generation writes to `data/<data.name>/baseline/`:

- `cases.parquet`: solved case metadata
- `load_inputs.parquet`: per-load perturbed inputs
- `bus_targets.parquet`: per-bus solved OPF targets
- `dispatch_targets.parquet`: per-device solved OPF targets
- `buses_static.parquet`: static bus graph features
- `edges_static.parquet`: static edge graph features