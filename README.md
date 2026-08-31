# ML-Optimized AC-OPF Warm Starts

A modular implementation of the **PPO + GNN warm-start method** for **AC optimal power flow (AC-OPF)** from *Azad Deihim et al., "Initial estimate of AC optimal power flow with graph neural networks" (Electric Power Systems Research, 2024)*, ported to the **pandapower + PandaModels/PowerModels (Ipopt)** solver stack instead of the original PyPower.

On **PGLib-OPF `case118`** (5000 perturbed cases), the learned warm start cuts mean end-to-end solve time from **2.95 s (flat start) to 1.06 s**, with much better tail behaviour than a power-flow warm start.

## Method notes

Following the paper, both actor and critic are GNNs; the actor predicts warm-start values for voltage magnitude, voltage angle, and active/reactive power, which are handed to the AC-OPF solver. One observation from reimplementing it: although the paper trains with PPO, the underlying problem is effectively a **contextual bandit** rather than a sequential RL task, since each episode is a single step on an independently sampled case. There is no temporal reward dependency, and the discount factor has no effect in this setting.

Differences to the original: this implementation uses the **PandaModels/PowerModels + Ipopt** stack via pandapower (the paper used PyPower), adds optional **supervised pretraining**, and wraps everything in a configurable, tested CLI pipeline.

## What is included

- **PGLib-OPF** case import through pandapower (`case14`, `case30`, `case118`, `case300`, `case118_api`, `case118_sad`)
- Generation of perturbed, solved AC-OPF datasets (parquet)
- GNN + PPO warm-start training of the primal variables, optional supervised pretraining
- A reproducible benchmark structure comparing **flat start**, **power-flow warm start**, and the **learned warm start**
- Training and benchmark plotting

## Benchmark snapshot

Paper-inspired PPO + GNN settings (configs: [`configs/case118_ppo.toml`](configs/case118_ppo.toml), [`configs/case14_ppo.toml`](configs/case14_ppo.toml)):

| Network | Method | Mean total time [s] | P90 total time [s] | Median total time [s] |
|---|---|---:|---:|---:|
| case118 | Flat | 2.950 | 3.640 | 3.259 |
| case118 | PF | 2.582 | 8.109 | 1.226 |
| case118 | GNN + PPO | 1.060 | 1.652 | 0.938 |
| case14 | Flat | 0.094 | 0.136 | 0.089 |
| case14 | PF | 0.068 | 0.087 | 0.067 |
| case14 | GNN + PPO | 0.070 | 0.098 | 0.065 |

Timings measured on an idle desktop (Fedora 44, Ryzen 7 7700X, 16 GB RAM, RTX 4060 Ti 16 GB).

### Solver-time distribution
![case118 ECDF](docs/paper_case118_ppo/benchmark/118_hist.svg)

On 5000 perturbed `case118` instances the learned warm start reduces end-to-end solve time relative to both baselines. The gain is most visible in the tail: PF is competitive on median time but has substantially worse **P90** behaviour than the learned method. More figures, training diagnostics and benchmark summaries: [docs](docs).

## Installation

```bash
uv sync            # main dependencies (Julia part resolves on first run)
uv sync --group dev
```

## Run with Docker

The image includes Python and the Julia setup:

```bash
docker build -t ml-acopf .
docker run --rm ml-acopf list-networks
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" \
  ml-acopf generate-cases -c configs/case14.toml
```

Note: the image installs the locked PyTorch build (CUDA wheels) and is correspondingly large.

## Basic workflow

```bash
uv run ml_acopf list-networks
uv run ml_acopf generate-cases --config configs/default.toml
uv run ml_acopf search    --config configs/default.toml --out outputs/search
uv run ml_acopf train     --config configs/default.toml --out outputs/models [--pretrain]
uv run ml_acopf benchmark outputs/models/<run_name>/agent_ppo.pt --out outputs/results
uv run ml_acopf plot-training  outputs/models/<run_name>
uv run ml_acopf plot-benchmark outputs/results/<run_name>/<run_name>_benchmark.parquet
```

## Output layout

Case generation writes to `data/<data.name>/baseline/`: `cases.parquet`, `attempted_cases.parquet`, `load_inputs.parquet`, `attempted_load_inputs.parquet`, `bus_targets.parquet`, `dispatch_targets.parquet`, `buses_static.parquet`, `edges_static.parquet`, `device_metadata.parquet`.

Training writes to `outputs/models/<run_name>/` (`agent_ppo.pt`, `ppo_history.parquet`, optional `pretrain_history.parquet`, plots); search runs to `outputs/search/<run_name>/`; benchmarks to `outputs/results/<run_name>/` (`*_benchmark.parquet`, `*_benchmark_summary.parquet`, plots).

## Reference

Azad Deihim et al., *Initial estimate of AC optimal power flow with graph neural networks*, Electric Power Systems Research, 2024.
