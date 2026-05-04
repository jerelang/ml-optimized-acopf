from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..cases.generate import apply_load_inputs
from ..cases.networks import network_template
from ..config import SolverConfig
from ..solver.io import WarmStartPayload
from ..solver.solver import solve_ac_opf
from .dataset import (
    WarmStartDataset,
    build_bus_warmstart_frame,
    build_device_warmstart_frame,
    denormalize_bus_values,
    denormalize_device_values,
)
from .models import VoltageWarmStartActor, VoltageWarmStartCritic, WarmStartAction
from .ppo import PPOAgent, PPOConfig, RolloutBuffer


@dataclass(frozen=True, slots=True)
class WarmStartStep:
    reward: float
    success: bool
    wall_time_s: float
    solver_time_s: float | None
    iterations: int | None
    objective: float | None
    error: str | None


def evaluate_warmstart_action(
    dataset: WarmStartDataset,
    solver: SolverConfig,
    *,
    case_index: int,
    normalized_action: WarmStartAction,
    nonconvergence_penalty: float = 10.0,
) -> WarmStartStep:
    metadata = dataset.case_metadata(case_index)
    graph = dataset[case_index]

    net = copy.deepcopy(network_template(metadata.network_name))
    apply_load_inputs(net, dataset.load_inputs_for_case(metadata.case_id))

    bus_values = denormalize_bus_values(
        normalized_action.bus.detach().cpu(),
        graph.vm_lower.cpu(),
        graph.vm_upper.cpu(),
        graph.va_lower.cpu(),
        graph.va_upper.cpu(),
    )
    device_values = denormalize_device_values(
        normalized_action.device.detach().cpu(),
        graph.device_p_lower.cpu(),
        graph.device_p_upper.cpu(),
        graph.device_q_lower.cpu(),
        graph.device_q_upper.cpu(),
    )

    bus_frame = build_bus_warmstart_frame(metadata.case_id, graph.bus_id.cpu(), bus_values)
    device_frame = build_device_warmstart_frame(
        metadata.case_id,
        graph.device_type_id.cpu(),
        graph.device_element_id.cpu(),
        device_values,
    )

    stats = solve_ac_opf(
        net,
        solver,
        warmstart=WarmStartPayload(bus=bus_frame, device=device_frame),
    )

    solve_time = stats.solver_time_s if stats.solver_time_s is not None else stats.wall_time_s
    reward = -solve_time if stats.success else -nonconvergence_penalty

    return WarmStartStep(
        reward=reward,
        success=stats.success,
        wall_time_s=stats.wall_time_s,
        solver_time_s=stats.solver_time_s,
        iterations=stats.iterations,
        objective=stats.objective,
        error=stats.error,
    )


def pretrain_actor_supervised(
    actor: VoltageWarmStartActor,
    dataset: WarmStartDataset,
    device: torch.device,
    *,
    epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> list[dict[str, float]]:
    actor.to(device)

    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loader = DataLoader(
        dataset=dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=True,
    )

    history: list[dict[str, float]] = []
    epoch_progress = tqdm(range(epochs), desc="Pretraining", unit="epoch", position=1, leave=True)
    for epoch in epoch_progress:
        actor.train()
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            loss = _evaluate_batch(actor, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1
        epoch_loss = running_loss / max(n_batches, 1)
        history.append({"epoch": float(epoch), "loss": epoch_loss})
        epoch_progress.set_postfix(loss=f"{epoch_loss:.4f}")

    return history


def evaluate_supervised_actor(
    dataset: WarmStartDataset, actor: VoltageWarmStartActor, batch_size: int, device: torch.device
) -> float:
    validation_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    actor.eval()
    actor.to(device)
    total_loss = 0.0
    number_batches = 0
    with torch.no_grad():
        for batch in validation_loader:
            loss = _evaluate_batch(actor, batch, device)
            total_loss += float(loss.item())
            number_batches += 1
    return total_loss / max(number_batches, 1)


def _evaluate_batch(
    actor: VoltageWarmStartActor,
    batch: Batch,
    device: torch.device,
) -> torch.Tensor:
    batch = batch.to(device)
    prediction = actor(
        batch.x,
        batch.edge_index,
        batch.device_bus_index,
        batch.device_type_id,
    )

    losses = [F.mse_loss(prediction.bus, batch.y_bus)]
    if hasattr(batch, "y_device") and prediction.device.numel() > 0:
        losses.append(F.mse_loss(input=prediction.device, target=batch.y_device))

    return torch.stack(losses).mean()


def train_ppo(
    actor: VoltageWarmStartActor,
    critic: VoltageWarmStartCritic,
    dataset: WarmStartDataset,
    solver: SolverConfig,
    device: torch.device,
    *,
    ppo_config: PPOConfig | None = None,
    updates: int = 100,
    rollout_size: int = 32,
    nonconvergence_penalty: float = 10.0,
    seed: int = 0,
) -> tuple[PPOAgent, list[dict[str, float]]]:
    agent = PPOAgent(
        actor=actor,
        critic=critic,
        config=ppo_config or PPOConfig(),
        device=device,
    )

    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    update_progress = tqdm(range(updates), desc="PPO Updates", unit="updates", position=1)
    for update_index in update_progress:
        buffer = RolloutBuffer()
        reward_values: list[float] = []
        success_values: list[float] = []
        solve_times: list[float] = []

        rollout_bar = tqdm(
            range(rollout_size), desc="Rollout", unit="step", leave=False, position=2
        )
        for _ in rollout_bar:
            case_index = rng.randrange(len(dataset))
            graph = dataset[case_index]

            action, log_prob, value = agent.act(graph, stochastic=True)
            step = evaluate_warmstart_action(
                dataset,
                solver,
                case_index=case_index,
                normalized_action=action,
                nonconvergence_penalty=nonconvergence_penalty,
            )

            buffer.add(
                data=graph,
                action_bus=action.bus,
                action_device=action.device,
                log_prob=log_prob,
                value=value,
                reward=step.reward,
            )

            reward_values.append(step.reward)
            success_values.append(1.0 if step.success else 0.0)
            solve_times.append(step.solver_time_s or step.wall_time_s)
            rollout_bar.set_postfix(
                reward=f"{step.reward:.3f}",
                success=step.success,
            )
            agent.decay_action_std()

        update_stats = agent.update(buffer)
        mean_reward = sum(reward_values) / max(len(reward_values), 1)
        success_rate = sum(success_values) / max(len(success_values), 1)
        mean_solve_time = sum(solve_times) / max(len(solve_times), 1)

        history.append(
            {
                "update": float(update_index),
                "mean_reward": mean_reward,
                "success_rate": success_rate,
                "mean_solve_time": mean_solve_time,
                "action_std": agent.current_action_std,
                **update_stats,
            }
        )
        update_progress.set_postfix(
            reward=f"{mean_reward:.3f}",
            success=f"{success_rate:.1%}",
            solve_time=f"{mean_solve_time:.2f}s",
        )

    return agent, history


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    mean_reward: float
    success_rate: float
    mean_solve_time: float


def evaluate_policy(
    dataset: WarmStartDataset,
    actor: VoltageWarmStartActor,
    solver: SolverConfig,
    device: torch.device,
    *,
    nonconvergence_penalty: float = 10.0,
) -> ValidationMetrics:
    actor = actor.to(device)
    actor.eval()

    rewards: list[float] = []
    successes: list[float] = []
    solve_times: list[float] = []

    with torch.no_grad():
        for case_index in range(len(dataset)):
            graph = dataset[case_index].to(device)

            action = actor(
                graph.x,
                graph.edge_index,
                graph.device_bus_index,
                graph.device_type_id,
            )

            step = evaluate_warmstart_action(
                dataset,
                solver,
                case_index=case_index,
                normalized_action=WarmStartAction(
                    bus=action.bus.cpu(),
                    device=action.device.cpu(),
                ),
                nonconvergence_penalty=nonconvergence_penalty,
            )

            rewards.append(step.reward)
            successes.append(1.0 if step.success else 0.0)
            solve_times.append(step.solver_time_s or step.wall_time_s)

    return ValidationMetrics(
        mean_reward=sum(rewards) / max(len(rewards), 1),
        success_rate=sum(successes) / max(len(successes), 1),
        mean_solve_time=sum(solve_times) / max(len(solve_times), 1),
    )
