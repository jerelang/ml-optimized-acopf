from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool

from .models import VoltageWarmStartActor, VoltageWarmStartCritic, WarmStartAction


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    ppo_epochs: int = 4
    max_grad_norm: float = 1.0


@dataclass(frozen=True, slots=True)
class RolloutTransition:
    data: Data
    action_bus: Tensor
    action_device: Tensor
    log_prob: float
    value: float
    reward: float


@dataclass(slots=True)
class RolloutBuffer:
    items: list[RolloutTransition] = field(default_factory=list)

    def add(
        self,
        data: Data,
        action_bus: Tensor,
        action_device: Tensor,
        log_prob: float,
        value: float,
        reward: float,
    ) -> None:
        self.items.append(
            RolloutTransition(
                data=data,
                action_bus=action_bus,
                action_device=action_device,
                log_prob=log_prob,
                value=value,
                reward=reward,
            )
        )

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)


class PPOAgent:
    def __init__(
        self,
        actor: VoltageWarmStartActor,
        critic: VoltageWarmStartCritic,
        config: PPOConfig,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.config = config
        self.device = device or torch.device("cpu")

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
        )

    def act(self, data: Data, *, stochastic: bool = True) -> tuple[WarmStartAction, float, float]:
        self.actor.eval()
        self.critic.eval()

        graph = data.to(self.device)
        if (
            graph.x is None
            or graph.edge_index is None
            or graph.device_bus_index is None
            or graph.device_type_id is None
        ):
            raise ValueError("Data Tensors must not be None")
        with torch.no_grad():
            dist = self.actor.distribution(
                graph.x,
                graph.edge_index,
                graph.device_bus_index,
                graph.device_type_id,
            )
            if stochastic:
                bus_action = dist.bus.sample()
                if dist.device is None:
                    device_action = torch.empty((0, 2), dtype=graph.x.dtype, device=self.device)
                else:
                    device_action = dist.device.sample()
            else:
                mean_action = self.actor(
                    graph.x,
                    graph.edge_index,
                    graph.device_bus_index,
                    graph.device_type_id,
                )
                bus_action = mean_action.bus
                device_action = mean_action.device

            bus_action = bus_action.clamp(0.0, 1.0)
            device_action = device_action.clamp(0.0, 1.0)

            log_prob = dist.bus.log_prob(bus_action).sum(dim=-1).sum()
            if dist.device is not None and device_action.numel() > 0:
                log_prob = log_prob + dist.device.log_prob(device_action).sum(dim=-1).sum()

            value = float(self.critic(graph.x, graph.edge_index).squeeze().item())

        return (
            WarmStartAction(bus=bus_action.cpu(), device=device_action.cpu()),
            float(log_prob.item()),
            value,
        )

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        if not buffer.items:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "mean_reward": 0.0,
            }

        data_list = [item.data for item in buffer.items]
        batch = Batch.from_data_list(data_list).to(self.device)  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

        actions_bus = torch.cat([item.action_bus for item in buffer.items], dim=0).to(self.device)
        if any(item.action_device.numel() > 0 for item in buffer.items):
            actions_device = torch.cat(
                [item.action_device for item in buffer.items],
                dim=0,
            ).to(self.device)
        else:
            actions_device = torch.empty((0, 2), dtype=torch.float32, device=self.device)

        old_log_probs = torch.tensor(
            [item.log_prob for item in buffer.items],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            [item.reward for item in buffer.items],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.tensor(
            [item.value for item in buffer.items],
            dtype=torch.float32,
            device=self.device,
        )

        advantages = rewards - old_values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.actor.train()
        self.critic.train()

        loss_value = 0.0
        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0

        for _ in range(self.config.ppo_epochs):
            dist = self.actor.distribution(
                batch.x,
                batch.edge_index,
                batch.device_bus_index,
                batch.device_type_id,
            )

            bus_node_log_prob = dist.bus.log_prob(actions_bus).sum(dim=-1, keepdim=True)
            bus_graph_log_prob = global_add_pool(
                bus_node_log_prob,
                batch.batch,
                size=old_log_probs.numel(),
            ).squeeze(-1)

            bus_node_entropy = dist.bus.entropy().sum(dim=-1, keepdim=True)
            bus_graph_entropy = global_add_pool(
                bus_node_entropy,
                batch.batch,
                size=old_log_probs.numel(),
            ).squeeze(-1)

            if dist.device is not None and actions_device.numel() > 0:
                device_graph_index = batch.batch[batch.device_bus_index]

                device_node_log_prob = dist.device.log_prob(actions_device).sum(
                    dim=-1, keepdim=True
                )
                device_graph_log_prob = global_add_pool(
                    device_node_log_prob,
                    device_graph_index,
                    size=old_log_probs.numel(),
                ).squeeze(-1)

                device_node_entropy = dist.device.entropy().sum(dim=-1, keepdim=True)
                device_graph_entropy = global_add_pool(
                    device_node_entropy,
                    device_graph_index,
                    size=old_log_probs.numel(),
                ).squeeze(-1)
            else:
                device_graph_log_prob = torch.zeros_like(bus_graph_log_prob)
                device_graph_entropy = torch.zeros_like(bus_graph_entropy)

            graph_log_prob = bus_graph_log_prob + device_graph_log_prob
            graph_entropy = bus_graph_entropy + device_graph_entropy

            values = self.critic(batch.x, batch.edge_index, batch.batch)

            ratios = torch.exp(graph_log_prob - old_log_probs)
            surrogate_1 = ratios * advantages
            surrogate_2 = (
                torch.clamp(
                    ratios,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                )
                * advantages
            )

            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            value_loss = F.mse_loss(values, rewards)
            entropy_bonus = graph_entropy.mean()

            loss = (
                policy_loss
                + self.config.value_loss_weight * value_loss
                - self.config.entropy_weight * entropy_bonus
            )

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

            loss_value = float(loss.item())
            policy_loss_value = float(policy_loss.item())
            value_loss_value = float(value_loss.item())
            entropy_value = float(entropy_bonus.item())

        return {
            "loss": loss_value,
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
            "mean_reward": float(rewards.mean().item()),
        }
