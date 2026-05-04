from __future__ import annotations

import torch
from torch_geometric.data import Data

from ml_acopf.learning.models import VoltageWarmStartActor, VoltageWarmStartCritic
from ml_acopf.learning.ppo import PPOAgent, PPOConfig, RolloutBuffer


def test_learning_smoke() -> None:
    graph = Data(
        x=torch.tensor(
            [
                [1.0, 0.5, 110.0, 0.95, 1.05, 1.0],
                [0.8, 0.3, 110.0, 0.95, 1.05, 1.0],
            ],
            dtype=torch.float32,
        ),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        bus_id=torch.tensor([0, 1], dtype=torch.long),
        vm_lower=torch.tensor([0.95, 0.95], dtype=torch.float32),
        vm_upper=torch.tensor([1.05, 1.05], dtype=torch.float32),
        va_lower=torch.tensor([-30.0, -30.0], dtype=torch.float32),
        va_upper=torch.tensor([30.0, 30.0], dtype=torch.float32),
        device_bus_index=torch.tensor([0], dtype=torch.long),
        device_type_id=torch.tensor([0], dtype=torch.long),
        device_element_id=torch.tensor([0], dtype=torch.long),
        device_p_lower=torch.tensor([0.0], dtype=torch.float32),
        device_p_upper=torch.tensor([100.0], dtype=torch.float32),
        device_q_lower=torch.tensor([-50.0], dtype=torch.float32),
        device_q_upper=torch.tensor([50.0], dtype=torch.float32),
        y_bus=torch.tensor([[0.5, 0.5], [0.4, 0.6]], dtype=torch.float32),
        y_device=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
    )

    actor = VoltageWarmStartActor(in_channels=6, hidden_channels=(32, 32, 32))
    critic = VoltageWarmStartCritic(in_channels=6, hidden_channels=(32, 32, 32))
    agent = PPOAgent(actor=actor, critic=critic, config=PPOConfig())

    action, log_prob, value = agent.act(graph, stochastic=True)

    assert action.bus.shape == (2, 2)
    assert action.device.shape == (1, 2)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)

    buffer = RolloutBuffer()
    buffer.add(
        data=graph,
        action_bus=action.bus,
        action_device=action.device,
        log_prob=log_prob,
        value=value,
        reward=-0.1,
    )

    stats = agent.update(buffer)

    assert "loss" in stats
    assert "policy_loss" in stats
    assert "value_loss" in stats
