from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import juliacall  # noqa: F401
import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from ..config import Config
from ..solver.io import WarmStartPayload
from .dataset import (
    build_bus_warmstart_frame,
    build_device_warmstart_frame,
    denormalize_bus_values,
    denormalize_device_values,
)


@dataclass(frozen=True, slots=True)
class WarmStartAction:
    bus: Tensor
    device: Tensor


@dataclass(frozen=True, slots=True)
class WarmStartDistribution:
    bus: Normal
    device: Normal | None


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList(
            [
                GCNConv(in_channels, hidden_channels),
                GCNConv(hidden_channels, hidden_channels),
                GCNConv(hidden_channels, hidden_channels),
            ]
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.tanh(x)
            if self.dropout > 0.0:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class VoltageWarmStartActor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        dropout: float = 0.0,
        action_std_init: float = 0.05,
        device_type_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

        self.bus_head = nn.Linear(hidden_channels, 2)
        self.device_type_embedding = nn.Embedding(3, device_type_embedding_dim)  # 3 device types
        self.device_head = nn.Sequential(
            nn.Linear(hidden_channels + device_type_embedding_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 2),
        )

        self.bus_log_std = nn.Parameter(torch.full((2,), math.log(action_std_init)))
        self.device_log_std = nn.Parameter(torch.full((2,), math.log(action_std_init)))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        device_bus_index: Tensor,
        device_type_id: Tensor,
    ) -> WarmStartAction:
        h = self.encoder(x, edge_index)

        bus = torch.sigmoid(self.bus_head(h))

        if device_bus_index.numel() == 0:
            device = h.new_empty((0, 2))
        else:
            device_h = h[device_bus_index]
            type_h = self.device_type_embedding(device_type_id)
            device = torch.sigmoid(self.device_head(torch.cat([device_h, type_h], dim=-1)))

        return WarmStartAction(bus=bus, device=device)

    def distribution(
        self,
        x: Tensor | None,
        edge_index: Tensor | None,
        device_bus_index: Tensor | None,
        device_type_id: Tensor | None,
    ) -> WarmStartDistribution:
        if x is None or edge_index is None or device_bus_index is None or device_type_id is None:
            raise ValueError("Data Tensors must not be None")
        mean = self.forward(x, edge_index, device_bus_index, device_type_id)

        bus_std = torch.exp(self.bus_log_std).clamp(min=1e-3, max=0.5)
        bus_dist = Normal(mean.bus, bus_std.expand_as(mean.bus))

        if mean.device.numel() == 0:
            device_dist = None
        else:
            device_std = torch.exp(self.device_log_std).clamp(min=1e-3, max=0.5)
            device_dist = Normal(mean.device, device_std.expand_as(mean.device))

        return WarmStartDistribution(bus=bus_dist, device=device_dist)


class VoltageWarmStartCritic(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = GNNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None) -> Tensor:
        h = self.encoder(x, edge_index)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = global_mean_pool(h, batch)
        return self.value_head(pooled).squeeze(-1)


@dataclass(slots=True)
class WarmStartPredictor:
    model: VoltageWarmStartActor
    device: torch.device
    name: str = "gnn_actor"

    def predict(self, graph: Data, case_id: str) -> WarmStartPayload:
        self.model.eval()
        graph = graph.to(self.device)

        with torch.no_grad():
            normalized = self.model(
                graph.x,
                graph.edge_index,
                graph.device_bus_index,
                graph.device_type_id,
            )

        bus_values = denormalize_bus_values(
            normalized.bus.cpu(),
            graph.vm_lower.cpu(),
            graph.vm_upper.cpu(),
            graph.va_lower.cpu(),
            graph.va_upper.cpu(),
        )
        device_values = denormalize_device_values(
            normalized.device.cpu(),
            graph.device_p_lower.cpu(),
            graph.device_p_upper.cpu(),
            graph.device_q_lower.cpu(),
            graph.device_q_upper.cpu(),
        )

        bus_frame = build_bus_warmstart_frame(case_id, graph.bus_id.cpu(), bus_values)
        device_frame = build_device_warmstart_frame(
            case_id,
            graph.device_type_id.cpu(),
            graph.device_element_id.cpu(),
            device_values,
        )

        return WarmStartPayload(bus=bus_frame, device=device_frame)


def save_actor(
    actor: VoltageWarmStartActor,
    input_channels: int,
    cfg: Config,
    history: list[dict[str, float]],
    out: Path | str,
) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "model_config": {
                "in_channels": input_channels,
                "hidden_channels": cfg.model.hidden_channels,
                "dropout": cfg.model.dropout,
                "action_std_init": cfg.model.action_std_init,
                "device_type_embedding_dim": cfg.model.device_type_embedding_dim,
            },
            "full_cfg": cfg.model_dump(),
            "history": history,
        },
        out / "agent_ppo.pt",
    )


def load_actor(path: Path | str) -> VoltageWarmStartActor:
    checkpoint = torch.load(path, map_location="cpu")
    actor = VoltageWarmStartActor(**checkpoint["model_config"])
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor
