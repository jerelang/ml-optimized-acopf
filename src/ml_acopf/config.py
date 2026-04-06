from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NetworkName = Literal["case14", "case30", "case57", "case118", "case300"]
OpfFlowLimit = Literal["S", "I"]


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    name: str = "case14_baseline"
    network_name: NetworkName = "case14"
    n_cases: int = Field(default=50, gt=0)
    seed: int = 123
    max_attempts_multiplier: int = Field(default=5, ge=1)

    @field_validator("name")
    @classmethod
    def validate_data_name(cls, value: str) -> str:
        if not value:
            raise ValueError("data.name must be non-empty.")
        return value


class PerturbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    load_scale_min: float = Field(default=0.90, gt=0.0)
    load_scale_max: float = Field(default=1.10, gt=0.0)

    @model_validator(mode="after")
    def validate_range(self) -> PerturbConfig:
        if self.load_scale_min > self.load_scale_max:
            raise ValueError("perturb.load_scale_min must be <= perturb.load_scale_max.")
        return self


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    pm_solver: str = "ipopt"
    pm_tol: float = Field(default=1e-8, gt=0.0)
    pm_log_level: int = Field(default=0, ge=0)
    delta: float = Field(default=1e-8, gt=0.0)
    calculate_voltage_angles: bool = True
    check_connectivity: bool = True
    correct_pm_network_data: bool = True
    silence: bool = True
    delete_buffer_file: bool = True
    opf_flow_lim: OpfFlowLimit = "S"


class ModelConfig(BaseModel):
    hidden_channels: int = 128
    dropout: float = 0.0
    device_type_embedding_dim: int = 4
    action_std_init: float = 0.05


class PretrainConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class PpoTrainConfig(BaseModel):
    updates: int = 100
    rollout_size: int = 32
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    ppo_epochs: int = 4
    max_grad_norm: float = 1.0
    nonconvergence_penalty: float = 10.0
    seed: int = 123


class SearchConfig(BaseModel):
    hidden_channels: list[int] = [64, 128, 256, 512]
    ppo_learning_rate: list[float] = [5e-4, 3e-4, 1e-4]
    ppo_entropy_weight: list[float] = [1e-2, 1e-3]
    nonconvergence_penalty: list[int] = [10, 12, 15]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data: DataConfig = Field(default_factory=DataConfig)
    perturb: PerturbConfig = Field(default_factory=PerturbConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    ppo: PpoTrainConfig = Field(default_factory=PpoTrainConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    @property
    def dataset_root(self) -> Path:
        return Path("data") / self.data.name


DEFAULT_CONFIG = Config()


def load_config(path: Path | str) -> Config:
    path = Path(path)
    with path.open("rb") as file:
        raw = tomllib.load(file)
    return Config.model_validate(raw)
