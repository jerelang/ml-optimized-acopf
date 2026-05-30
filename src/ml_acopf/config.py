from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NetworkName = Literal["case14", "case30", "case118", "case300", "case118_api", "case118_sad"]
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
    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_channels: tuple[int, ...] = (64, 64, 64)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    device_type_embedding_dim: int = Field(default=4, gt=0)


class PretrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=8, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)


class PpoTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    updates: int = Field(default=100, gt=0)
    rollout_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    clip_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    value_loss_weight: float = Field(default=0.5, ge=0.0)
    entropy_weight: float = Field(default=0.01, ge=0.0)
    ppo_epochs: int = Field(default=4, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    nonconvergence_penalty: float = Field(default=12.0, ge=0.0)
    seed: int = 123
    action_std_init: float = Field(default=0.25, gt=0.0)
    action_std_decay_rate: float = Field(default=0.01, ge=0.0)
    action_std_decay_every_steps: int = Field(default=250, gt=0)
    action_std_min: float = Field(default=0.0, ge=0.0)


class NormalizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_voltage_angle_deg: float = Field(default=60.0, gt=0.0)

    @property
    def angle_bounds(self) -> tuple[float, float]:
        max_angle = float(self.max_voltage_angle_deg)
        return -max_angle, max_angle


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_channels: tuple[tuple[int, ...], ...] = ((64, 64, 64), (128, 128, 128))
    ppo_learning_rate: list[float] = Field(default_factory=lambda: [5e-4, 3e-4, 1e-4])
    ppo_entropy_weight: list[float] = Field(default_factory=lambda: [1e-2, 1e-3])
    nonconvergence_penalty: list[float] = Field(default_factory=lambda: [10.0, 12.0, 15.0])

    @field_validator(
        "hidden_channels", "ppo_learning_rate", "ppo_entropy_weight", "nonconvergence_penalty"
    )
    @classmethod
    def validate_search_grid(cls, value: list[float] | list[int]) -> list[float] | list[int]:
        if not value:
            raise ValueError("search grids must not be empty.")
        return value


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    network_names: list[NetworkName] = Field(default_factory=list)
    n_cases: int = Field(default=100, gt=0)
    seed: int = 123
    include_flat: bool = True
    include_pf: bool = True
    create_plots: bool = True


class PerturbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    load_scale_min: float = Field(default=0.90, gt=0.0)
    load_scale_max: float = Field(default=1.10, gt=0.0)
    local_load_noise_scale: float = Field(default=0.05, ge=0.0)
    reactive_noise_scale: float = Field(default=0.02, ge=0.0)

    @model_validator(mode="after")
    def validate_range(self) -> PerturbConfig:
        if self.load_scale_min > self.load_scale_max:
            raise ValueError("perturb.load_scale_min must be <= perturb.load_scale_max.")
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data: DataConfig = Field(default_factory=DataConfig)
    perturb: PerturbConfig = Field(default_factory=PerturbConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    ppo: PpoTrainConfig = Field(default_factory=PpoTrainConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @property
    def dataset_root(self) -> Path:
        return Path("data") / self.data.name


DEFAULT_CONFIG = Config()


def load_config(path: Path | str) -> Config:
    path = Path(path)
    with path.open("rb") as file:
        raw = tomllib.load(file)
    return Config.model_validate(raw)
