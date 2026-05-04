from __future__ import annotations

import random
from abc.collections import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm

from ..config import load_config
from ..utils import make_run_name
from .dataset import WarmStartDataset
from .models import (
    VoltageWarmStartActor,
    VoltageWarmStartCritic,
    save_actor,
)
from .ppo import PPOConfig
from .train import (
    evaluate_policy,
    evaluate_supervised_actor,
    pretrain_actor_supervised,
    train_ppo,
)


@dataclass(slots=True)
class BestPretrain:
    hidden_channels: Sequence[int]
    score: float
    actor_state_dict: dict[str, torch.Tensor]


@dataclass(slots=True)
class BestPPO:
    success_rate: float
    mean_solve_time: float
    mean_reward: float
    search_config: dict[str, float | int | str]


def to_cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def is_better(
    success_rate: float,
    mean_solve_time: float,
    current_best: BestPPO | None,
) -> bool:
    if current_best is None:
        return True
    if success_rate != current_best.success_rate:
        return success_rate > current_best.success_rate
    return mean_solve_time < current_best.mean_solve_time


def search(config: Path, out: Path, device: torch.device, pretrain: bool = False):
    cfg = load_config(config)
    dataset_root = cfg.dataset_root
    pretrain_dataset = WarmStartDataset(
        cfg.dataset_root,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
        case_source="converged",
    )

    ppo_dataset = WarmStartDataset(
        cfg.dataset_root,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
        case_source="all",
    )
    print(f"pretrain/PPO cases: {len(pretrain_dataset)} / {len(ppo_dataset)}")

    pretrain_case_ids = [
        pretrain_dataset.case_metadata(i).case_id for i in range(len(pretrain_dataset))
    ]
    rng = random.Random(cfg.ppo.seed)
    rng.shuffle(pretrain_case_ids)

    split_index = max(1, int(0.8 * len(pretrain_case_ids)))
    if split_index >= len(pretrain_case_ids):
        split_index = len(pretrain_case_ids) - 1

    train_case_ids = pretrain_case_ids[:split_index]
    val_case_ids = pretrain_case_ids[split_index:]

    train_dataset = WarmStartDataset(
        dataset_root,
        case_ids=train_case_ids,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
        case_source="converged",
    )
    val_dataset = WarmStartDataset(
        dataset_root,
        case_ids=val_case_ids,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
        case_source="converged",
    )

    # print(message=device)
    run_name = make_run_name(cfg)
    output_dir = out / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrain_rows: list[dict[str, str | float]] = []
    ppo_rows: list[dict[str, str | float]] = []
    best_pretrain: BestPretrain | None = None
    if pretrain:
        for hidden_channels in tqdm(
            cfg.search.hidden_channels,
            desc="Pretraining search",
            unit="config",
            position=0,
        ):
            actor = VoltageWarmStartActor(
                in_channels=pretrain_dataset.input_channels,
                hidden_channels=hidden_channels,
                dropout=cfg.model.dropout,
                device_type_embedding_dim=cfg.model.device_type_embedding_dim,
            )

            pretrain_actor_supervised(
                actor,
                dataset=train_dataset,
                device=device,
                epochs=cfg.pretrain.epochs,
                batch_size=cfg.pretrain.batch_size,
                learning_rate=cfg.pretrain.learning_rate,
                weight_decay=cfg.pretrain.weight_decay,
            )

            val_loss = evaluate_supervised_actor(
                val_dataset,
                actor,
                cfg.pretrain.batch_size,
                device,
            )
            pretrain_rows.append(
                {
                    "hidden_channels": "-".join(str(v) for v in hidden_channels),
                    "val_loss": float(val_loss),
                }
            )
            if best_pretrain is None or val_loss < best_pretrain.score:
                best_pretrain = BestPretrain(
                    hidden_channels=hidden_channels,
                    score=val_loss,
                    actor_state_dict=to_cpu_state_dict(actor),
                )

        if best_pretrain is None:
            raise ValueError("No pretrained actor.")
        pretrain_results = pl.DataFrame(pretrain_rows).sort("val_loss")
        pretrain_results.write_csv(output_dir / "search_pretrain_results.csv")

        print(
            f"Best hidden channel dimension during pretraining: "
            f"{best_pretrain.hidden_channels} (val_loss={best_pretrain.score:.6f})"
        )

    best_final: BestPPO | None = None

    ppo_combinations = list(
        product(
            cfg.search.ppo_learning_rate,
            cfg.search.ppo_entropy_weight,
            cfg.search.nonconvergence_penalty,
        )
    )
    ppo_case_ids = [ppo_dataset.case_metadata(i).case_id for i in range(len(ppo_dataset))]
    rng.shuffle(ppo_case_ids)

    ppo_split_index = max(1, int(0.8 * len(ppo_case_ids)))
    if ppo_split_index >= len(ppo_case_ids):
        ppo_split_index = len(ppo_case_ids) - 1

    ppo_train_case_ids = ppo_case_ids[:ppo_split_index]
    ppo_val_case_ids = ppo_case_ids[ppo_split_index:]

    ppo_train_dataset = WarmStartDataset(
        dataset_root,
        case_ids=ppo_train_case_ids,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
        case_source="all",
    )
    ppo_val_dataset = WarmStartDataset(
        dataset_root,
        case_ids=ppo_val_case_ids,
        max_voltage_angle_deg=cfg.normalization.max_voltage_angle_deg,
        case_source="all",
    )
    hidden_channels = cfg.model.hidden_channels
    if pretrain and best_pretrain:
        hidden_channels = best_pretrain.hidden_channels

    for learning_rate, entropy_weight, nonconvergence_penalty in tqdm(
        ppo_combinations, desc="PPO search", unit="config", position=0
    ):
        actor = VoltageWarmStartActor(
            in_channels=ppo_dataset.input_channels,
            hidden_channels=hidden_channels,
            dropout=cfg.model.dropout,
            device_type_embedding_dim=cfg.model.device_type_embedding_dim,
        )
        if pretrain and best_pretrain:
            actor.load_state_dict(best_pretrain.actor_state_dict)

        critic = VoltageWarmStartCritic(
            in_channels=ppo_dataset.input_channels,
            hidden_channels=hidden_channels,
            dropout=cfg.model.dropout,
        )

        ppo_model_config = PPOConfig(
            learning_rate=learning_rate,
            clip_ratio=cfg.ppo.clip_ratio,
            value_loss_weight=cfg.ppo.value_loss_weight,
            entropy_weight=entropy_weight,
            ppo_epochs=cfg.ppo.ppo_epochs,
            max_grad_norm=cfg.ppo.max_grad_norm,
        )

        agent, history = train_ppo(
            actor,
            critic,
            ppo_train_dataset,
            device=device,
            solver=cfg.solver,
            ppo_config=ppo_model_config,
            updates=cfg.ppo.updates,
            rollout_size=cfg.ppo.rollout_size,
            nonconvergence_penalty=nonconvergence_penalty,
            seed=cfg.ppo.seed,
        )

        metrics = evaluate_policy(
            ppo_val_dataset,
            agent.actor,
            cfg.solver,
            nonconvergence_penalty=nonconvergence_penalty,
            device=device,
        )
        ppo_rows.append(
            {
                "hidden_channels": "-".join(str(v) for v in hidden_channels),
                "ppo_learning_rate": float(learning_rate),
                "ppo_entropy_weight": float(entropy_weight),
                "nonconvergence_penalty": float(nonconvergence_penalty),
                "success_rate": float(metrics.success_rate),
                "mean_solve_time": float(metrics.mean_solve_time),
                "mean_reward": float(metrics.mean_reward),
            }
        )
        if is_better(metrics.success_rate, metrics.mean_solve_time, best_final):
            best_final = BestPPO(
                success_rate=metrics.success_rate,
                mean_solve_time=metrics.mean_solve_time,
                mean_reward=metrics.mean_reward,
                search_config={
                    "hidden_channels": "-".join(str(v) for v in hidden_channels),
                    "ppo_learning_rate": learning_rate,
                    "ppo_entropy_weight": entropy_weight,
                    "nonconvergence_penalty": nonconvergence_penalty,
                },
            )

            best_cfg = cfg.model_copy(
                update={
                    "model": cfg.model.model_copy(
                        update={
                            "hidden_channels": "-".join(str(v) for v in hidden_channels),
                        }
                    ),
                    "ppo": cfg.ppo.model_copy(
                        update={
                            "learning_rate": float(learning_rate),
                            "entropy_weight": float(entropy_weight),
                            "nonconvergence_penalty": float(nonconvergence_penalty),
                        }
                    ),
                }
            )

            save_actor(
                actor=agent.actor,
                input_channels=ppo_dataset.input_channels,
                cfg=best_cfg,
                history=history,
                out=output_dir,
            )

            print(
                f"New best config saved:\n"
                f"  hidden_channels:         {best_final.search_config['hidden_channels']}\n"
                f"  ppo_learning_rate:       {best_final.search_config['ppo_learning_rate']}\n"
                f"  ppo_entropy_weight:      {best_final.search_config['ppo_entropy_weight']}\n"
                f"  nonconvergence_penalty:  {best_final.search_config['nonconvergence_penalty']}\n"
                f"  success_rate:            {best_final.success_rate:.2%}\n"
                f"  mean_solve_time:         {best_final.mean_solve_time:.6f}\n"
                f"  mean_reward:             {best_final.mean_reward:.6f}\n"
                f"saved to: {output_dir}"
            )

    if best_final is None:
        raise ValueError("No PPO config was selected.")

    ppo_results = pl.DataFrame(ppo_rows).sort(
        by=["success_rate", "mean_solve_time", "mean_reward"],
        descending=[True, False, True],
    )
    ppo_results.write_csv(output_dir / "search_ppo_results.csv")
    print(
        f"Best final config:\n"
        f"  hidden_channels:         {best_final.search_config['hidden_channels']}\n"
        f"  ppo_learning_rate:       {best_final.search_config['ppo_learning_rate']}\n"
        f"  ppo_entropy_weight:      {best_final.search_config['ppo_entropy_weight']}\n"
        f"  nonconvergence_penalty:  {best_final.search_config['nonconvergence_penalty']}\n"
        f"  success_rate:            {best_final.success_rate:.2%}\n"
        f"  mean_solve_time:         {best_final.mean_solve_time:.6f}\n"
        f"  mean_reward:             {best_final.mean_reward:.6f}\n"
        f"saved to: {output_dir}"
    )
