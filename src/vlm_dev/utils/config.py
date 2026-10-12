from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vision_model_name: str
    language_model_name: str
    projector_type: str = "mlp2x_gelu"
    freeze_vision_encoder: bool = True
    freeze_language_model: bool = True
    image_token: str = "<image>"
    model_dtype: str = "bfloat16"


@dataclass
class DataConfig:
    train_file: str
    validation_file: str
    image_root: str | None = None
    max_length: int = 256
    num_workers: int = 4


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/default"
    seed: int = 42
    epochs: int = 1
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    log_every_steps: int = 10
    save_every_steps: int = 500
    resume_from: str | None = None


@dataclass
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        raw_config: dict[str, Any] = yaml.safe_load(config_file)
    return ExperimentConfig(
        model=ModelConfig(**raw_config["model"]),
        data=DataConfig(**raw_config["data"]),
        training=TrainingConfig(**raw_config.get("training", {})),
    )
