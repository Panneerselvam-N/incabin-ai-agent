from __future__ import annotations

import argparse

from vlm_dev.data.dataset import VLMDataCollator, VLMJsonlDataset, build_dataloader
from vlm_dev.models.vlm import VisionLanguageModel
from vlm_dev.training.trainer import Trainer
from vlm_dev.utils.config import load_config
from vlm_dev.utils.runtime import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a prefix-fusion vision-language model.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML configuration.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.training.seed)
    model = VisionLanguageModel(config.model)
    collator = VLMDataCollator(
        model.vision_encoder.processor,
        model.language_model.tokenizer,
        config.data.max_length,
    )
    train_loader = build_dataloader(
        VLMJsonlDataset(config.data.train_file, config.data.image_root),
        collator,
        config.training.train_batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    validation_loader = build_dataloader(
        VLMJsonlDataset(config.data.validation_file, config.data.image_root),
        collator,
        config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    Trainer(config.training).fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    main()
