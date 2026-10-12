from __future__ import annotations

import argparse

from accelerate import Accelerator

from vlm_dev.data.dataset import VLMDataCollator, VLMJsonlDataset, build_dataloader
from vlm_dev.models.vlm import VisionLanguageModel
from vlm_dev.training.validator import Validator
from vlm_dev.utils.config import load_config
from vlm_dev.utils.runtime import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VLM checkpoint.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML configuration.")
    parser.add_argument("--checkpoint", required=True, help="Directory containing checkpoint.pt.")
    args = parser.parse_args()

    config = load_config(args.config)
    accelerator = Accelerator(mixed_precision=config.training.mixed_precision)
    model = VisionLanguageModel(config.model)
    load_checkpoint(args.checkpoint, model)
    collator = VLMDataCollator(
        model.vision_encoder.processor,
        model.language_model.tokenizer,
        config.data.max_length,
    )
    loader = build_dataloader(
        VLMJsonlDataset(config.data.validation_file, config.data.image_root),
        collator,
        config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    model, loader = accelerator.prepare(model, loader)
    metrics = Validator(accelerator).evaluate(model, loader)
    accelerator.print(metrics)


if __name__ == "__main__":
    main()
