from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ProcessorMixin, PreTrainedTokenizerBase


class VLMJsonlDataset(Dataset[dict[str, Any]]):
    """JSONL records must contain `image`, `prompt`, and `answer` fields."""

    def __init__(self, annotation_file: str | Path, image_root: str | Path | None = None) -> None:
        self.image_root = Path(image_root) if image_root else None
        with Path(annotation_file).open(encoding="utf-8") as source:
            self.examples = [json.loads(line) for line in source if line.strip()]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        image_path = Path(example["image"])
        if self.image_root and not image_path.is_absolute():
            image_path = self.image_root / image_path
        with Image.open(image_path) as image:
            return {
                "image": image.convert("RGB").copy(),
                "prompt": example["prompt"],
                "answer": example["answer"],
            }


class VLMDataCollator:
    def __init__(
        self,
        image_processor: ProcessorMixin,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pixel_values = self.image_processor(
            images=[example["image"] for example in examples], return_tensors="pt"
        )["pixel_values"]
        prompts = [example["prompt"] for example in examples]
        answers = [example["answer"] + self.tokenizer.eos_token for example in examples]
        prompt_tokens = self.tokenizer(prompts, add_special_tokens=True)
        tokens = self.tokenizer(
            [prompt + answer for prompt, answer in zip(prompts, answers, strict=True)],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokens.input_ids.clone()
        labels[tokens.attention_mask == 0] = -100
        for row, prompt_ids in enumerate(prompt_tokens.input_ids):
            labels[row, : min(len(prompt_ids), labels.shape[1])] = -100
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "labels": labels,
        }


def build_dataloader(
    dataset: Dataset[dict[str, Any]],
    collator: VLMDataCollator,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
