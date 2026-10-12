from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from vlm_dev.models.vlm import VisionLanguageModel
from vlm_dev.utils.config import ExperimentConfig
from vlm_dev.utils.runtime import load_checkpoint


class VLMInference:
    """Generates a response using either a trained checkpoint or a random projector."""

    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_dir: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = VisionLanguageModel(config.model)
        if checkpoint_dir is not None:
            load_checkpoint(checkpoint_dir, self.model)
        self.model.to(self.device).eval()
        self.tokenizer = self.model.language_model.tokenizer
        self.image_processor = self.model.vision_encoder.processor

    @torch.inference_mode()
    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> str:
        with Image.open(image_path) as image:
            pixel_values = self.image_processor(
                images=image.convert("RGB"), return_tensors="pt"
            )["pixel_values"].to(self.device)
        text = self.tokenizer(prompt, return_tensors="pt")
        input_ids = text.input_ids.to(self.device)
        attention_mask = text.attention_mask.to(self.device)
        generation_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p)
        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
