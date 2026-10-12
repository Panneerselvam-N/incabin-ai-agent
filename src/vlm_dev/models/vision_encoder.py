from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoProcessor

from vlm_dev.utils.dtype import module_dtype, resolve_model_dtype


class VisionEncoder(nn.Module):
    """Adapter for vision-only Hugging Face models, including SigLIP2."""

    def __init__(self, model_name: str, model_dtype: str, freeze: bool = True) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=resolve_model_dtype(model_dtype)
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.hidden_size = self.model.config.vision_config.hidden_size
        if freeze:
            self.requires_grad_(False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_model = getattr(self.model, "vision_model", self.model)
        pixel_values = pixel_values.to(dtype=module_dtype(vision_model))
        outputs = vision_model(pixel_values=pixel_values)
        if not hasattr(outputs, "last_hidden_state"):
            raise TypeError("Vision model must return patch features in `last_hidden_state`.")
        return outputs.last_hidden_state
