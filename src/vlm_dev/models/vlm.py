from __future__ import annotations

import torch
from torch import nn

from vlm_dev.models.language_model import LanguageModel
from vlm_dev.models.projector import build_projector
from vlm_dev.models.vision_encoder import VisionEncoder
from vlm_dev.utils.config import ModelConfig


class VisionLanguageModel(nn.Module):
    """Prefix-fusion VLM: vision tokens are projected and prepended to text tokens."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(
            config.vision_model_name,
            model_dtype=config.model_dtype,
            freeze=config.freeze_vision_encoder,
        )
        self.language_model = LanguageModel(
            config.language_model_name,
            image_token=config.image_token,
            model_dtype=config.model_dtype,
            freeze=config.freeze_language_model,
        )
        self.projector = build_projector(
            config.projector_type,
            self.vision_encoder.hidden_size,
            self.language_model.hidden_size,
        )
        self.projector.to(dtype=self.language_model.input_embeddings().weight.dtype)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_features = self.vision_encoder(pixel_values)
        projector_dtype = next(self.projector.parameters()).dtype
        return self.projector(vision_features.to(dtype=projector_dtype))

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.language_model.input_embeddings()(input_ids)
        combined_embeds = torch.cat((image_embeds.to(dtype=text_embeds.dtype), text_embeds), dim=1)

        batch_size, image_tokens, _ = image_embeds.shape
        image_attention = torch.ones(
            (batch_size, image_tokens), dtype=attention_mask.dtype, device=attention_mask.device
        )
        combined_attention = torch.cat((image_attention, attention_mask), dim=1)

        combined_labels = None
        if labels is not None:
            image_labels = torch.full(
                (batch_size, image_tokens), -100, dtype=labels.dtype, device=labels.device
            )
            combined_labels = torch.cat((image_labels, labels), dim=1)

        return self.language_model.model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention,
            labels=combined_labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **generation_kwargs: object,
    ) -> torch.Tensor:
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.language_model.input_embeddings()(input_ids)
        image_attention = torch.ones(
            image_embeds.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device
        )
        return self.language_model.model.generate(
            inputs_embeds=torch.cat(
                (image_embeds.to(dtype=text_embeds.dtype), text_embeds), dim=1
            ),
            attention_mask=torch.cat((image_attention, attention_mask), dim=1),
            pad_token_id=self.language_model.tokenizer.pad_token_id,
            **generation_kwargs,
        )
