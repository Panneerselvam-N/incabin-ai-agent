from __future__ import annotations

from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlm_dev.utils.dtype import resolve_model_dtype


class LanguageModel(nn.Module):
    """Adapter for decoder-only Hugging Face language models, including Qwen2.5."""

    def __init__(
        self, model_name: str, image_token: str, model_dtype: str, freeze: bool = True
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=resolve_model_dtype(model_dtype)
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.hidden_size = self.model.config.hidden_size
        if freeze:
            self.requires_grad_(False)

    def input_embeddings(self):
        return self.model.get_input_embeddings()
