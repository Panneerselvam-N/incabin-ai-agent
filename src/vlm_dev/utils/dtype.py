from __future__ import annotations

import torch


_MODEL_DTYPES: dict[str, torch.dtype | None] = {
    "auto": None,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_model_dtype(name: str) -> torch.dtype | None:
    try:
        return _MODEL_DTYPES[name]
    except KeyError as error:
        choices = ", ".join(_MODEL_DTYPES)
        raise ValueError(f"Unknown model_dtype '{name}'. Choose one of: {choices}") from error


def module_dtype(module: torch.nn.Module) -> torch.dtype:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.float32
    return parameter.dtype
