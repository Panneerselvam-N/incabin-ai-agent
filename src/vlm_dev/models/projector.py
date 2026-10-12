from __future__ import annotations

from torch import nn


def build_projector(projector_type: str, vision_hidden_size: int, language_hidden_size: int) -> nn.Module:
    projectors: dict[str, nn.Module] = {
        "linear": nn.Linear(vision_hidden_size, language_hidden_size),
        "mlp2x_gelu": nn.Sequential(
            nn.Linear(vision_hidden_size, language_hidden_size),
            nn.GELU(),
            nn.Linear(language_hidden_size, language_hidden_size),
        ),
    }
    if projector_type not in projectors:
        choices = ", ".join(projectors)
        raise ValueError(f"Unknown projector '{projector_type}'. Choose one of: {choices}")
    return projectors[projector_type]
