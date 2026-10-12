from __future__ import annotations

import math

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader


class Validator:
    def __init__(self, accelerator: Accelerator) -> None:
        self.accelerator = accelerator

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> dict[str, float]:
        model.eval()
        losses: list[torch.Tensor] = []
        for batch in dataloader:
            outputs = model(**batch)
            losses.append(self.accelerator.gather_for_metrics(outputs.loss.detach()).float())
        mean_loss = torch.cat(losses).mean().item()
        return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20))}
