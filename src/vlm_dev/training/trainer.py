from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from vlm_dev.training.validator import Validator
from vlm_dev.utils.config import TrainingConfig
from vlm_dev.utils.runtime import load_checkpoint, save_checkpoint


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )

    def fit(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
    ) -> None:
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters. Unfreeze a model component or projector.")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        updates_per_epoch = (len(train_loader) + self.config.gradient_accumulation_steps - 1) // self.config.gradient_accumulation_steps
        total_updates = updates_per_epoch * self.config.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_updates * self.config.warmup_ratio),
            num_training_steps=total_updates,
        )
        if validation_loader is not None:
            model, optimizer, train_loader, validation_loader, scheduler = self.accelerator.prepare(
                model, optimizer, train_loader, validation_loader, scheduler
            )
        else:
            model, optimizer, train_loader, scheduler = self.accelerator.prepare(
                model, optimizer, train_loader, scheduler
            )

        start_epoch, global_step = 0, 0
        if self.config.resume_from:
            start_epoch, global_step = load_checkpoint(
                self.config.resume_from,
                self.accelerator.unwrap_model(model),
                optimizer,
                scheduler,
            )

        validator = Validator(self.accelerator)
        for epoch in range(start_epoch, self.config.epochs):
            model.train()
            for batch in train_loader:
                with self.accelerator.accumulate(model):
                    outputs = model(**batch)
                    self.accelerator.backward(outputs.loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                if self.accelerator.sync_gradients:
                    global_step += 1
                    if global_step % self.config.log_every_steps == 0:
                        self.accelerator.print(f"step={global_step} train_loss={outputs.loss.item():.4f}")
                    if global_step % self.config.save_every_steps == 0:
                        self._save(model, optimizer, scheduler, epoch, global_step, f"step-{global_step}")
            if validation_loader is not None:
                metrics = validator.evaluate(model, validation_loader)
                self.accelerator.print(f"epoch={epoch + 1} val_loss={metrics['loss']:.4f} ppl={metrics['perplexity']:.2f}")
            self._save(model, optimizer, scheduler, epoch + 1, global_step, "last")

    def _save(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, global_step: int, name: str,
    ) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            save_checkpoint(
                Path(self.config.output_dir) / name,
                self.accelerator.unwrap_model(model), optimizer, scheduler, epoch, global_step,
            )
