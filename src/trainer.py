import logging
import os
from typing import Dict, Optional, Tuple

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        config: dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.checkpoint_base_path = os.path.join(
            config.training.checkpointing.dirpath,
            config.model.name
        )
        os.makedirs(config.training.checkpointing.dirpath, exist_ok=True)
        os.makedirs(self.checkpoint_base_path, exist_ok=True)


    def train(self) -> None:
        for epoch in range(self.config.training.num_epochs):
            log.info(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            self.model.training_progress = epoch / (self.config.training.num_epochs - 1)
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            metrics = {**train_metrics, **val_metrics}
            self._log_metrics(metrics)
            self.show_translation_examples()
            
            if self._should_save_checkpoint(val_metrics['val_loss']):
                self._save_checkpoint(epoch, val_metrics['val_loss'])
            
            if self._should_stop_training():
                log.info(
                    f"Early stopping triggered after {epoch + 1} epochs."
                )
                break

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            loss = self._training_step(src, tgt)
            total_loss += loss.item()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            self._log_batch_metrics(loss.item())
            
        return {"train_loss": total_loss / len(self.train_loader)}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        with torch.no_grad():
            for src, tgt in progress_bar:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                output = self.model(src, tgt)
                loss = self.criterion(
                    output.view(-1, output.size(-1)),
                    tgt.view(-1)
                )
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return {"val_loss": total_loss / len(self.val_loader)}

    def show_translation_examples(self, examples_to_show: int = 5) -> None:
        self.model.eval()
        log.info("\nTranslation Examples:")
        with torch.no_grad():
            for i, (src_batch, tgt_batch) in enumerate(self.val_loader):
                if examples_to_show <= 0:
                    break
                for src, tgt in zip(src_batch, tgt_batch):
                    if examples_to_show <= 0:
                        break

                    examples_to_show -= 1
                    src = src.to(self.device)
                    src_text = self.val_loader.dataset.tokenizer.decode(
                        src, skip_special_tokens=True
                    )
                    
                    tgt_text = self.val_loader.dataset.tokenizer.decode(
                        tgt, skip_special_tokens=True
                    )
                    
                    output = self.model.translate(
                        src.unsqueeze(0),
                        max_len=self.config.data.tokenizer.max_length
                    )
                    output = output.squeeze(0)
                                           
                    pred_text = self.val_loader.dataset.tokenizer.decode(
                        output, skip_special_tokens=True
                    )
                    
                    log.info(f"\nExample {i+1}:")
                    log.info(f"target tokens : {tgt}")
                    log.info(f"pred tokens   : {output}")
                    log.info(f"Source        : {src_text}")
                    log.info(f"Prediction    : {pred_text}")
                    log.info(f"Reference     : {tgt_text}")


    def _training_step(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        
        output = self.model(src, tgt)
        loss = self.criterion(
            output.view(-1, output.size(-1)),
            tgt.view(-1)
        )
        
        loss.backward()
        
        if self.config.training.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip_val
            )
        
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss

    def _should_save_checkpoint(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False

    def _should_stop_training(self) -> bool:
        return self.patience_counter >= self.config.training.early_stopping.patience

    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        checkpoint_path = os.path.join(
            self.checkpoint_base_path,
            f"best_model_epoch_{epoch+1}.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        log.info(f"Saved best model checkpoint to: {checkpoint_path}")

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        log.info(
            f"Train Loss: {metrics['train_loss']:.4f}, "
            f"Val Loss: {metrics['val_loss']:.4f}"
        )
        
        if self.config.wandb.mode != "disabled":
            wandb.log(metrics)

    def _log_batch_metrics(self, loss: float) -> None:
        if self.config.wandb.mode != "disabled" and self.config.wandb.log_batches:
            wandb.log({
                "train_batch_loss": loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            })