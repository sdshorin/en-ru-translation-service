import logging
import os
from typing import Tuple

import hydra
import torch
import wandb
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from src.utils.helper import seed_everything
from src.utils.device import get_device

log = logging.getLogger(__name__)

def initialize_wandb(config: DictConfig) -> None:
    if config.wandb.mode != "disabled":
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.username,
            config=OmegaConf.to_container(config, resolve=True),
            mode=config.wandb.mode,
        )


def create_dataloaders(config: DictConfig) -> Tuple[DataLoader, DataLoader, PreTrainedTokenizer]:
    tokenizer = instantiate(config.data.tokenizer)
    
    dataset = instantiate(config.data.dataset, tokenizer=tokenizer)
    train_dataset, val_dataset = dataset.split_train_val()
    
    train_loader = DataLoader(
        train_dataset,
        **config.data.dataloader
    )
    
    val_loader = DataLoader(
        val_dataset,
        **config.data.dataloader
    )
    
    return train_loader, val_loader, tokenizer

def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    device: torch.device,
    config: DictConfig,
) -> dict:
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (src, tgt) in enumerate(progress_bar):
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, tgt)
        loss = criterion(
            output.view(-1, output.size(-1)),
            tgt.view(-1)
        )
        
        loss.backward()
        
        if config.training.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clip_val
            )
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}"}
        )
        
        if config.wandb.mode != "disabled":
            wandb.log({
                "train_batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
            })
    
    return {"train_loss": total_loss / len(train_loader)}

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Validate model performance."""
    model.eval()
    total_loss = 0
    
    progress_bar = tqdm(val_loader, desc="Validation")
    for src, tgt in progress_bar:
        src = src.to(device)
        tgt = tgt.to(device)
        
        output = model(src, tgt)
        loss = criterion(
            output.view(-1, output.size(-1)),
            tgt.view(-1)
        )
        
        total_loss += loss.item()
        
        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}"}
        )
    
    return {"val_loss": total_loss / len(val_loader)}

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig):
    log.info("\n" + OmegaConf.to_yaml(config))
    
    seed_everything(config.seed)
    
    device = get_device(config.device)
    log.info(f"Using device: {device}")
    
    train_loader, val_loader, tokenizer = create_dataloaders(config)
    log.info(f"Training samples: {len(train_loader.dataset)}")
    log.info(f"Validation samples: {len(val_loader.dataset)}")
    
    config.model.encoder.vocab_size = tokenizer.vocab_size
    config.model.decoder.vocab_size = tokenizer.vocab_size
    config.model.seq2seq.pad_token_id = tokenizer.pad_token_id
    
    initialize_wandb(config)
    
    model = instantiate(config.model.seq2seq).to(device)
    log.info(f"Created model: {model.__class__.__name__}")
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = instantiate(config.training.optimizer, params=model.parameters())
    
    scheduler = None
    if "scheduler" in config.training:
        scheduler = instantiate(
            config.training.scheduler,
            optimizer=optimizer,
            total_steps=len(train_loader) * config.training.num_epochs
        )
    
    os.makedirs(config.training.checkpointing.dirpath, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.num_epochs):
        log.info(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
    
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, config
        )
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        metrics = {**train_metrics, **val_metrics}
        log.info(
            f"Train Loss: {metrics['train_loss']:.4f}, "
            f"Val Loss: {metrics['val_loss']:.4f}"
        )
        
        if config.wandb.mode != "disabled":
            wandb.log(metrics)
        
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            
            checkpoint_path = os.path.join(
                config.training.checkpointing.dirpath,
                f"best_model_epoch_{epoch+1}.pt"
            )
            save_checkpoint(
                model, optimizer, epoch,
                val_metrics['val_loss'],
                checkpoint_path
            )
            log.info(f"Saved best model checkpoint to: {checkpoint_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= config.training.early_stopping.patience:
            log.info(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"without improvement."
            )
            break
    
    if config.wandb.mode != "disabled":
        wandb.finish()

if __name__ == "__main__":
    main()