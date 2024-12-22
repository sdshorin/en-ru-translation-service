import logging
import os
from typing import Tuple

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.trainer import Trainer
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

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: DictConfig):
    log.info("\n" + OmegaConf.to_yaml(config))
    
    seed_everything(config.seed)
    
    device = get_device(config.device)
    log.info(f"Using device: {device}")
    
    train_loader, val_loader, tokenizer = create_dataloaders(config)
    log.info(f"Training samples: {len(train_loader.dataset)}")
    log.info(f"Validation samples: {len(val_loader.dataset)}")
    
    config.model.pad_token_id = tokenizer.pad_token_id
    config.model.vocab_size = tokenizer.vocab_size
    
    initialize_wandb(config)
    
    model = instantiate(config.model.model_info).to(device)
    log.info(f"Created model: {model.__class__.__name__}")
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = instantiate(config.training.optimizer, params=model.parameters())
    
    scheduler = None
    scheduler_steps_parameter = {config.training.scheduler_steps_parameter : len(train_loader) * config.training.num_epochs}
    print(scheduler_steps_parameter)
    
    if "scheduler" in config.training:
        scheduler = instantiate(
            config.training.scheduler,
            optimizer=optimizer,
            **scheduler_steps_parameter,
            # optimizer=optimizer,
            # total_steps=len(train_loader) * config.training.num_epochs
            # num_training_steps=
            
        )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        scheduler=scheduler
    )
    
    trainer.train()
    
    if config.wandb.mode != "disabled":
        wandb.finish()

if __name__ == "__main__":
    main()