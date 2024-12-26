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


def test_tokenizer(dataset, tokenizer, num_samples=5):
    log.info("Testing tokenizer functionality...")
    
    test_samples = []
    for idx in range(min(num_samples, len(dataset.dataset))):
        item = dataset.dataset[idx]
        source_text = item["translation"][dataset.source_col]
        target_text = item["translation"][dataset.target_col]
        test_samples.append((source_text, target_text))
    
    all_tests_passed = True
    
    for lang_idx, lang_name in [(0, "source"), (1, "target")]:
        for idx, (source, target) in enumerate(test_samples):
            original_text = source if lang_idx == 0 else target
            encoded = tokenizer.encode(
                original_text,
                add_special_tokens=True,
                return_tensors="pt"
            )
            decoded_text = tokenizer.decode(
                encoded["input_ids"].squeeze(),
                skip_special_tokens=True
            )

            original_normalized = "".join(original_text.split()).strip()
            decoded_normalized = "".join(decoded_text.split()).strip()
            
            original_readable = " ".join(original_text.split()).strip()
            decoded_readable = " ".join(decoded_text.split()).strip()
            if original_normalized != decoded_normalized:
                all_tests_passed = False
                log.error(f"Test failed for {lang_name} language, sample {idx}:")
                log.error(f"Original : {original_readable}")
                log.error(f"Decoded  : {decoded_readable}")
                log.error("Token IDs: " + str(encoded["input_ids"].squeeze().tolist()))
    
    if all_tests_passed:
        log.info("All tokenizer tests passed")
    
    return all_tests_passed


def create_dataloaders(config: DictConfig) -> Tuple[DataLoader, DataLoader, PreTrainedTokenizer]:
    tokenizer = instantiate(config.data.tokenizer)
    
    dataset = instantiate(config.data.dataset, tokenizer=tokenizer)
    if not test_tokenizer(dataset, tokenizer):
        raise ValueError("Tokenizer tests failed! Please check the logs above.")

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
    print("vocab_size: ", tokenizer.vocab_size)
    
    initialize_wandb(config)
    
    model = instantiate(config.model.model_info)
    model.set_tokenizer(tokenizer)
    model.to(device)
    
    log.info(f"Created model: {model.__class__.__name__}")
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id,
                                          label_smoothing=0.1)
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