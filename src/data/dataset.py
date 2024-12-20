# src/data/dataset.py
import logging
from typing import Dict, List, Optional, Tuple

import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

log = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "open_subtitles",
        lang1: str = "en",
        lang2: str = "ru",
        split: str = "train",
        max_length: int = 128,
        train_size: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
        
        log.info(f"Loading {dataset_name} dataset...")
        self.dataset = datasets.load_dataset(
            dataset_name,
            lang1=lang1,
            lang2=lang2,
            split=split
        )
        
        if train_size is not None:
            log.info(f"Using {train_size} examples for training")
            self.dataset = self.dataset.select(range(train_size))
        
        self.source_col = f"{lang1}"
        self.target_col = f"{lang2}"
        
        log.info(f"Loaded {len(self.dataset)} examples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[idx]
        
        source_text = item["translation"][self.source_col]
        target_text = item["translation"][self.target_col]
        
        source_encoding = self.tokenizer.encode(
            source_text,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer.encode(
            target_text,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        return (
            source_encoding["input_ids"].squeeze(),
            target_encoding["input_ids"].squeeze()
        )

    def split_train_val(
        self,
        val_size: float = 0.1,
        seed: int = 42
    ) -> Tuple["TranslationDataset", "TranslationDataset"]:
        dataset_dict = self.dataset.train_test_split(
            test_size=val_size,
            seed=seed
        )
        
        train_dataset = TranslationDataset(
            tokenizer=self.tokenizer,
            dataset_name=self.dataset_name,
            lang1=self.source_col,
            lang2=self.target_col,
            max_length=self.max_length
        )
        train_dataset.dataset = dataset_dict["train"]
        
        val_dataset = TranslationDataset(
            tokenizer=self.tokenizer,
            dataset_name=self.dataset_name,
            lang1=self.source_col,
            lang2=self.target_col,
            max_length=self.max_length
        )
        val_dataset.dataset = dataset_dict["test"]
        
        return train_dataset, val_dataset

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

