
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizer
import torch

log = logging.getLogger(__name__)

class TranslationTokenizer:
    def __init__(
        self,
        name: str = "facebook/wmt19-ru-en",
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        
        self.name = name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        log.info(f"Loading tokenizer {name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            cache_dir=cache_dir
        )
        
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        log.info(f"Vocabulary size: {self.vocab_size}")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: str = "pt"
    ) -> Dict:
        
        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=return_tensors
        )

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )

    def save(self, path: Union[str, Path]):
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        **kwargs
    ) -> "TranslationTokenizer":
        tokenizer = cls(name=str(path), **kwargs)
        return tokenizer