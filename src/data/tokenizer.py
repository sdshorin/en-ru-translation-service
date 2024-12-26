import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizer
import torch

log = logging.getLogger(__name__)

class TranslationTokenizer:
    DEFAULT_SPECIAL_TOKENS = {
        'pad_token': '[PAD]',
        'unk_token': '[UNK]',
        'bos_token': '[BOS]',
        'eos_token': '[EOS]',
        'sep_token': '[SEP]',
        'cls_token': '[CLS]',
        'mask_token': '[MASK]'
    }

    def __init__(
        self,
        name: str = "t5-base",
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

        missing_tokens = {}
        for token_name, token_value in self.DEFAULT_SPECIAL_TOKENS.items():
            if getattr(self.tokenizer, token_name, None) is None:
                missing_tokens[token_name] = token_value
                log.info(f"Adding missing special token: {token_name}={token_value}")

        if missing_tokens:
            self.tokenizer.add_special_tokens(missing_tokens)

        for token_name in self.DEFAULT_SPECIAL_TOKENS.keys():
            token_value = getattr(self.tokenizer, token_name)
            setattr(self, token_name, token_value)
            
            token_id_name = f"{token_name}_id"
            token_id_value = getattr(self.tokenizer, token_id_name)
            setattr(self, token_id_name, token_id_value)

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