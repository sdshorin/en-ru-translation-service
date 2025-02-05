import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        teacher_forcing_ratio: float = 0.5,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.pad_token_id = pad_token_id

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None
    ) -> torch.Tensor:
        batch_size = src.size(0)
        
        if max_len is None:
            max_len = tgt.size(1) if tgt is not None else src.size(1) + 10
        
        src_mask = (src != self.pad_token_id).sum(dim=1)
        
        _, hidden = self.encoder(src, src_mask)
        
        decoder_input = torch.full(
            (batch_size, 1),
            self.decoder.embedding.num_embeddings - 2,  # <BOS>
            device=src.device
        )
        
        decoder_outputs = []
        
        for t in range(max_len):
            decoder_output, hidden = self.decoder(
                decoder_input,
                hidden
            )
            
            decoder_outputs.append(decoder_output)
            
            if tgt is not None and random.random() < self.teacher_forcing_ratio:
                decoder_input = tgt[:, t:t+1]
            else:
                decoder_input = decoder_output.argmax(2)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        return decoder_outputs

    @torch.no_grad()
    def translate(
        self,
        src: torch.Tensor,
        max_len: int = 100,
        temperature: float = 1.0
    ) -> torch.Tensor:
        batch_size = src.size(0)
        
        _, hidden = self.encoder(src)
        
        decoder_input = torch.full(
            (batch_size, 1),
            self.decoder.embedding.num_embeddings - 2,  # <BOS>
            device=src.device
        )
        
        predictions = []
        
        for _ in range(max_len):
            decoder_output, hidden = self.decoder(
                decoder_input,
                hidden
            )
            
            if temperature != 1.0:
                decoder_output = decoder_output / temperature
            
            prediction = F.gumbel_softmax(
                decoder_output,
                tau=temperature,
                hard=True
            )
            predictions.append(prediction)
            
            decoder_input = prediction.argmax(2)
            
            # Stop if all sequences if EOS
            if (decoder_input == self.decoder.embedding.num_embeddings - 1).all():
                break
        
        predictions = torch.cat(predictions, dim=1)
        
        return predictions