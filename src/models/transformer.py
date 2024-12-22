import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from transformers import AutoTokenizer


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 64,
        tokenizer_name:str= ""
    ):
        super().__init__()
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id

        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)
        
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm
        )
        
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm
        )
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, src):
        return src == self.pad_id

    def create_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        if src_key_padding_mask is None:
            src_key_padding_mask = self.create_padding_mask(src)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.create_padding_mask(tgt)
        
        tgt_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
        
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        memory = self.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        output = self.output_layer(output)
        
        return output

    @torch.no_grad()
    def translate(
        self,
        src: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        assert src.size(0) == 1

        device = src.device
        
        src_padding_mask = self.create_padding_mask(src)
        
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoder(src_embedded)
        memory = self.encoder(src_embedded, src_key_padding_mask=src_padding_mask)
        
        decoder_input = torch.tensor([[self.bos_id]], device=device)
        
        for _ in range(self.max_seq_length):
            tgt_mask = self.create_causal_mask(decoder_input.size(1)).to(device)
            tgt_padding_mask = self.create_padding_mask(decoder_input)
            
            tgt_embedded = self.embedding(decoder_input) * math.sqrt(self.d_model)
            tgt_embedded = self.pos_encoder(tgt_embedded)

            decoder_output = self.decoder(
                tgt_embedded,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )

            output = self.output_layer(decoder_output[:, -1:, :])
            
            logits = output.squeeze(1) / temperature
        
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, top_k)[0][:, -1].unsqueeze(-1)
                indices_to_remove = logits < threshold
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == self.eos_id:
                break
            
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        return decoder_input


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)