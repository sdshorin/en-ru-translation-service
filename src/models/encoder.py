from typing import Optional, Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        embedded = self.dropout(self.embedding(src))
        
        if src_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(),
                batch_first=True, enforce_sorted=False
            )
        
        outputs, hidden = self.rnn(embedded)
        
        if src_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs,
                batch_first=True
            )
        
        if self.rnn.bidirectional:
            hidden = self._combine_bidir(hidden, batch_size=src.size(0))
        
        return outputs, hidden

    def _combine_bidir(
        self,
        hidden: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        
        hidden = hidden.view(
            self.num_layers,
            2,
            batch_size,
            self.hidden_dim
        )
        return hidden.transpose(1, 2).contiguous().view(
            self.num_layers,
            batch_size,
            self.hidden_dim * 2
        )