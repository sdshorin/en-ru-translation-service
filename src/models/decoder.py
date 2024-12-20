from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        attention: bool = True
    ):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = attention
        
        rnn_input_size = embedding_dim + hidden_dim if attention else embedding_dim
        
        self.rnn = nn.GRU(
            rnn_input_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        if attention:
            self.attention_layer = nn.Linear(
                hidden_dim * 2,
                hidden_dim
            )
            self.attention_combine = nn.Linear(
                hidden_dim + embedding_dim,
                embedding_dim
            )
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        embedded = self.dropout(self.embedding(input))
        
        attention_weights = None
        if self.attention and encoder_outputs is not None:
            attention_weights = self._calculate_attention(
                hidden[-1],
                encoder_outputs
            )
            
            context = torch.bmm(
                attention_weights,
                encoder_outputs
            )
            
            rnn_input = torch.cat(
                (embedded, context),
                dim=2
            )
        else:
            rnn_input = embedded
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        output = self.output_layer(output)
        
        return output, hidden, attention_weights

    def _calculate_attention(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        src_len = encoder_outputs.size(1)
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attention_layer(
            torch.cat((hidden, encoder_outputs), dim=2)
        ))
        energy = torch.sum(energy, dim=2)
        
        return F.softmax(energy, dim=1).unsqueeze(1)
