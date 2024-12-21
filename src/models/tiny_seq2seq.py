import torch
import torch.nn as nn

class TinySeq2Seq(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.decoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        self.pad_token_id = pad_token_id
        
    def forward(self, src, tgt=None, max_len=None):
        batch_size = src.size(0)
        
        if max_len is None:
            max_len = tgt.size(1) if tgt is not None else src.size(1) + 10
            
        embedded_src = self.embedding(src)
        _, hidden = self.encoder(embedded_src)
        
        decoder_input = torch.full(
            (batch_size, 1),
            self.embedding.num_embeddings - 2,
            device=src.device
        )
        
        outputs = []
        
        for _ in range(max_len):
            embedded_decoder = self.embedding(decoder_input)
            decoder_output, hidden = self.decoder(embedded_decoder, hidden)
            projection = self.output_layer(decoder_output)
            outputs.append(projection)
            
            if self.training and tgt is not None and torch.rand(1).item() < 0.5:
                decoder_input = tgt[:, _:_+1]
            else:
                decoder_input = projection.argmax(2)
            
            if not self.training and (decoder_input == self.embedding.num_embeddings - 1).all():
                break
                
        return torch.cat(outputs, dim=1)
    
    @torch.no_grad()
    def translate(self, src, max_len=50):
        self.eval()
        return self(src, max_len=max_len)