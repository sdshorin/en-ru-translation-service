import torch
import torch.nn as nn
import torch.nn.functional as F

class TinySeq2Seq(nn.Module):
    def __init__(
        self, 
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        min_teacher_forcing_ratio: float = 0.2,
        max_teacher_forcing_ratio: float = 1.2,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.min_teacher_forcing_ratio = min_teacher_forcing_ratio
        self.max_teacher_forcing_ratio = max_teacher_forcing_ratio

        self.training_progress = 0
        
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(tokenizer.vocab_size, self.embedding_dim, padding_idx=tokenizer.pad_token_id)
        self.output_layer = nn.Linear(self.hidden_dim, tokenizer.vocab_size)

    def get_teacher_forcing_ratio(self):
        ratio = self.max_teacher_forcing_ratio - (
            (self.max_teacher_forcing_ratio - self.min_teacher_forcing_ratio) * 
            (self.training_progress)
        )
        return ratio

    def forward(self, src, tgt=None, max_len=None):
        batch_size = src.size(0)
        if max_len is None:
            max_len = tgt.size(1) if tgt is not None else src.size(1) + 10
            
        embedded_src = self.embedding(src)
        _, hidden = self.encoder(embedded_src)
        
        decoder_input = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            device=src.device
        )
        
        outputs = []
        teacher_forcing_ratio = self.get_teacher_forcing_ratio()
        
        for pos in range(max_len):
            embedded_decoder = self.embedding(decoder_input)
            decoder_output, hidden = self.decoder(embedded_decoder, hidden)
            projection = self.output_layer(decoder_output)
            outputs.append(projection)
            
            if self.training and tgt is not None:
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                decoder_input = tgt[:, pos:pos+1] if use_teacher_forcing else projection.argmax(2)
            else:
                decoder_input = projection.argmax(2)
            
            if not self.training and (decoder_input == self.tokenizer.eos_token_id).all():
                break
                
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def translate(self, src, max_len=50, debug_print=False):
        self.eval()
        batch_size = src.size(0)
        
        embedded_src = self.embedding(src)
        _, hidden = self.encoder(embedded_src)
        
        decoder_input = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,  # Start token
            device=src.device
        )
        
        outputs = []
        
        for pos in range(max_len):
            embedded_decoder = self.embedding(decoder_input)
            decoder_output, hidden = self.decoder(embedded_decoder, hidden)
            logits = self.output_layer(decoder_output)
            
            if debug_print:
                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = probs.topk(5, dim=-1)
                for batch_idx in range(batch_size):
                    print(f"\nPosition {pos}, Batch {batch_idx}:")
                    for prob_idx in range(5):
                        token_id = top_indices[batch_idx, 0, prob_idx].item()
                        prob = top_probs[batch_idx, 0, prob_idx].item()
                        print(f"  Token {token_id}: {prob:.4f}")
            
            outputs.append(logits)
            decoder_input = logits.argmax(2)
            
            if (decoder_input ==  self.tokenizer.eos_token_id).all():  # End token
                break
        
        output_logits = torch.cat(outputs, dim=1)
        return output_logits.argmax(dim=-1)