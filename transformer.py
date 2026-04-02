import torch
import torch.nn as nn

class Board2048Transformer(nn.Module):
    def __init__(self, embedding_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        
        # Embed tile values (0-15 for empty through 32768)
        self.tile_embedding = nn.Embedding(16, embedding_dim)
        
        # Learnable position embeddings for 16 board positions
        self.position_embedding = nn.Parameter(
            torch.randn(16, embedding_dim)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output head: predict single afterstate value
        self.value_head = nn.Linear(embedding_dim, 1)
    
    def forward(self, board):
        # board: [batch, 16] of tile values (0-15)
        
        # Get embeddings
        x = self.tile_embedding(board)   # [batch, 16, embedding_dim]
        x = x + self.position_embedding  # Add positional info
        
        # Transformer expects [seq_len, batch, embedding_dim]
        # x = x.transpose(0, 1)
        
        # Self-attention over all 16 positions
        x = self.transformer(x)  # [16, batch, embedding_dim]
        
        # Pool over positions
        x = x.mean(dim=1)  # [batch, embedding_dim]
        
        # Predict value
        value = self.value_head(x)  # [batch, 1]
        
        return value.squeeze(-1)