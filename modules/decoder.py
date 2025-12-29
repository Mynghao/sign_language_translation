import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding.

    Input:  x (T, N, D)
    Output: x + pe (same shape)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        # even indices (0,2,4,...) use sin, odd use cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register as buffer: not a parameter, but saved with the model & moved to GPU
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (T, N, D)
        """
        T = x.size(0)
        x = x + self.pe[:T]  # broadcast over batch dimension
        return self.dropout(x)

class TextTransformerDecoder(nn.Module):
    """
    Transformer-based text decoder for sign language translation.

    Args:
        vocab_size: size of target vocabulary
        d_model: feature size (must match STGCNEncoder.output_dim)
        nhead: number of attention heads
        num_layers: number of transformer decoder layers
        dim_feedforward: hidden dim in FFN inside transformer
        dropout: dropout probability
    """

    def __init__(self,
                 vocab_size,
                 d_model=256,
                 nhead=8,
                 num_layers=3,
                 dim_feedforward=1024,
                 dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # 1) Token embedding: maps token ids -> continuous vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 2) Positional encoding for target sequence
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3) Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # we use (T, N, D)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # 4) Final projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, T):
        """
        Causal mask so position t cannot attend to t+1, t+2, ...
        Shape: (T, T)
        """
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        return mask  # True means "blocked"

    def forward(self, tgt_tokens, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        tgt_tokens: (T_tgt, N)        - token ids (teacher forcing)
        memory:    (T_src, N, D)      - encoder outputs from ST-GCN
        *_key_padding_mask: optionals (N, T)

        Returns:
            logits: (T_tgt, N, vocab_size)
        """
        T_tgt, N = tgt_tokens.size()

        # 1) Token embedding + scale
        tgt = self.embedding(tgt_tokens) * math.sqrt(self.d_model)  # (T_tgt, N, D)

        # 2) Add positional encoding
        tgt = self.pos_encoder(tgt)  # (T_tgt, N, D)

        # 3) Create causal mask for autoregressive decoding
        tgt_mask = self.generate_square_subsequent_mask(T_tgt).to(tgt_tokens.device)

        # 4) Run transformer decoder
        # memory is (T_src, N, D) from the encoder
        decoder_out = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # (T_tgt, N, D)

        # 5) Project to vocabulary logits
        logits = self.fc_out(decoder_out)  # (T_tgt, N, vocab_size)

        return logits

