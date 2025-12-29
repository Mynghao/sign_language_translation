import torch
import torch.nn as nn
import math
from modules.encoder import STGCNEncoder
from modules.decoder import TextTransformerDecoder, PositionalEncoding

class Sign2TextModel(nn.Module):
    """
    End-to-end model: ST-GCN encoder + Transformer text decoder.

    - Visual encoder: STGCNEncoder
        Input:  (N, C, T, V, M)
        Output: (T_src, N, D_model)

    - Text decoder: TextTransformerDecoder
        Input:  target tokens (teacher forcing)
        Output: logits over vocabulary at each time step.
    """

    def __init__(self,
                 STGCNEncoder: nn.Module,
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.encoder = STGCNEncoder  # your STGCNEncoder instance
        assert d_model == getattr(self.encoder, "output_dim", d_model), \
            "d_model must match STGCNEncoder.output_dim"
        self.visual_pos_encoder = PositionalEncoding(d_model, dropout)
        self.decoder = TextTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.d_model = d_model

    def forward(self, x_skel, tgt_tokens, x_lengths):
        # 1) Encode
        # src_feats: (T_src, N, D_model)
        src_feats = self.encoder(x_skel)
        src_feats = src_feats * math.sqrt(self.d_model) * 5.0 
        src_feats = self.visual_pos_encoder(src_feats)
        
        # 2) Generate Mask for the Encoder Output
        # The ST-GCN reduces time roughly by factor of 4. 
        # We need to know the actual T_src output by the encoder.
        T_out = src_feats.shape[0] 
        N = src_feats.shape[1]
        if len(x_lengths) > N:
             current_lengths = x_lengths[:N]
        else:
             current_lengths = x_lengths

        valid_lengths = (current_lengths // 4).clamp(min=1)

        # Create Boolean Mask: (N, T_src)
        # True = Padding (Ignore), False = Data (Keep)
        src_key_padding_mask = torch.zeros((N, T_out), dtype=torch.bool, device=x_skel.device)

        for i, length in enumerate(valid_lengths):
            # Mask out everything AFTER the valid length
            if length < T_out:
                src_key_padding_mask[i, length:] = True

        # 3) Decode
        logits = self.decoder(
            tgt_tokens=tgt_tokens,
            memory=src_feats,
            memory_key_padding_mask=src_key_padding_mask, # <--- Pass the mask here
            # tgt_key_padding_mask is handled inside decoder usually, 
            # or you can pass one based on PAD_ID in tgt_tokens
            tgt_key_padding_mask=(tgt_tokens == 0).transpose(0, 1) # Assuming PAD_ID=0
        )

        return logits
