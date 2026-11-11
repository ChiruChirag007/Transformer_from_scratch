import torch
import torch.nn as nn
from attention import MultiHeadAttention
from encoder import FeedForwardNetwork

class DecoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        #Masked Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        #Cross-Attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,encoder_output,tgt_mask=None,src_mask=None):
        # 1. Masked Self-Attention
        attentioon_output,_ = self.self_attention(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout(attentioon_output))

        # 2. Cross-Attention
        attention_output,_ = self.cross_attention(x,encoder_output,encoder_output,src_mask)
        x = self.norm2(x + self.dropout(attention_output))

        # 3. Feed Forward Network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x

class TransformerDecoder(nn.Module):
    def __init__(self,num_layers,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )

    def forward(self,x,encoder_output,tgt_mask=None,src_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x