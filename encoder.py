import torch
import torch.nn as nn
from attention import MultiHeadAttention

class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn=FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        attention_output,_=self.attention(x,x,x,mask)
        x=self.norm1(x + self.dropout(attention_output))
        ffn_output=self.ffn(x)
        x=self.norm2(x + self.dropout(ffn_output))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,num_layers,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )

    def forward(self,x,mask=None):
        for layer in self.layers:
            x=layer(x,mask)
        return x