import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k=Q.size(-1)
        scores=torch.matmul(
            Q, K.transpose(-2,-1)
        )/ math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float("-inf")
            )

        attention_weights = torch.softmax(
            scores, dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        output = torch.matmul(
            attention_weights, V
        )

        return output, attention_weights

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model          
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output, weights = self.attention(Q, K, V, mask)
        output = output.view(batch_size, -1, self.d_model)
        output= self.W_O(output)

        return output, weights



