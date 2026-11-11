import torch
import torch.nn as nn

from embeddings import InputEmbedding
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from config import *

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Source and target embeddings
        self.src_embedding = InputEmbedding(
            VOCAB_SIZE, D_MODEL, MAX_LEN
        )

        self.tgt_embedding = InputEmbedding(
            VOCAB_SIZE, D_MODEL, MAX_LEN
        )

        #Encoder
        self.encoder = TransformerEncoder(
            NUM_ENCODER_LAYERS, D_MODEL, NUM_HEADS, D_FF, DROPOUT
        )

        #Decoder
        self.decoder = TransformerDecoder(
            NUM_DECODER_LAYERS, D_MODEL, NUM_HEADS, D_FF, DROPOUT
        )

        #Output
        self.output_layer = nn.Linear(D_MODEL, VOCAB_SIZE)

    def create_masks(self, src, tgt):

        # Source padding mask
        src_mask = (src != PAD_ID).unsqueeze(1).unsqueeze(2)

        # Target padding mask
        tgt_pad_mask = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)

        seq_len = tgt.size(1)

        # Causal mask
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()

        # Combine padding and causal masks
        tgt_mask = tgt_pad_mask & causal_mask.unsqueeze(0)

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_masks(src, tgt)

        # Embeddings
        src_embedded = self.src_embedding(src)
        tgt_embedded = self.tgt_embedding(tgt)

        # Encoder
        encoder_output = self.encoder(src_embedded, src_mask)

        # Decoder
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)

        # Output layer
        logits = self.output_layer(decoder_output)

        return logits
