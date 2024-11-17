from torch import nn

from model.activation_func import WaveAct
from model.embedding import GBE, PatchEmbedding
from model.serialization import serialization, sort_tensor, desort_tensor


class MLP(nn.Module):
    def __init__(self, d_input, d_output, d_ff=256, num_layers=3):
        super(MLP, self).__init__()

        # Define the layers
        layers = [nn.Linear(d_input, d_ff), WaveAct()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(d_ff, d_ff))
            layers.append(WaveAct())
        if num_layers > 2:
            layers.append(nn.Linear(d_ff, d_output))
            layers.append(WaveAct())

        # Sequentially stack the layers
        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, patch_size, d_emb, num_heads):
        super(EncoderLayer, self).__init__()
        d_model = d_emb * patch_size

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.mlp = MLP(d_model, d_model)

    def forward(self, x):
        x2 = self.layer_norm_1(x)
        x = x + self.attn(x2, x2, x2)[0]

        x2 = self.layer_norm_2(x)
        x = x + self.mlp(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, patch_size, d_emb, num_layers, num_atten, num_heads):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_atten = num_atten

        self.pe = PatchEmbedding(patch_size=patch_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(patch_size, d_emb, num_heads) for _ in range(num_layers * num_atten)]
        )

    def forward(self, x, indices_all):
        # Set the sequence length of the positional encoding
        seq_len = x.shape[1]
        self.pe.set_seq_len(seq_len)

        for i in range(self.num_layers):
            indices = indices_all[i % len(indices_all)]
            x = sort_tensor(x, indices)
            x = self.pe.patching(x)

            for j in range(self.num_atten):
                x = self.layers[i * self.num_atten + j](x)

            x = self.pe.depatching(x)
            x = desort_tensor(x, indices)
        return x


import torch
import torch.nn as nn


class PinnsFormer(nn.Module):
    def __init__(self, d_output, patch_size, bc_dims, d_emb, num_layers, num_atten, num_heads, num_mlp_layers=4):
        super(PinnsFormer, self).__init__()
        # Gated Boundary Embedding layer
        self.gbe = GBE(bc_dims=bc_dims, num_expand=4, d_hidden=d_emb, d_out=d_emb)

        # Encoder only Transformer model
        self.encoder = Encoder(patch_size, d_emb, num_layers, num_atten, num_heads)
        self.mlp = nn.Sequential(
            *[
                MLP(d_emb, d_emb, d_emb * patch_size, num_mlp_layers),
                nn.LayerNorm(d_emb),
                nn.Linear(d_emb, d_output)
            ]
        )

    def forward(self, x, y, bc):
        # Serialize the input using z-order and transpose z-order
        indices_z = serialization(x, y, bc, depth=16)
        indices_zt = serialization(y, x, bc, depth=16)
        indices_all = [indices_z, indices_zt]

        # Embedding the input
        emb = self.gbe(x, y, bc)

        # Pass through the encoder and MLP
        output = self.encoder(emb, indices_all)
        output = self.mlp(output)
        return output
