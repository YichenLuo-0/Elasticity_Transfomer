import torch
import torch.nn as nn

from model.activation_func import WaveAct


# Gated Boundary Embedding (GBE) layer
class GBE(nn.Module):
    def __init__(self, bc_dims, num_expand=2, d_hidden=256, d_out=256):
        super(GBE, self).__init__()
        self.bc_dims = bc_dims
        self.n = num_expand

        self.begin_index = []
        self.end_index = []
        for i in range(len(bc_dims)):
            self.begin_index.append(sum(bc_dims[:i]) + 1)
            self.end_index.append(sum(bc_dims[:i + 1]) + 1)

        # Dimension expansion layer
        self.liners = nn.ModuleList([nn.Linear(bc_dim, bc_dim * num_expand) for bc_dim in bc_dims])

        # Learnable embedding layer
        self.learnable_emb = nn.Sequential(*[
            nn.Linear(sum(bc_dims) * num_expand + 3, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out),
        ])

    def forward(self, x, y, bc):
        # Encode boundary conditions, expand dimensions
        bc_encode = []
        for i, liner in enumerate(self.liners):
            # Mask to expand boundary conditions
            mask = (i + 1) == bc[:, :, 0]
            mask = mask.unsqueeze(-1).repeat(1, 1, self.bc_dims[i] * self.n)

            # Only for the i-th boundary condition will be expanded, others will be masked
            bc_encoded_i = liner(bc[:, :, self.begin_index[i]:self.end_index[i]]) * mask
            bc_encode.append(bc_encoded_i)

        # Concatenate all the input features
        bc_encode = torch.cat(bc_encode, dim=-1)
        encode = torch.cat([x, y, bc[:, :, 0:1], bc_encode], dim=-1)
        return self.learnable_emb(encode)


# Patch embedding layer
class PatchEmbedding:
    def __init__(self, patch_size=4):
        self.patch_size = patch_size
        self.seq_len = 0
        self.pad_len = 0

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len
        self.pad_len = self.patch_size - (seq_len % self.patch_size)

    def patching(self, x):
        # Padding the input to make the sequence length can be divided by patch size
        batch_size, _, d_emb = x.shape
        if self.pad_len != 0 and self.pad_len != self.patch_size:
            last_input = x[:, -1, :].reshape(batch_size, 1, -1).expand(-1, self.pad_len, -1)
            x = torch.cat([x, last_input], dim=1)

        # Reshape the input to patches
        x = x.reshape(batch_size, -1, d_emb * self.patch_size)
        return x

    def depatching(self, x):
        batch_size, seq_len, d_emb = x.shape
        x = x.reshape(batch_size, -1, d_emb // self.patch_size)
        return x[..., :self.seq_len, :]