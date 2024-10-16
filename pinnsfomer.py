import torch
from torch import nn


class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            *[
                nn.Linear(d_model, d_ff),
                WaveAct(),
                nn.Linear(d_ff, d_ff),
                WaveAct(),
                nn.Linear(d_ff, d_model),
            ]
        )

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.act = WaveAct()
        self.ff = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x2 = self.act(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.act1 = WaveAct()
        self.attn2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.act2 = WaveAct()
        self.ff = FeedForward(d_model)

    def forward(self, x, e_outputs):
        x = x + self.attn1(x, x, x)[0]
        x2 = self.act1(x)
        x = x + self.attn2(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, heads) for _ in range(N)]
        )
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, heads) for _ in range(N)]
        )
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


class PinnsFormer(nn.Module):
    def __init__(self, d_model, d_hidden, N, heads, E, nu):
        super(PinnsFormer, self).__init__()
        self.coord_encoding = nn.Linear(2, d_model)
        self.bc_embedding = nn.Sequential(
            *[
                nn.Linear(7, d_hidden),
                WaveAct(),
                nn.Linear(d_hidden, d_hidden),
                WaveAct(),
                nn.Linear(d_hidden, d_model),
            ]
        )

        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            *[
                nn.Linear(d_model, d_hidden),
                WaveAct(),
                nn.Linear(d_hidden, d_hidden),
                WaveAct(),
                nn.Linear(d_hidden, 2),
            ]
        )

        self.E = E
        self.nu = nu
        self.G = E / (2 * (1 + nu))

    def forward(self, x, y, bc):
        coord = torch.cat([x, y], dim=-1)
        coord_src = self.coord_encoding(coord)
        bc_src = self.bc_embedding(bc)

        e_inputs = coord_src + bc_src
        e_outputs = self.encoder(e_inputs)
        d_output = self.decoder(coord_src, e_outputs)
        output = self.linear_out(d_output)

        u = output[:, :, 0:1]
        v = output[:, :, 1:2]

        # Calculate the derivatives of the displacement field
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        dv_dx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        dv_dy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        # Calculate the strain
        epsilon_x = du_dx
        epsilon_y = dv_dy
        gamma_xy = du_dy + dv_dx

        # Calculate the stress
        sigma_x = ((self.E * epsilon_x) + (self.nu * self.E * epsilon_y)) / (1 - (self.nu ** 2))
        sigma_y = ((self.E * epsilon_y) + (self.nu * self.E * epsilon_x)) / (1 - (self.nu ** 2))
        tau_xy = self.G * gamma_xy
        return u, v, epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy
