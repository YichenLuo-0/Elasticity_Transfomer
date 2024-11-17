import torch.nn as nn


# Conditional Positional Encoding (CPE) layer
class CPE(nn.Module):
    def __init__(self, d_emb, kernel_size=3):
        super(CPE, self).__init__()
        self.conv = nn.Conv1d(d_emb, d_emb, kernel_size, padding=kernel_size // 2, groups=d_emb)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x
