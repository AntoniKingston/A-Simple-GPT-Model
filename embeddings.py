import torch
from torch import nn as nn
import torch.nn.parameter as parameter



class Embedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super().__init__()
        self.weight = parameter.Parameter(torch.Tensor(num_embedding, embedding_dim))
    def forward(self, x):
        return self.weight[x, :]

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, block_size, embedding_dim):
        super().__init__()
        self.table = torch.Tensor([[torch.sin(torch.tensor(pos / 10000 ** (2 * i / embedding_dim))) if (i % 2 == 0) else torch.cos(torch.tensor(pos / 10000 ** (2 * i / embedding_dim))) for i in range(embedding_dim)] for pos in range(block_size)])
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.long)
        T = x.shape[0]
        # Positional encoding is constant across in inputs, it's length just has to mach input's.
        return self.table[:T, :]
