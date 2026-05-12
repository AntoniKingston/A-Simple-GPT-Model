import torch
import torch.nn as nn
from dataclasses import dataclass

# class FeedForwardSubBlock(nn.Module):
#     def __init__(self, input_dim,  output_dim):
#         super().__init__
@dataclass
class BlockConfig:
    n_heads: int = 6
    n_embd: int = 512
    context_size: int = 128
    attention_inner_dim: int = 128

class AttentionHead(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        self.query = nn.Linear(config.n_embd, config.attention_inner_dim)
        self.key = nn.Linear(config.n_embd, config.attention_inner_dim)
        self.value = nn.Linear(config.n_embd, config.attention_inner_dim)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_size, config.context_size))
        )

    def forward(self, x):
        _, T, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        attention = query @ key.transpose(-1, -2) / (key.shape[-1] ** 0.5)
        attention = attention.masked_fill(self.mask[:T, :T] == 0, float('-inf')) # [:T, :T] so x shorter than context_size doesn't break the mask
        attention = nn.Softmax(dim=-1)(attention)
        return attention @ value

class MultiHeadAttentionSubBlock(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_heads)])
        self.attention_output = nn.Linear(config.attention_inner_dim * config.n_heads, config.n_embd)
        self.normalization = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        attention = torch.cat([head(x) for head in self.attention_heads], dim=-1)
        attention = self.attention_output(attention)
        return self.normalization(attention + x)

class Block(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        self.attention = MultiHeadAttentionSubBlock(config)
        self.fc = nn.Sequential(nn.Linear(config.n_embd, config.n_embd), nn.ReLU())
        self.normalization = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        post_attention = self.attention(x)
        post_fc = self.fc(post_attention)
        return self.normalization(post_fc + x)