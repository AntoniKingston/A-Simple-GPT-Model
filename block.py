import torch
import torch.nn as nn
from torch.nn.functional import softmax
from dataclasses import dataclass

# class FeedForwardSubBlock(nn.Module):
#     def __init__(self, input_dim,  output_dim):
#         super().__init__
@dataclass
class BlockConfig:
    n_heads: int = 10
    n_embd: int = 640
    context_size: int = 512
    attention_inner_dim: int = 64
    ff_embedding_to_dim_ratio: float = 4.0
    dropout: float = 0.1


# Separate AttentionHead modules were used previously
# class AttentionHead(nn.Module):
#     def __init__(self, config: BlockConfig):
#         super().__init__()
#         self.query = nn.Linear(config.n_embd, config.attention_inner_dim, bias=False)
#         self.key = nn.Linear(config.n_embd, config.attention_inner_dim, bias=False)
#         self.value = nn.Linear(config.n_embd, config.attention_inner_dim, bias=False)
#         self.register_buffer(
#             "mask",
#             torch.tril(torch.ones(config.context_size, config.context_size))
#         )
#
#     def forward(self, x):
#         _, T, _ = x.shape
#         key = self.key(x)
#         query = self.query(x)
#         value = self.value(x)
#         attention = query @ key.transpose(-1, -2) / (key.shape[-1] ** 0.5)
#         attention = attention.masked_fill(self.mask[:T, :T] == 0, float('-inf')) # [:T, :T] so x shorter than context_size doesn't break the mask
#         attention = nn.Softmax(dim=-1)(attention)
#         return attention @ value

class MultiHeadAttentionSubBlock(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.n_embd, config.attention_inner_dim * 3 * config.n_heads, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_size, config.context_size))
        )
        self.projection = nn.Linear(config.attention_inner_dim * config.n_heads, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        query_key_value = self.qkv(x).reshape(B, T, 3, self.config.n_heads, self.config.attention_inner_dim)
        query, key, value = query_key_value.unbind(dim=2)
        # transpose from (B, T, n_heads, n_embd) to (B, n_heads, T, n_embd) fo attention computation
        query = query.transpose(-2, -3)
        key = key.transpose(-2, -3)
        value = value.transpose(-2, -3)
        attention = query @ key.transpose(-1, -2) / (self.config.attention_inner_dim ** 0.5)
        attention = attention.masked_fill(self.mask[:T, :T][None, None, :, :] == 0, float('-inf')) # [:T, :T] so x shorter than context_size doesn't break the mask
        attention = softmax(attention, dim=-1)
        attention = self.dropout(attention)
        stacked_attention = (
            (attention @ value)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, self.config.attention_inner_dim * self.config.n_heads)
        )
        projection = self.projection(stacked_attention)

        return projection

class FeedForward(nn.Module):
    def __init__(self, dim_in_out, dim_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in_out, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_in_out)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.attn = MultiHeadAttentionSubBlock(config)
        self.ln2 = nn.RMSNorm(config.n_embd)
        self.ff = FeedForward(config.n_embd, int(config.n_embd * config.ff_embedding_to_dim_ratio))
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x