import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from embeddings import Embedding, SinusoidalPositionalEmbedding
from block import BlockConfig, Block
from typing import List
@dataclass
class SimpleGPTConfig:
    vocab_size: int = 2137
    n_blocks: int = 16
    block_config: BlockConfig = field(default_factory=BlockConfig)


class SimpleGPT(nn.Module):
    def __init__(self, config: SimpleGPTConfig):
        super().__init__()
        self.embedding = Embedding(config.vocab_size, config.block_config.n_embd)
        self.positional_embedding = SinusoidalPositionalEmbedding(config.block_config.context_size, config.block_config.n_embd)
        self.blocks = nn.ModuleList([Block(config.block_config) for _ in range(config.n_blocks)])
        self.lm_head = nn.Linear(config.block_config.n_embd, config.vocab_size)
        self.config = config

    def forward(self, x):
        # print(x.device)
        # print(self.embedding.weight.device)
        # print(self.positional_embedding.table.device)
        # print(x.shape)
        x = self.embedding(x) + self.positional_embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    # This function is not to be working with batches since it is to stop on generating a '<EOS>' token.
    def generate(self, idx: List[int], temperature: float = 1.0, max_new_tokens: int = 1000):
        idx = torch.tensor(idx, dtype=torch.long, device=self.lm_head.weight.device)
        new_idx = None
        new_tokens = 0
        while new_idx != 1:
            cur_idx = idx[max(-self.config.block_config.context_size, -len(idx)):].unsqueeze(0)
            logits = self(cur_idx)
            logits = logits / temperature
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_idx = torch.multinomial(probs, 1).squeeze(0)
            idx = torch.cat([idx, new_idx], dim=-1)
            new_tokens += 1
            if new_tokens >= max_new_tokens:
                break
        return idx[:-1].tolist()
