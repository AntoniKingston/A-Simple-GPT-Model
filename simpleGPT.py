import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from embeddings import Embedding, SinusoidalPositionalEmbedding
from block import BlockConfig, Block
@dataclass
class SimpleGPTConfig:
    vocab_size: int = 2137
    n_blocks: int = 6
    block_config: BlockConfig = BlockConfig()

class SimpleGPT(nn.Module):
    def __init__(self, config: SimpleGPTConfig):
        super().__init__()
        self.embedding = Embedding(config.vocab_size, config.block_config.n_embd)
        self.positional_embedding = SinusoidalPositionalEmbedding(config.block_config.context_size, config.block_config.n_embd)
        self.blocks = nn.ModuleList([Block(config.block_config) for _ in range(config.n_blocks)])
        self.lm_head = nn.Linear(config.block_config.n_embd, config.vocab_size)
        self.config = config

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)

    # This function is not to be working with batches since it is to stop on generating a '<EOS>' token.
    def generate(self, idx: torch.Tensor, temperature: float = 1.0):
        new_idx = -1
        while new_idx != torch.Tensor(1): # 1 means end of sequence ('<EOS>')
            cur_idx = idx[-self.config.block_config.context_size:]
            logits = self(cur_idx)
            logits = logits / temperature
            logits = logits[-1, :]
            probs = F.softmax(logits, dim=-1)
            new_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, new_idx], dim=-1)
        return idx
