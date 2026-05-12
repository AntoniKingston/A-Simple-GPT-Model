import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from embeddings import Embedding, SinusoidalPositionalEmbedding
from block import BlockConfig, Block
@dataclass
class SimpleGPTConfig:
    batch_size: int = 8
    tr_val_split: float = 0.9
    vocab_size: int = 2137
    lr: float = 1e-4
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs: int = 10
    info_interval: int = 1
    n_blocks: int = 6
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
    def generate(self, idx: torch.Tensor, new_tokens = 10, temperature: float = 1.0):
        for _ in range(new_tokens):
            cur_idx = idx[max(-self.config.block_config.context_size, -len(idx)):].unsqueeze(0)
            logits = self(cur_idx)
            logits = logits / temperature
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            new_idx = torch.multinomial(probs, 1).squeeze(0)
            idx = torch.cat([idx, new_idx], dim=-1)
        return idx
