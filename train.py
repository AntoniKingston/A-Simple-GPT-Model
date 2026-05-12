import torch
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from datasets import OneOffDataset
from simpleGPT import SimpleGPT, SimpleGPTConfig
from dataclasses import dataclass, field
from tokenizer import bpe_tokenizer
from save_load_model import save_model

@dataclass
class TrainingPipelineConfig:
    gpt_config: SimpleGPTConfig = field(default_factory=SimpleGPTConfig)
    corpus_path: str = "data/mickiewicz_merged.txt"
    model_save_path: str = "models/mickiewicz_model.pt"
    num_merges: int = 10000



def calculate_validation_loss(model: SimpleGPT, dataloader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled = (device.type == "cuda")):
                logits = model(x)
                B, T, V = logits.shape
                batch_loss = F.cross_entropy(
                    logits.reshape(B * T, V),
                    y.reshape(B * T),
                    reduction="sum",
                )

            loss += batch_loss.item()
            total += B*T
    model.train()
    return loss / total

def train(model: SimpleGPT, dataset: OneOffDataset, config : SimpleGPTConfig):
    tr_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * config.tr_val_split), len(dataset) - int(len(dataset) * config.tr_val_split)])
    device = config.device
    print(f"Using device: {device}")
    print(config.batch_size)
    tr_dl = DataLoader(tr_dataset, batch_size=config.batch_size, pin_memory=(device == torch.device("cuda")))
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=(device == torch.device("cuda")))
    optimizer = O.Adam(model.parameters(), lr=config.lr)
    model.to(device)
    for epoch in range(config.num_epochs):
        train_loss = 0
        total = 0
        for x, y in tr_dl:
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B * T, V),
                    y.reshape(B * T),
                    reduction="sum",
                )
            train_loss += loss.item()
            total += B*T
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= total
        val_loss = calculate_validation_loss(model, val_dl, device)
        if epoch % config.info_interval == 0:
            print(f"Epoch {epoch+1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

def training_pipeline(config: TrainingPipelineConfig=TrainingPipelineConfig(), checkpoint_path=None):
    with open(config.corpus_path, "r") as f:
        text = f.read()
    encode, decode, vocab, merges = bpe_tokenizer(text, config.num_merges)
    encoded_text = torch.tensor(encode(text)).long()
    dataset = OneOffDataset(encoded_text, config.gpt_config.block_config.context_size)
    config.gpt_config.vocab_size = len(vocab)
    model = SimpleGPT(config.gpt_config)
    save_model(model, config.gpt_config, vocab, checkpoint_path.split('.')[0] + '_pre.pl', merges)
    train(model, dataset, config.gpt_config)
    save_model(model, config.gpt_config, vocab, checkpoint_path, merges)


if __name__ == "__main__":
    training_pipeline(TrainingPipelineConfig(), "models/mickiewicz_model.pt")