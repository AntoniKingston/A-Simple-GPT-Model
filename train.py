import torch
import torch.optim as O
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from datasets import StrideDataset
from simpleGPT import SimpleGPT
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 128
    num_epochs: int = 15
    lr: float = 1e-4
    info_interval: int = 1
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def calculate_validation_loss(model: SimpleGPT, dataloader: DataLoader, device: torch.device):
    model.eval()
    total = 0
    loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            logits = model(x)
            B, T, V = logits.shape
            batch_loss = F.cross_entropy(
                logits.reshape(B * T, V),
                y.reshape(B * T),
                reduction="mean",
            )

            loss += batch_loss.item() * B * T
            total += B*T
    model.train()
    return loss / total

def train(model: SimpleGPT, tr_dataset: StrideDataset, val_dataset: StrideDataset, config: TrainingConfig):
    device = config.device
    print(f"Using device: {device}")
    print(f"Train dataset length: {len(tr_dataset)} | Val dataset length: {len(val_dataset)}")
    tr_dl = DataLoader(tr_dataset, batch_size=config.batch_size, pin_memory=(device == torch.device("cuda")), shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=(device == torch.device("cuda")))
    optimizer = O.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-1)
    model.to(device)
    train_loss = calculate_validation_loss(model, tr_dl, device)
    val_loss = calculate_validation_loss(model, val_dl, device)
    print(f"Epoch 0 | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    for epoch in range(1, 1 + config.num_epochs):
        train_loss = 0
        total = 0
        for i, (x, y) in enumerate(tr_dl):
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                y.reshape(B * T),
                reduction="mean",
            )
            train_loss += loss.item() * B * T
            total += B*T
            loss.backward()
            optimizer.step()
            # print(f"Epoch {epoch+1} | Batch {i+1} | Train loss: {loss.item():.4f}")
        train_loss /= total
        val_loss = calculate_validation_loss(model, val_dl, device)
        if epoch % config.info_interval == 0:
            print(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")



if __name__ == "__main__":
    pass