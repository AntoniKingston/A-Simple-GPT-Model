import torch
import torch.nn as nn
import torch.optim as O
import torch.nn.functional as F
from datasets import OneOffDataset
from simpleGPT import SimpleGPT

def train(model: SimpleGPT, dataset: OneOffDataset, ):