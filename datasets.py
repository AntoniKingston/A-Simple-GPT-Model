from torch.utils.data import Dataset, DataLoader

class OneOffDataset(Dataset):
    def __init__(self, data, context_size):
        super().__init__()
        self.data = data
        self.context_size = context_size

    # Custom length is needed in order to avoid index out of range error.
    def __len__(self):
        return max(0, len(self.data) - self.context_size)

    def __getitem__(self, index):
        x = self.data[index : index + self.context_size]
        y = self.data[index + 1 : index + self.context_size + 1]
        return x, y




