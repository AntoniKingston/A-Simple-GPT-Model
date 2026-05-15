from torch.utils.data import Dataset

class StrideDataset(Dataset):
    def __init__(self, data, context_size: int, stride: int = None):
        super().__init__()
        self.data = data
        self.context_size = context_size
        if stride is None:
            stride = context_size // 2
        self.stride = stride

    # Custom length is needed in order to avoid index out of range error.
    def __len__(self):
        return max(0, (len(self.data) - self.context_size) // self.stride)

    def __getitem__(self, index):
        start = index * self.stride
        x = self.data[start : start + self.context_size]
        y = self.data[start + 1 : start + self.context_size + 1]
        return x, y




