import torch

class LR(torch.nn.Module):
     def __init__(self, name, args):
        super(LR, self).__init__()
        self.args = args
        self.name = name
        self.linear = torch.nn.Linear(2, 1)
     def forward(self, x):
        return torch.sigmoid(self.linear(x))
