import torch
import torch.nn as nn

class DigitGenerator(nn.Module):
    def __init__(self):
        super(DigitGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], dim=1)
        output = self.model(x)
        return output.view(-1, 1, 28, 28)
